#include <cmath>
#include <iostream>
#include <mpi.h>
#include <lodepng.h>
#include <time.h>
#include <Gauss.h>

const unsigned BYTES_IN_PIXEL = static_cast<unsigned>(sizeof(unsigned));
const int ROOT_RANK = 0;
const int FRAGMENT_TAG = 0;
const int PADDINGS_TAG = 1;
const int WIDTH_TAG = 2;
clock_t startTime;
clock_t endTime;

int worldSize, worldRank;
unsigned originalFullWidth, originalFullHeight;

void generateGaussianKernel(Matrix2D<float> gaussianKernel, float standardDeviation);

std::vector<unsigned char> convert1DPixelArrayToImage(unsigned int *pixelArray, int imageSize);

void reconstructImage();

void appendArray(unsigned int *fullArray, size_t offset, unsigned int *partialArray, size_t partialArrayLength);

unsigned **allocateContiguous2DPixelArray(int rows, int columns);

Padding getPaddingForImageFragments(Matrix2D<float> gaussianKernel);

void
convertImageTo2DPixelArray(unsigned **pixelArray2D, const std::vector<unsigned char> &originalImage,
                           unsigned imageWidth, unsigned imageHeight);

void
convert2DPixelArrayTo1D(unsigned *pixelArray, unsigned **pixelArray2D, int imageWidth, int imageHeight);

std::vector<int>
cropTransparentPixels(unsigned **croppedImage, unsigned **originalImage, Padding padding, unsigned imageWidth,
                      unsigned imageHeight);

std::vector<int>
padImageWithTransparentPixels(unsigned **paddedImage, unsigned **originalImage, Padding padding, unsigned imageWidth,
                              unsigned imageHeight);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    Matrix2D<float> gaussianKernel(nullptr, 5, 5);
    gaussianKernel.elements =
            static_cast<float *>(malloc(gaussianKernel.width * gaussianKernel.height * sizeof(float)));

    // Load image and distribute image fragments across processes
    if (worldRank == ROOT_RANK) {
        std::vector<unsigned char> flowers;

        unsigned error = lodepng::decode(flowers, originalFullWidth, originalFullHeight, "img/mountains25.png");
        if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

        // Start clock
        startTime = clock();

        generateGaussianKernel(gaussianKernel, 1.0f);

        Padding padding = getPaddingForImageFragments(gaussianKernel);

        /**
         * Allocate memory for 2D pixel array
         *
         * ++++
         * ++++
         * ++++
         */
        unsigned **pixelArray2D = allocateContiguous2DPixelArray(originalFullHeight, originalFullWidth);
        convertImageTo2DPixelArray(pixelArray2D, flowers, originalFullWidth, originalFullHeight);

        /**
         * Allocate memory for padded image
         *
         * ------
         * -++++-
         * -++++-
         * -++++-
         * ------
         */
        const int newRowCount = originalFullHeight + padding.top + padding.bottom;
        const int newColumnCount = originalFullWidth + padding.left + padding.right;
        unsigned **paddedImage = allocateContiguous2DPixelArray(newRowCount, newColumnCount);

        // Pad the image with pixels based on dimensions of Gaussian kernel
        const std::vector<int> newWidthAndHeight = padImageWithTransparentPixels(paddedImage, pixelArray2D, padding,
                                                                                 originalFullWidth, originalFullHeight);
        int newWidth = newWidthAndHeight[0];
        int newHeight = newWidthAndHeight[1];

        // Convert image from 2D array (good for padding) to 1D array (good for partitioning & distribution)
        const int imageSize = newWidth * newHeight;
        auto *paddedImage1D = static_cast<unsigned *>(calloc((size_t) imageSize, BYTES_IN_PIXEL));
        convert2DPixelArrayTo1D(paddedImage1D, paddedImage, newWidth, newHeight);

        // Set up initial values needed for distribution
        int imageSizes[1] = {imageSize};
        const int minimumFragmentCoreHeight = originalFullHeight / worldSize;
        const int fragmentHeightRemainder = originalFullHeight % worldSize;

        /**
         * Sets initial starting row of image inside padding
         *
         * e.g:
         * -------
         * -------
         * S-+++--
         * --+++--
         * --+++--
         * -------
         * -------
         * S is the coreStartRowIndex
         */
        int coreStartRowIndex = padding.top * newWidth;

        // Loop through all ranks
        for (int targetRank = 0; targetRank < worldSize; ++targetRank) {
            /**
             * Spread fragment heights as evenly as possible
             * e.g:
             * Image is 25 px high
             * 7 processes are running
             * 25 / 7 = 3r4
             * min core fragment height = 3
             * remainder = 4
             * Distrubute remainder across heights:
             * { 4, 4, 4, 4, 3, 3, 3 }
             */
            int coreFragmentHeight = minimumFragmentCoreHeight + (targetRank < fragmentHeightRemainder ? 1 : 0);
            int fragmentLength = (coreFragmentHeight + padding.top + padding.bottom) * newWidth;

            int paddingArray[4] = {padding.top, padding.right, padding.bottom, padding.left};

            // Send padding
            MPI_Request paddingRequest;
            MPI_Isend(paddingArray, 4, MPI_INT, targetRank, PADDINGS_TAG, MPI_COMM_WORLD, &paddingRequest);
            MPI_Request_free(&paddingRequest);

            // Send fragment size, but don't check for request completion
            MPI_Request widthRequest;
            MPI_Isend(&newWidth, 1, MPI_INT, targetRank, WIDTH_TAG, MPI_COMM_WORLD, &widthRequest);
            MPI_Request_free(&widthRequest);

            int startIndex = coreStartRowIndex - (padding.top * newWidth);
            int fragmentLengths[1] = {fragmentLength};
            int startIndexes[1] = {startIndex};

            // Create a subarray (fragment) of pixels to send
            MPI_Datatype fragment;
            MPI_Type_create_subarray(1, imageSizes, fragmentLengths, startIndexes, MPI_ORDER_C, MPI_UNSIGNED,
                                     &fragment);
            MPI_Type_commit(&fragment);

            // Send fragment, but don't check for request completion
            MPI_Request fragmentRequest;
            MPI_Isend(paddedImage1D, 1, fragment, targetRank, FRAGMENT_TAG, MPI_COMM_WORLD, &fragmentRequest);
            MPI_Request_free(&fragmentRequest);
            MPI_Type_free(&fragment);

            coreStartRowIndex += coreFragmentHeight * newWidth;
        }
    }

    // Share Gaussian kernel from root process to all processes
    MPI_Bcast(gaussianKernel.elements, gaussianKernel.width * gaussianKernel.height, MPI_FLOAT, ROOT_RANK,
              MPI_COMM_WORLD);

    // Receive padding from root process
    int paddings[4];
    MPI_Status paddingsStatus;
    MPI_Recv(&paddings, 4, MPI_INT, ROOT_RANK, PADDINGS_TAG, MPI_COMM_WORLD, &paddingsStatus);

    // Receive width from root process
    int width;
    MPI_Status widthStatus;
    MPI_Recv(&width, 1, MPI_INT, ROOT_RANK, WIDTH_TAG, MPI_COMM_WORLD, &widthStatus);

    // Get fragment size from root process
    int fragmentSize;
    MPI_Status fragmentStatus;
    MPI_Probe(ROOT_RANK, FRAGMENT_TAG, MPI_COMM_WORLD, &fragmentStatus);
    MPI_Get_count(&fragmentStatus, MPI_UNSIGNED, &fragmentSize);

    int height = fragmentSize / width;

    // Allocate memory and receive my fragment from root process
    auto *pixelArray = static_cast<unsigned *>(calloc((size_t) (fragmentSize), BYTES_IN_PIXEL));
    MPI_Recv(pixelArray, fragmentSize, MPI_UNSIGNED, ROOT_RANK, FRAGMENT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<unsigned char> fragmentToBlur = convert1DPixelArrayToImage(pixelArray, fragmentSize);

    Padding padding(paddings[0], paddings[1], paddings[2], paddings[3]);

    // Loop within the bounds of the padding so we only blur original pixels, and use the padding to blur
    for (int row = padding.top; row < height - padding.bottom; ++row) {
        for (int column = padding.left; column < width - padding.right; ++column) {
            size_t pixelIndex = (row * BYTES_IN_PIXEL * width) + (column * BYTES_IN_PIXEL);
            float newRed = 0.0f;
            float newGreen = 0.0f;
            float newBlue = 0.0f;
            float newAlpha = 0.0f;

            // Loop through the kernel, deriving the new RGBA values based on convolution with surrounding pixels
            for (int kernelIndex = 0; kernelIndex < gaussianKernel.width * gaussianKernel.height; ++kernelIndex) {
                int kernelX = kernelIndex % gaussianKernel.width;
                int kernelY = kernelIndex / gaussianKernel.width;

                int offsetX = kernelX - (gaussianKernel.width / 2);
                int offsetY = kernelY - (gaussianKernel.height / 2);

                int blendPixelX = column + offsetX;
                int blendPixelY = row + offsetY;

                int blendPixelIndex = (blendPixelY * width) + blendPixelX;

                int startOfBlendPixel = blendPixelIndex * BYTES_IN_PIXEL;
                newRed += fragmentToBlur[startOfBlendPixel] * gaussianKernel.elements[kernelIndex];
                newGreen += fragmentToBlur[startOfBlendPixel + 1] * gaussianKernel.elements[kernelIndex];
                newBlue += fragmentToBlur[startOfBlendPixel + 2] * gaussianKernel.elements[kernelIndex];
                newAlpha += fragmentToBlur[startOfBlendPixel + 3] * gaussianKernel.elements[kernelIndex];
            }
            fragmentToBlur[pixelIndex] = static_cast<unsigned char>(newRed);
            fragmentToBlur[pixelIndex + 1] = static_cast<unsigned char>(newGreen);
            fragmentToBlur[pixelIndex + 2] = static_cast<unsigned char>(newBlue);
            fragmentToBlur[pixelIndex + 3] = static_cast<unsigned char>(newAlpha);
        }
    }

    // Perform transformations needed to crop & send fragment back to root process
    int originalHeight = height - (padding.top + padding.bottom);
    int originalWidth = width - (padding.left + padding.right);
    unsigned **pixelArray2D = allocateContiguous2DPixelArray(height, width);
    unsigned **croppedImage = allocateContiguous2DPixelArray(originalHeight, originalWidth);

    // Convert image to croppable format, the crop the padding
    convertImageTo2DPixelArray(pixelArray2D, fragmentToBlur, (unsigned) width, (unsigned) height);
    cropTransparentPixels(croppedImage, pixelArray2D, padding, (unsigned) width, (unsigned) height);

    auto *pixelArray1D = static_cast<unsigned *>(malloc(originalHeight * originalWidth * BYTES_IN_PIXEL));
    convert2DPixelArrayTo1D(pixelArray1D, croppedImage, originalWidth, originalHeight);

    MPI_Request fragmentRequest;
    MPI_Isend(pixelArray1D, originalHeight * originalWidth, MPI_UNSIGNED, ROOT_RANK, FRAGMENT_TAG, MPI_COMM_WORLD,
              &fragmentRequest);
    MPI_Request_free(&fragmentRequest);

    // Receive and reconstruct the image if this process is the root process
    if (worldRank == ROOT_RANK) {
        reconstructImage();
    }

    // Shut down MPI and end the program
    MPI_Finalize();
    return 0;
}

/**
 * Reconstructs the image after convolution by appending adjacent fragments of the image (based on source process rank)
 */
void reconstructImage() {
    size_t fullArrayLength = (size_t) originalFullHeight * originalFullWidth;
    auto *fullPixelArray = static_cast<unsigned *>(calloc(fullArrayLength, sizeof(unsigned)));

    size_t copyOffset = 0;

    for (int targetRank = 0; targetRank < worldSize; ++targetRank) {
        // Receive my fragment
        int fragmentSize;
        MPI_Status fragmentStatus;
        MPI_Probe(targetRank, FRAGMENT_TAG, MPI_COMM_WORLD, &fragmentStatus);
        MPI_Get_count(&fragmentStatus, MPI_UNSIGNED, &fragmentSize);

        auto *pixelArray = static_cast<unsigned *>(calloc((size_t) (fragmentSize), sizeof(unsigned)));

        MPI_Recv(pixelArray, fragmentSize, MPI_UNSIGNED, targetRank, FRAGMENT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        appendArray(fullPixelArray, copyOffset, pixelArray, (size_t) fragmentSize);
        copyOffset += fragmentSize;
    }

    std::vector<unsigned char> writableBlurredImage = convert1DPixelArrayToImage(fullPixelArray,
                                                                                 originalFullWidth *
                                                                                 originalFullHeight);

    // Get end time
    endTime = clock();

    unsigned error = lodepng::encode("img/gb_mountains25.png", writableBlurredImage,
                                     originalFullWidth, originalFullHeight);
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    float millisecondsTaken = ((endTime - startTime) * 1000.0f) / CLOCKS_PER_SEC;
    printf("Process without disk I/O took: %fms\n", millisecondsTaken);
}

void appendArray(unsigned int *fullArray, size_t offset, unsigned int *partialArray, size_t partialArrayLength) {
    memcpy(fullArray + offset, partialArray, partialArrayLength * BYTES_IN_PIXEL);
}

/**
 * Converts a 1 dimensional array of pixels to an image useable by lodepng
 * @param pixelArray Array to be converted
 * @param imageSize Total length of the pixel array
 * @return A vector of bytes that can be used by lodepng
 */
std::vector<unsigned char> convert1DPixelArrayToImage(unsigned int *pixelArray, int imageSize) {
    const int imageSizeInBytes = imageSize * BYTES_IN_PIXEL;
    auto *buffer = static_cast<unsigned char *>(malloc((size_t) imageSizeInBytes));

    memcpy(buffer, pixelArray, (size_t) imageSizeInBytes);

    std::vector<unsigned char> image(buffer, buffer + (imageSize * BYTES_IN_PIXEL));

    return image;
}

/**
 * Generates a Gaussian kernel to be used for convolution with the image
 * @param gaussianKernel Matrix2D containing pre-allocated array, as well as width and height of matrix to create
 * @param standardDeviation Standard deviation of the Gaussian distribution
 */
void generateGaussianKernel(Matrix2D<float> gaussianKernel, float standardDeviation) {
    float sum = 0.0f;
    float twoSigmaSquared = 2.0f * standardDeviation * standardDeviation;

    // Generate non-normalized kernel
    for (size_t row = 0; row < gaussianKernel.height; ++row) {
        for (size_t column = 0; column < gaussianKernel.width; ++column) {
            size_t index = row * gaussianKernel.width + column;
            size_t x = column - (gaussianKernel.width / 2);
            size_t y = row - (gaussianKernel.height / 2);
            float sumOfX2AndY2 = (x * x) + (y * y);

            gaussianKernel.elements[index] = static_cast<float>(std::exp(-sumOfX2AndY2 / twoSigmaSquared) /
                                                                (M_PI * twoSigmaSquared));
            sum += gaussianKernel.elements[index];
        }
    }

    // Normalize kernel
    for (size_t i = 0; i < gaussianKernel.width * gaussianKernel.height; ++i) {
        gaussianKernel.elements[i] /= sum;
    }
}

/**
 * Calculates the number of rows and columns to padded or shared on each side of a scattered image fragment,
 * so edge pixels in that fragment can still be blurred together.
 *
 * e.g. If we assume a 4x4 image is scattered between 2 nodes, with a 3x3 Gaussian kernel...
 *
 * Original pixel: +
 * Padded pixel (added by respective node): -
 * Shared pixel (added during image partitioning): =
 *
 * + + + +    Node 1: - - - - - -  Node 2: - = = = = -
 * + + + + =>         - + + + + -          - + + + + -
 * + + + +            - + + + + -          - + + + + -
 * + + + +            - = = = = -          - - - - - -
 *
 * Now every original pixel is surrounded on all sides by the required number of pixels for a 3x3 Gaussian kernel.
 *
 * @param gaussianKernel The Gaussian kernel with which an image is to be convolved
 * @return Padding data structure containing the required padding for edge pixels so they can be blurred
 */
Padding getPaddingForImageFragments(Matrix2D<float> gaussianKernel) {
    // Convention for Gaussian kernels with an even dimension is to place the extra element before the centre...
    unsigned paddedPixelsAbove = gaussianKernel.height / 2;
    // Therefore we subtract 1 here if kernel width is even, and 0 if it's odd, so we don't get extra padding after.
    unsigned paddedPixelsBelow = paddedPixelsAbove - ((gaussianKernel.width % 2) ^ 1);
    unsigned paddedPixelsLeft = gaussianKernel.height / 2;
    unsigned paddedPixelsRight = paddedPixelsLeft - ((gaussianKernel.width % 2) ^ 1);

    return Padding(paddedPixelsAbove, paddedPixelsRight, paddedPixelsBelow, paddedPixelsLeft);
}

/**
 * Allocates a 2D array, ensuring rows are contiguous so that memcpy operations are made easier
 * @param rows Number of rows of the 2D array
 * @param columns Number of columns of the 2D array
 * @return Allocated 2D array
 */
unsigned **allocateContiguous2DPixelArray(int rows, int columns) {
    auto *pixels = static_cast<unsigned *>(calloc((size_t) rows * columns, sizeof(unsigned)));
    auto **pixelArray2D = static_cast<unsigned **>(malloc(rows * sizeof(unsigned *)));

    for (size_t i = 0; i < rows; ++i) {
        pixelArray2D[i] = &(pixels[columns * i]);
    }

    return pixelArray2D;
}

/**
 * Converts an image that can be used by lodepng to a 2D pixel array, ready for padding
 * @param pixelArray2D The output 2D pixel array
 * @param originalImage The image vector loaded by lodepng
 * @param imageWidth The originalImage width in pixels
 * @param imageHeight The originalImage height in pixels
 */
void
convertImageTo2DPixelArray(unsigned **pixelArray2D, const std::vector<unsigned char> &originalImage,
                           const unsigned imageWidth,
                           const unsigned imageHeight) {
    const size_t imageSize = imageWidth * imageHeight;

    // Convert originalImage from byte array to pixel array
    // Each element now represents a whole 32bit pixel rather than R, G, B, or A separately
    auto *buffer = static_cast<unsigned *>(calloc(imageSize, BYTES_IN_PIXEL));
    memcpy(buffer, originalImage.data(), imageSize * sizeof(int));
    std::vector<unsigned> originalImage32Bit(buffer, buffer + imageSize);

    for (size_t y = 0; y < imageHeight; ++y) {
        for (size_t x = 0; x < imageWidth; ++x) {
            pixelArray2D[y][x] = originalImage32Bit[(y * imageWidth) + x];
        }
    }
}

void
convert2DPixelArrayTo1D(unsigned *pixelArray, unsigned **pixelArray2D, const int imageWidth, const int imageHeight) {
    for (int i = 0; i < imageHeight; ++i) {
        memcpy(pixelArray + (i * imageWidth), pixelArray2D[i], BYTES_IN_PIXEL * imageWidth);
    }
}

/**
 * Pads originalImage with transparent pixels specified by padding
 *
 * e.g:
 * Old image:
 * ++++
 * ++++
 * ++++
 *
 * New image:
 * ------
 * -++++-
 * -++++-
 * -++++-
 * ------
 *
 * @param paddedImage The padded image output
 * @param originalImage The image to pad
 * @param padding The dimensions of the padding to add to originalImage
 * @param imageWidth The originalImage width in pixels
 * @param imageHeight The originalImage height in pixels
 */
std::vector<int>
padImageWithTransparentPixels(unsigned **paddedImage, unsigned **originalImage, Padding padding, unsigned imageWidth,
                              unsigned imageHeight) {
    const int newImageHeight = imageHeight + padding.top + padding.bottom;
    const int newImageWidth = imageWidth + padding.left + padding.right;

    for (size_t originalRow = 0; originalRow < imageHeight; ++originalRow) {
        const size_t coreRow = padding.top + originalRow;
        unsigned *rowPointer = &(paddedImage[coreRow][padding.left]);
        unsigned *oldRowPointer = &(originalImage[originalRow][0]);
        memcpy(rowPointer, oldRowPointer, imageWidth * BYTES_IN_PIXEL);
    }

    const std::vector<int> newWidthAndHeight = {newImageWidth, newImageHeight};

    return newWidthAndHeight;
}

/**
 * Fills in croppedImage with pixels from originalImage minus the padding pixels signified by padding param;
 *
 * e.g:
 * Old image:
 * ------
 * -++++-
 * -++++-
 * -++++-
 * ------
 *
 * New image:
 * ++++
 * ++++
 * ++++
 *
 * @param croppedImage Pre-allocated 2D pixel array to store cropped image.
 * @param originalImage 2D pixel array of the original, padded image.
 * @param padding The padding to be removed from originalImage
 * @param imageWidth Current image width
 * @param imageHeight Current image height
 * @return vector containing the new height and width of the image
 */
std::vector<int>
cropTransparentPixels(unsigned **croppedImage, unsigned **originalImage, Padding padding, unsigned imageWidth,
                      unsigned imageHeight) {
    const int newImageHeight = imageHeight - (padding.top + padding.bottom);
    const int newImageWidth = imageWidth - (padding.left + padding.right);

    for (size_t croppedRow = 0; croppedRow < newImageHeight; ++croppedRow) {
        const size_t coreRow = padding.top + croppedRow;
        unsigned *rowPointer = &(croppedImage[croppedRow][0]);
        unsigned *oldRowPointer = &(originalImage[coreRow][padding.left]);
        memcpy(rowPointer, oldRowPointer, newImageWidth * BYTES_IN_PIXEL);
    }

    const std::vector<int> newWidthAndHeight = {newImageWidth, newImageHeight};

    return newWidthAndHeight;
}
