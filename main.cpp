#include <cmath>

#include <iostream>
#include <lodepng.h>
#include <mpi.h>
#include <Gauss.h>
#include <cmath>

const unsigned bytesInPixel = static_cast<unsigned>(sizeof(unsigned));
const int ROOT_RANK = 0;
const int FRAGMENT_TAG = 0;
const int FRAGMENT_SIZE_TAG = 1;
const int PADDINGS_TAG = 2;

const int WIDTH_TAG = 3;

void generateGaussianKernel(Matrix2D<float> matrix2D, float d);

std::vector<unsigned char> convert1DPixelArrayToImage(unsigned int *pixelArray, int imageSize);

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
//
//std::vector<unsigned> padImageFragmentWithSharedPixels(
//        std::vector<unsigned> fragment,
//        const size_t fragmentIndex,
//        const size_t fragmentCount,
//        const std::vector<unsigned> &originalImage32Bit,
//        const size_t originalImageWidth,
//        const Padding padding) {
//    if (fragmentIndex != 0) {
//        auto
//        fragment.insert(0,)
//    }
//}
//
//std::vector<std::vector<unsigned>>
//addSharedPixelsToFragments(std::vector<std::vector<unsigned>> fragments,
//                           const std::vector<unsigned char> &originalImage,
//                           const unsigned imageWidth,
//                           const unsigned imageHeight) {
//    size_t firstPixelIndex = 0;
//
//    for (auto fragment : fragments) {
//        size_t pixelsToFinishFirstRow = firstPixelIndex % imageWidth;
//
//        size_t fragmentEndIndex = firstPixelIndex + fragment.size();
//        size_t pixelsToFinishLastRow = fragmentEndIndex % imageWidth;
//    }
//}
//
//std::vector<std::vector<unsigned>> partitionImageIntoFragments(const std::vector<unsigned char> &originalImage,
//                                                               const int fragmentCount,
//                                                               const unsigned imageWidth,
//                                                               const unsigned imageHeight) {
//    const size_t imageSize = imageWidth * imageHeight;
//    const size_t minimumFragmentSize = imageSize / fragmentCount;
//    const size_t remainder = imageSize % fragmentCount;
//
//    // Convert originalImage from byte array to pixel array
//    // Each element now represents a whole 32bit pixel rather than R, G, B, or A separately
//    unsigned buffer[imageSize];
//    memcpy(buffer, originalImage.data(), imageSize * bytesInPixel);
//    std::vector<unsigned> originalImage32Bit(buffer, buffer + imageSize);
//
//    std::vector<std::vector<unsigned>> fragments;
//    size_t pixelIndex = 0;
//
//    for (int i = 0; i < fragmentCount; ++i) {
//        // Make fragment size as even as possible
//        // e.g. 5x5 image, 7 nodes, 25 / 7 = 3r4: fragment sizes = {4, 4, 4, 4, 3, 3, 3}
//        const size_t fragmentSize = minimumFragmentSize + (i < remainder ? 1 : 0);
//
//        // Cut out the fragment, with complete first and last rows
//        size_t startPixelIndexToStore = pixelIndex;
//        size_t indexAfterLastPixelToStore = pixelIndex + fragmentSize;
//
//        size_t startColumn = startPixelIndexToStore % imageWidth;
//        size_t endColumn = (indexAfterLastPixelToStore % imageWidth);
//        auto firstPixelOfFirstRow = originalImage32Bit.begin() + startPixelIndexToStore - startColumn;
//        auto lastPixelOfLastRow = originalImage32Bit.begin() + indexAfterLastPixelToStore + imageWidth - endColumn - 1;
//        std::vector<unsigned> fragment(firstPixelOfFirstRow, lastPixelOfLastRow);
//
//        // Increase starting pixel index and push the fragment onto the array
//        pixelIndex += fragmentSize;
//        fragments.push_back(fragment);
//    }
//
//    return fragments;
//}

unsigned** allocateContiguous2DPixelArray(int rows, int columns) {
    auto *pixels = static_cast<unsigned *>(calloc((size_t) rows * columns, sizeof(unsigned)));
    auto **pixelArray2D = static_cast<unsigned **>(malloc(rows * sizeof(unsigned *)));

    for (size_t i = 0; i < rows; ++i) {
        pixelArray2D[i] = &(pixels[columns * i]);
    }

    return pixelArray2D;
}

void
convertImageTo2DPixelArray(unsigned **pixelArray2D, const std::vector<unsigned char> &originalImage,
                           const unsigned imageWidth,
                           const unsigned imageHeight) {
    const size_t imageSize = imageWidth * imageHeight;

    // Convert originalImage from byte array to pixel array
    // Each element now represents a whole 32bit pixel rather than R, G, B, or A separately
    unsigned buffer[imageSize];
    memcpy(buffer, originalImage.data(), imageSize * sizeof(int));
    std::vector<unsigned> originalImage32Bit(buffer, buffer + imageSize);

    for (size_t y = 0; y < imageHeight; ++y) {
        for (size_t x = 0; x < imageWidth; ++x) {
            pixelArray2D[y][x] = originalImage32Bit[(y * imageWidth) + x];
        }
    }
}

std::vector<unsigned char>
convert2DPixelArrayToImage(unsigned **pixelArray2D, const int imageWidth, const int imageHeight) {
    const int imageSize = imageWidth * imageHeight;
    const int imageSizeInBytes = imageSize * bytesInPixel;
    auto *buffer = static_cast<unsigned char *>(malloc((size_t) imageSizeInBytes));

    for (int i = 0; i < imageHeight; ++i) {
        memcpy(buffer + (bytesInPixel * i * imageWidth), pixelArray2D[i], bytesInPixel * imageWidth);
    }

    std::vector<unsigned char> image(buffer, buffer + (imageSize * bytesInPixel));

    return image;
}

void
convert2DPixelArrayTo1D(unsigned *pixelArray, unsigned **pixelArray2D, const int imageWidth, const int imageHeight) {
    const int imageSize = imageWidth * imageHeight;
//    const int imageSizeInBytes = imageSize * bytesInPixel;
//    auto *buffer = static_cast<unsigned*>(malloc((size_t) imageSizeInBytes));

    for (int i = 0; i < imageHeight; ++i) {
        memcpy(pixelArray + (i * imageWidth), pixelArray2D[i], bytesInPixel * imageWidth);
    }
//
//    std::vector<unsigned> image(buffer, buffer + (imageSize));
//
//    return image;
}

/**
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
 * @param paddedImage
 * @param originalImage
 * @param padding
 * @param imageWidth
 * @param imageHeight
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
        memcpy(rowPointer, oldRowPointer, imageWidth * bytesInPixel);
    }

    const std::vector<int> newWidthAndHeight = {newImageWidth, newImageHeight};

    return newWidthAndHeight;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    Matrix2D<float> gaussianKernel(nullptr, 5, 5);
    gaussianKernel.elements = static_cast<float *>(malloc(gaussianKernel.width * gaussianKernel.height * sizeof(float)));

    // Load and
    if (worldRank == ROOT_RANK) {
        unsigned originalWidth, originalHeight;
        std::vector<unsigned char> flowers;

        unsigned error = lodepng::decode(flowers, originalWidth, originalHeight, "img/hmm.png");
        if (error) {
            std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
            return 1;
        }

        generateGaussianKernel(gaussianKernel, 0.89f);

        Padding padding = getPaddingForImageFragments(gaussianKernel);

        /**
         * Allocate memory for 2D pixel array
         *
         * ++++
         * ++++
         * ++++
         */
        unsigned **pixelArray2D = allocateContiguous2DPixelArray(originalHeight, originalWidth);
        convertImageTo2DPixelArray(pixelArray2D, flowers, originalWidth, originalHeight);

        /**
         * Allocate memory for padded image
         *
         * ------
         * -++++-
         * -++++-
         * -++++-
         * ------
         */
        const int newRowCount = originalHeight + padding.top + padding.bottom;
        const int newColumnCount = originalWidth + padding.left + padding.right;
        unsigned **paddedImage = allocateContiguous2DPixelArray(newRowCount, newColumnCount);

        const std::vector<int> newWidthAndHeight = padImageWithTransparentPixels(paddedImage, pixelArray2D, padding, originalWidth, originalHeight);
        int newWidth = newWidthAndHeight[0];
        int newHeight = newWidthAndHeight[1];

        const int imageSize = newWidth * newHeight;
        auto *paddedImage1D = (unsigned*) calloc(imageSize, bytesInPixel);
        convert2DPixelArrayTo1D(paddedImage1D, paddedImage, newWidth, newHeight);

        // Distributing data
        int fullPaddedSize[1] = {newWidth * newHeight};
        printf("%d \n", fullPaddedSize[0]);
        int originalSize = originalHeight * originalWidth;
        const int minimumFragmentCoreHeight = originalHeight / worldSize;
        const int fragmentHeightRemainder = originalHeight % worldSize;

        int coreStartRowIndex = padding.top * newWidth;

        for (int targetRank = 0; targetRank < worldSize; ++targetRank) {
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
            MPI_Type_create_subarray(1, fullPaddedSize, fragmentLengths, startIndexes, MPI_ORDER_C, MPI_UNSIGNED, &fragment);
            MPI_Type_commit(&fragment);

            // Send fragment, but don't check for request completion
            MPI_Request fragmentRequest;
            MPI_Isend(paddedImage1D, 1, fragment, targetRank, FRAGMENT_TAG, MPI_COMM_WORLD, &fragmentRequest);
            MPI_Request_free(&fragmentRequest);
            MPI_Type_free(&fragment);

            coreStartRowIndex += coreFragmentHeight * newWidth;
        }
    }

    // Share Gaussian kernel
    MPI_Bcast(gaussianKernel.elements, gaussianKernel.width * gaussianKernel.height, MPI_FLOAT, ROOT_RANK, MPI_COMM_WORLD);

    // Receive padding
    int paddings[4];
    MPI_Status paddingsStatus;
    MPI_Recv(&paddings, 4, MPI_INT, ROOT_RANK, PADDINGS_TAG, MPI_COMM_WORLD, &paddingsStatus);

    // Receive width
    int width;
    MPI_Status widthStatus;
    MPI_Recv(&width, 1, MPI_INT, ROOT_RANK, WIDTH_TAG, MPI_COMM_WORLD, &widthStatus);

    // Receive my fragment
    int fragmentSize;
    MPI_Status fragmentStatus;
    MPI_Probe(ROOT_RANK, FRAGMENT_TAG, MPI_COMM_WORLD, &fragmentStatus);
    MPI_Get_count(&fragmentStatus, MPI_UNSIGNED, &fragmentSize);

    int height = fragmentSize / width;

    auto *pixelArray = (unsigned *) calloc((size_t) (fragmentSize), sizeof(unsigned));

    MPI_Recv(pixelArray, fragmentSize, MPI_UNSIGNED, ROOT_RANK, FRAGMENT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//
//    for (int i = 0; i < height; ++i) {
//        printf("{ ");
//        for (int j = 0; j < width; ++j) {
//            printf("%d ", pixelArray[(width * j) + i]);
//        }
//        printf(" }\n");
//    }

    std::vector<unsigned char> writableFragment = convert1DPixelArrayToImage(pixelArray, fragmentSize);
    // TODO: delete after
    unsigned error = lodepng::encode("img/tile" + std::to_string(worldRank) + ".png", writableFragment, (unsigned) width, (unsigned) height);
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    MPI_Finalize();
    return 0;
}

std::vector<unsigned char> convert1DPixelArrayToImage(unsigned int *pixelArray, int imageSize) {
    const int imageSizeInBytes = imageSize * bytesInPixel;
    auto *buffer = static_cast<unsigned char *>(malloc((size_t) imageSizeInBytes));

    memcpy(buffer, pixelArray, (size_t) imageSizeInBytes);

    std::vector<unsigned char> image(buffer, buffer + (imageSize * bytesInPixel));

    return image;
}

void generateGaussianKernel(Matrix2D<float> gaussianKernel, float standardDeviation) {
    float sum = 0.0f;
    float twoSigmaSquared = 2.0f * standardDeviation * standardDeviation;

    // Generate non-normalized kernel
    for(size_t row = 0; row < gaussianKernel.height; ++row) {
        for (size_t column = 0; column < gaussianKernel.width; ++column) {
            size_t index = row * gaussianKernel.width + column;
            size_t x = column - (gaussianKernel.width / 2);
            size_t y = row - (gaussianKernel.height / 2);
            float sumOfX2AndY2 = (x * x) + (y * y);

            gaussianKernel.elements[index] = (float) (std::exp(-sumOfX2AndY2 / twoSigmaSquared) / (M_PI * twoSigmaSquared));
            sum += gaussianKernel.elements[index];
        }
    }

    // Normalize kernel
    for (size_t i = 0; i < gaussianKernel.width * gaussianKernel.height; ++i) {
        gaussianKernel.elements[i] /= sum;
    }
}