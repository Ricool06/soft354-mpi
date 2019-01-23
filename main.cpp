#include <cmath>

#include <iostream>
#include <lodepng.h>
#include <mpi.h>
#include <Gauss.h>
#include <cmath>

const unsigned bytesInPixel = static_cast<unsigned>(sizeof(unsigned));

void generateGaussianKernel(Matrix2D<float> matrix2D, float d);

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

std::vector<std::vector<unsigned>> partitionImageIntoFragments(const std::vector<unsigned char> &originalImage,
                                                               const int fragmentCount,
                                                               const unsigned imageWidth,
                                                               const unsigned imageHeight) {
    const size_t imageSize = imageWidth * imageHeight;
    const size_t minimumFragmentSize = imageSize / fragmentCount;
    const size_t remainder = imageSize % fragmentCount;

    // Convert originalImage from byte array to pixel array
    // Each element now represents a whole 32bit pixel rather than R, G, B, or A separately
    unsigned buffer[imageSize];
    memcpy(buffer, originalImage.data(), imageSize * bytesInPixel);
    std::vector<unsigned> originalImage32Bit(buffer, buffer + imageSize);

    std::vector<std::vector<unsigned>> fragments;
    size_t pixelIndex = 0;

    for (int i = 0; i < fragmentCount; ++i) {
        // Make fragment size as even as possible
        // e.g. 5x5 image, 7 nodes, 25 / 7 = 3r4: fragment sizes = {4, 4, 4, 4, 3, 3, 3}
        const size_t fragmentSize = minimumFragmentSize + (i < remainder ? 1 : 0);

        // Cut out the fragment, with complete first and last rows
        size_t startPixelIndexToStore = pixelIndex;
        size_t indexAfterLastPixelToStore = pixelIndex + fragmentSize;

        size_t startColumn = startPixelIndexToStore % imageWidth;
        size_t endColumn = (indexAfterLastPixelToStore % imageWidth);
        auto firstPixelOfFirstRow = originalImage32Bit.begin() + startPixelIndexToStore - startColumn;
        auto lastPixelOfLastRow = originalImage32Bit.begin() + indexAfterLastPixelToStore + imageWidth - endColumn - 1;
        std::vector<unsigned> fragment(firstPixelOfFirstRow, lastPixelOfLastRow);

        // Increase starting pixel index and push the fragment onto the array
        pixelIndex += fragmentSize;
        fragments.push_back(fragment);
    }

    return fragments;
}

unsigned **allocateContiguous2DPixelArray(size_t rows, size_t columns) {
    auto *pixels = static_cast<unsigned *>(calloc(rows * columns, sizeof(unsigned)));
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
convert2DPixelArrayToImage(unsigned **pixelArray2D, const unsigned imageWidth, const unsigned imageHeight) {
    const size_t imageSize = imageWidth * imageHeight;
    const size_t imageSizeInBytes = imageSize * bytesInPixel;
    auto *buffer = static_cast<unsigned char *>(malloc(imageSizeInBytes));

    for (int i = 0; i < imageHeight; ++i) {
        memcpy(buffer + (bytesInPixel * i * imageWidth), pixelArray2D[i], bytesInPixel * imageWidth);
    }

    std::vector<unsigned char> image(buffer, buffer + (imageSize * bytesInPixel));

    return image;
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
std::vector<unsigned>
padImageWithTransparentPixels(unsigned **paddedImage, unsigned **originalImage, Padding padding, unsigned imageWidth,
                              unsigned imageHeight) {
    const unsigned newImageHeight = imageHeight + padding.top + padding.bottom;
    const unsigned newImageWidth = imageWidth + padding.left + padding.right;

    for (size_t originalRow = 0; originalRow < imageHeight; ++originalRow) {
        const size_t coreRow = padding.top + originalRow;
        unsigned *rowPointer = &(paddedImage[coreRow][padding.left]);
        unsigned *oldRowPointer = &(originalImage[originalRow][0]);
        memcpy(rowPointer, oldRowPointer, sizeof(originalImage[0]) * bytesInPixel);
    }

    const std::vector<unsigned> newWidthAndHeight = {newImageWidth, newImageHeight};
    return newWidthAndHeight;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int worldSize, worldRank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    Matrix2D<float> gaussianKernel(nullptr, 5, 5);

    // Load and
    if (worldRank == 0) {
        unsigned flowersWidth, flowersHeight;
        std::vector<unsigned char> flowers;

        unsigned error = lodepng::decode(flowers, flowersWidth, flowersHeight, "img/tiny.png");
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
        unsigned **pixelArray2D = allocateContiguous2DPixelArray(flowersHeight, flowersWidth);
        convertImageTo2DPixelArray(pixelArray2D, flowers, flowersWidth, flowersHeight);

        /**
         * Allocate memory for padded image
         *
         * ------
         * -++++-
         * -++++-
         * -++++-
         * ------
         */
        const size_t newRowCount = flowersHeight + padding.top + padding.bottom;
        const size_t newColumnCount = flowersWidth + padding.left + padding.right;
        unsigned **paddedImage = allocateContiguous2DPixelArray(newRowCount, newColumnCount);

        const std::vector<unsigned> newWidthAndHeight = padImageWithTransparentPixels(paddedImage, pixelArray2D, padding, flowersWidth, flowersHeight);
        unsigned newWidth = newWidthAndHeight[0];
        unsigned newHeight = newWidthAndHeight[1];



//        // TODO: delete after
        std::vector<unsigned char> newImageMate = convert2DPixelArrayToImage(paddedImage, newWidth, newHeight);
        error = lodepng::encode("img/pad2.png", newImageMate, newWidth, newHeight);
        if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

        // Tiling and distributing data
        int fullPaddedSize[2] = {(int) newWidth, (int) newHeight};
        const int minimumTileWidth = flowersWidth / worldSize;
        const int minimumTileHeight = flowersHeight / worldSize;
        const int widthRemainder = flowersWidth % worldSize;
        const int heightRemainder = flowersHeight % worldSize;

        int firstColumn = 0;
        int firstRow = 0;

        for (int targetRank = 0; targetRank < worldSize; ++targetRank) {
            int tileWidth = minimumTileWidth + (targetRank < widthRemainder ? 1 : 0);
            int tileHeight = minimumTileHeight + (targetRank < heightRemainder ? 1 : 0);
            tileWidth += padding.left + padding.right;
            tileHeight += padding.top + padding.bottom;

            int topLeftCorner[2] = {firstRow, firstColumn};
            int tileSize[2] = {tileWidth, tileHeight};

            // Create a 2D subarray (tile) of pixels to send
            MPI_Datatype tile;
            MPI_Type_create_subarray(2, fullPaddedSize, tileSize, topLeftCorner, MPI_ORDER_C, MPI::UNSIGNED, &tile);
            MPI_Type_commit(&tile);

            // Send tile, but don't check for request completion
            MPI_Request request;
            MPI_Isend(&(pixelArray2D[0][0]), 1, tile, targetRank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
    }
    
    // Compute!
    int sender = 0;

    int count;
    MPI_Status status;
    MPI_Probe(sender, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI::UNSIGNED, &count);

    unsigned **pixelArray2D = allocateContiguous2DPixelArray(16, 16);

    MPI_Recv(&(pixelArray2D[0][0]), count, MPI::UNSIGNED, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Finalize();
    return 0;
}

void generateGaussianKernel(Matrix2D<float> gaussianKernel, float standardDeviation) {
    float sum = 0.0f;
    float twoSigmaSquared = 2.0f * standardDeviation * standardDeviation;
    gaussianKernel.elements = static_cast<float *>(malloc(gaussianKernel.width * gaussianKernel.height * sizeof(float)));

    // Generate non-normalized kernel
    for(size_t row = 0; row < gaussianKernel.height; ++row) {
        for (size_t column = 0; column < gaussianKernel.width; ++column) {
            size_t index = row * gaussianKernel.width + column;
            size_t x = column - (gaussianKernel.width / 2);
            size_t y = row - (gaussianKernel.height / 2);
            float sumOfX2AndY2 = (x * x) + (y * y);

            gaussianKernel.elements[index] = static_cast<float>(std::exp(-sumOfX2AndY2 / twoSigmaSquared) / (M_PI * twoSigmaSquared));
            sum += gaussianKernel.elements[index];
        }
    }

    // Normalize kernel
    for (size_t i = 0; i < gaussianKernel.width * gaussianKernel.height; ++i) {
        gaussianKernel.elements[i] /= sum;
    }
}