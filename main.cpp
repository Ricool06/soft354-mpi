#include <iostream>
#include <lodepng.h>
#include <mpi.h>
#include <Gauss.h>

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
    size_t paddedPixelsAbove = gaussianKernel.height / 2;
    // Therefore we subtract 1 here if kernel width is even, and 0 if it's odd, so we don't get extra padding after.
    size_t paddedPixelsBelow = paddedPixelsAbove - ((gaussianKernel.width % 2) ^ 1);
    size_t paddedPixelsLeft = gaussianKernel.height / 2;
    size_t paddedPixelsRight = paddedPixelsLeft - ((gaussianKernel.width % 2) ^ 1);

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
    memcpy(buffer, originalImage.data(), imageSize * sizeof(unsigned));
    std::vector<unsigned> originalImage32Bit(buffer, buffer + imageSize);

    std::cout << "char size " << originalImage.size() << std::endl;
    std::cout << "int size " << originalImage32Bit.size() << std::endl;

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
        auto lastPixelOfLastRow = originalImage32Bit.begin() + indexAfterLastPixelToStore + (imageWidth - endColumn) - 1;
        std::vector<unsigned> fragment(firstPixelOfFirstRow, lastPixelOfLastRow);

        // Increase starting pixel index and push the fragment onto the array
        pixelIndex += fragmentSize;
        fragments.push_back(fragment);
    }

    return fragments;
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

        unsigned error = lodepng::decode(flowers, flowersWidth, flowersHeight, "img/hmm.png");
        if (error) {
            std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
            return 1;
        }

        std::vector<std::vector<unsigned>> fragments =
                partitionImageIntoFragments(flowers, worldSize, flowersWidth, flowersHeight);

        //TODO: deleteme
        int fragIndex = 0;
        for (const auto &fragment : fragments) {
            const auto fragmentHeight = static_cast<const unsigned int>(fragment.size() / flowersWidth);
            const size_t bufferSize = (fragment.size() * 4);
            unsigned char buffer[bufferSize];

            memcpy(buffer, fragment.data(), bufferSize);
            std::vector<unsigned char> writableFragment(buffer, buffer + bufferSize);

            error = lodepng::encode("img/frag" + std::to_string(fragIndex) + ".png", writableFragment, flowersWidth, fragmentHeight);
            if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
            fragIndex++;
        }

    }

    // Load images

//    error = lodepng::decode(tiger, tigerWidth, tigerHeight, "img/tiger.png");
//    if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
//
//    std::vector<unsigned char> greyscaleTiger = Gauss::generateGreyscaleImage(tiger, tigerWidth, tigerHeight);
//    error = lodepng::encode("img/gs_tiger.png", greyscaleTiger, tigerWidth, tigerHeight);
//    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
//
//    std::vector<unsigned char> gaussianBlurredGreyscaleTiger = Gauss::generateGaussianBlurredImage(greyscaleTiger, tigerWidth, tigerHeight, 4.0);
//    error = lodepng::encode("img/gb_gs_tiger.png", gaussianBlurredGreyscaleTiger, tigerWidth, tigerHeight);
//    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
//
//    std::vector<unsigned char> gaussianBlurredTiger = Gauss::generateGaussianBlurredImage(tiger, tigerWidth, tigerHeight, 4.0);
//    error = lodepng::encode("img/gb_tiger.png", gaussianBlurredTiger, tigerWidth, tigerHeight);
//    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
//
//    std::vector<unsigned char> gaussianBlurredFlowers = Gauss::generateGaussianBlurredImage(flowers, flowersWidth, flowersHeight, 4.0);
//    error = lodepng::encode("img/gb_flowers.png", gaussianBlurredFlowers, flowersWidth, flowersHeight);
//    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    MPI_Finalize();
    return 0;
}