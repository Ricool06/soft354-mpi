#include <gtest/gtest.h>
#include "Matrix2D.h"
#include "Matrix2D.h"
#include "Canny.h"
#include "Canny.h"

TEST(GrayscaleFilter, ShouldFilterSinglePixelImage) {
    unsigned int width = 1, height = 1;

    // Pixel with RGB mean of 100, at 20% opacity...
    std::vector<unsigned char> pixel = {50, 100, 150, 51};
    // Therefore, resulting grayscale intensity should be 20:
    std::vector<unsigned char> expectedGrayscalePixels = {20, 20, 20, 255};
    std::vector<unsigned char> actualGrayscalePixels = Canny::generateGreyscaleImage(pixel, width, height);

    EXPECT_EQ(actualGrayscalePixels[0], expectedGrayscalePixels[0]);
    EXPECT_EQ(actualGrayscalePixels[1], expectedGrayscalePixels[1]);
    EXPECT_EQ(actualGrayscalePixels[2], expectedGrayscalePixels[2]);
    EXPECT_EQ(actualGrayscalePixels[3], expectedGrayscalePixels[3]);
}

TEST(GrayscaleFilter, ShouldFilterMultiPixelImage) {
    unsigned int width = 2, height = 2;

    // 2x2 image of pixels with RGB mean of 100, at 20%, 40%, 60%, 80% opacity...
    std::vector<unsigned char> pixels = {50, 100, 150, 51,
                                         40, 100, 160, 102,
                                         30, 100, 170, 153,
                                         20, 90, 190, 204};

    // Therefore, resulting grayscale intensity matrix should be 20, 40, 60, 80:
    std::vector<unsigned char> expectedGrayscalePixels = {20, 20, 20, 255,
                                                          40, 40, 40, 255,
                                                          60, 60, 60, 255,
                                                          80, 80, 80, 255};
    std::vector<unsigned char> actualGrayscalePixels = Canny::generateGreyscaleImage(pixels, width, height);

    int i = 0;
    for (auto const& expectedRgbaValue: expectedGrayscalePixels) {
        EXPECT_EQ(actualGrayscalePixels[i], expectedRgbaValue);
        ++i;
    }
}
