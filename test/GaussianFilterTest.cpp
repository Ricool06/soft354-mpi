#include <gtest/gtest.h>
#include <lodepng.h>
#include "Canny.h"

// TODO: test properly
TEST(GaussianFilter, BlursPixels) {
    unsigned int width = 1, height = 1;

    // Pixel with RGB mean of 100, at 20% opacity...
    std::vector<unsigned char> pixels = {255, 255, 255, 255};
//                                         100, 100, 100, 255};
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255,
//                                         100, 100, 100, 255};
    // Therefore, resulting grayscale intensity should be 20:
//    std::vector<unsigned char> expectedGrayscalePixels = {100, 100, 100, 100};
    std::vector<unsigned char> blurredPixels = Canny::generateGaussianBlurredImage(pixels, width, height, 0.89);

    unsigned error = lodepng::encode("/home/ricool/dev/img/MEME.png", pixels, width, height);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    error = lodepng::encode("/home/ricool/dev/img/BLURRED_MEME.png", blurredPixels, width, height);
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    printf("TestBody IN_RED: %d, OUT_RED: %d\n", pixels[0], blurredPixels[0]);

//    EXPECT_EQ(actualGrayscalePixels[0], expectedGrayscalePixels[0]);
//    EXPECT_EQ(actualGrayscalePixels[1], expectedGrayscalePixels[1]);
//    EXPECT_EQ(actualGrayscalePixels[2], expectedGrayscalePixels[2]);
//    EXPECT_EQ(actualGrayscalePixels[3], expectedGrayscalePixels[3]);
}
