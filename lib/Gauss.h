#ifndef SOFT354_CUDA_GAUSS_H
#define SOFT354_CUDA_GAUSS_H

#include <vector>
#include "Matrix2D.h"
#include "Padding.h"

namespace Gauss {
    std::vector<unsigned char> generateGreyscaleImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight);

    std::vector<unsigned char> generateGaussianBlurredImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight, float standardDeviation);
}

#endif //SOFT354_CUDA_GAUSS_H
