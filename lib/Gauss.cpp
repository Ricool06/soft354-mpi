#include <cstdlib>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <mpi.h>
#include "Gauss.h"

namespace Gauss {
    void populateGaussianKernel(Matrix2D<float> gaussianKernel) {
    }

    std::vector<unsigned char> generateGreyscaleImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight) {
        return {0};
    }

    std::vector<unsigned char> generateGaussianBlurredImage(const std::vector<unsigned char> &originalImage, unsigned int pixelWidth, unsigned int pixelHeight, float standardDeviation) {
        return {0};
    }

}
