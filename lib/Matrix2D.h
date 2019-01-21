#ifndef SOFT354_CUDA_MATRIX_H
#define SOFT354_CUDA_MATRIX_H

#include <cstddef>

template <typename T>
struct Matrix2D {
    T* elements;
    size_t width;
    size_t height;

    Matrix2D(T* elements, size_t width, size_t height): width(width), height(height), elements(elements) {}
};


#endif //SOFT354_CUDA_MATRIX_H
