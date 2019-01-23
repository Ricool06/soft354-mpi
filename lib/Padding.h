#ifndef SOFT354_MPI_PADDING_H
#define SOFT354_MPI_PADDING_H

#include <cstddef>

struct Padding {
    size_t top;
    size_t right;
    size_t bottom;
    size_t left;

    Padding(size_t top, size_t right, size_t bottom, size_t left): top(top), right(right), bottom(bottom), left(left) {}
};

#endif //SOFT354_MPI_PADDING_H
