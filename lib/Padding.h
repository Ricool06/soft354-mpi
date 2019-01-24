#ifndef SOFT354_MPI_PADDING_H
#define SOFT354_MPI_PADDING_H

#include <cstddef>

struct Padding {
    int top;
    int right;
    int bottom;
    int left;

    Padding(int top, int right, int bottom, int left) : top(top), right(right), bottom(bottom),
                                                                            left(left) {}
};

#endif //SOFT354_MPI_PADDING_H
