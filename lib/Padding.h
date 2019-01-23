#ifndef SOFT354_MPI_PADDING_H
#define SOFT354_MPI_PADDING_H

#include <cstddef>

struct Padding {
    unsigned top;
    unsigned right;
    unsigned bottom;
    unsigned left;

    Padding(unsigned top, unsigned right, unsigned bottom, unsigned left) : top(top), right(right), bottom(bottom),
                                                                            left(left) {}
};

#endif //SOFT354_MPI_PADDING_H
