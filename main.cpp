#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Hello, World!" << std::endl;
    #pragma omp parallel for
    for(int i=0;i<10;i++){
        printf("%i\n",i);
    }
    return 0;
}