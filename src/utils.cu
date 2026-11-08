#include "utils.h"

#include <iostream>

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}
