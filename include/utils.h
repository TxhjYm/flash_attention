#pragma once

#include <cuda_runtime.h>

// Returns true if status == cudaSuccess. Otherwise prints the failed operation
// and the CUDA error string to stderr and returns false.
bool check_cuda(cudaError_t status, const char* operation);
