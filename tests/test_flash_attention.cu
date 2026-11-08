#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

#include "utils.h"

// TODO: as the kernels are implemented, add correctness tests here:
//   - naive forward / backward vs. CPU reference
//   - standard vs. online softmax (including numerical edge cases)
//   - FlashAttention-1 / FlashAttention-2 forward / backward
//   - causal mask on non-square sequence lengths
//   - shapes that are not multiples of the tile sizes
// Each kernel test should also cover a non-multiple-of-tile shape, mirroring
// the M=512, N=624, K=129 style used in gemm_cuda.

TEST(FlashAttentionTest, CudaDeviceIsAvailable) {
    int device_count = 0;
    ASSERT_TRUE(check_cuda(cudaGetDeviceCount(&device_count),
                           "cudaGetDeviceCount"));

    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    cudaDeviceProp properties{};
    ASSERT_TRUE(check_cuda(cudaGetDeviceProperties(&properties, 0),
                           "cudaGetDeviceProperties"));

    std::cout << "CUDA device: " << properties.name << " (sm_"
              << properties.major << properties.minor << ", "
              << properties.totalGlobalMem / (1024 * 1024) << " MiB)\n";

    EXPECT_GT(device_count, 0);
}
