#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "flash_attention.h"
#include "utils.h"

namespace {

constexpr int kMaxLength = 128;
constexpr int kMaxHeadDim = 64;
constexpr float kAbsoluteTolerance = 1e-3f;
constexpr float kRelativeTolerance = 1e-3f;
constexpr float kInitialOutputValue = 1.25f;
constexpr int kWarmupIterations = 10;
constexpr int kTimedIterations = 100;

struct QkShape {
    int length;
    int head_dim;
};

std::ostream& operator<<(std::ostream& os, const QkShape& shape) {
    return os << "length=" << shape.length << ", head_dim=" << shape.head_dim;
}

void ComputeQkReference(const std::vector<float>& Q,
                        const std::vector<float>& K, std::vector<float>& S,
                        int length, int head_dim) {
    const float scaling_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int row = 0; row < length; ++row) {
        for (int col = 0; col < length; ++col) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                sum += Q[static_cast<std::size_t>(row) * head_dim + d] *
                       K[static_cast<std::size_t>(col) * head_dim + d];
            }
            S[static_cast<std::size_t>(row) * length + col] =
                sum * scaling_factor;
        }
    }
}

class NaiveAttentionForwardTest : public ::testing::TestWithParam<QkShape> {
protected:
    void SetUp() override {
        int device_count = 0;
        ASSERT_TRUE(check_cuda(cudaGetDeviceCount(&device_count),
                               "cudaGetDeviceCount"));
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
        ASSERT_TRUE(check_cuda(cudaSetDevice(0), "cudaSetDevice"));

        const std::size_t max_input_size =
            static_cast<std::size_t>(kMaxLength) * kMaxHeadDim;
        const std::size_t max_output_size =
            static_cast<std::size_t>(kMaxLength) * kMaxLength;
        ASSERT_TRUE(check_cuda(
            cudaMalloc(&device_q_, max_input_size * sizeof(float)),
            "cudaMalloc(device_q)"));
        ASSERT_TRUE(check_cuda(
            cudaMalloc(&device_k_, max_input_size * sizeof(float)),
            "cudaMalloc(device_k)"));
        ASSERT_TRUE(check_cuda(
            cudaMalloc(&device_s_, max_output_size * sizeof(float)),
            "cudaMalloc(device_s)"));
        ASSERT_TRUE(check_cuda(cudaEventCreate(&start_event_),
                               "cudaEventCreate(start_event)"));
        ASSERT_TRUE(check_cuda(cudaEventCreate(&stop_event_),
                               "cudaEventCreate(stop_event)"));
    }

    void TearDown() override {
        if (stop_event_ != nullptr) {
            EXPECT_TRUE(check_cuda(cudaEventDestroy(stop_event_),
                                   "cudaEventDestroy(stop_event)"));
        }
        if (start_event_ != nullptr) {
            EXPECT_TRUE(check_cuda(cudaEventDestroy(start_event_),
                                   "cudaEventDestroy(start_event)"));
        }
        if (device_s_ != nullptr) {
            EXPECT_TRUE(check_cuda(cudaFree(device_s_), "cudaFree(device_s)"));
        }
        if (device_k_ != nullptr) {
            EXPECT_TRUE(check_cuda(cudaFree(device_k_), "cudaFree(device_k)"));
        }
        if (device_q_ != nullptr) {
            EXPECT_TRUE(check_cuda(cudaFree(device_q_), "cudaFree(device_q)"));
        }
    }

    void ExpectShapeMatchesReference(const QkShape& shape) {
        SCOPED_TRACE(::testing::Message()
                     << "length=" << shape.length
                     << ", head_dim=" << shape.head_dim);

        const std::size_t input_size =
            static_cast<std::size_t>(shape.length) * shape.head_dim;
        const std::size_t output_size =
            static_cast<std::size_t>(shape.length) * shape.length;
        std::vector<float> q(input_size);
        std::vector<float> k(input_size);
        std::vector<float> actual(output_size, kInitialOutputValue);
        std::vector<float> expected(output_size, 0.0f);

        for (std::size_t i = 0; i < input_size; ++i) {
            q[i] = std::sin(0.1f * static_cast<float>(i));
            k[i] = std::cos(0.07f * static_cast<float>(i));
        }
        ComputeQkReference(q, k, expected, shape.length, shape.head_dim);

        ASSERT_TRUE(check_cuda(
            cudaMemcpy(device_q_, q.data(), input_size * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(device_q)"));
        ASSERT_TRUE(check_cuda(
            cudaMemcpy(device_k_, k.data(), input_size * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(device_k)"));
        ASSERT_TRUE(check_cuda(
            cudaMemcpy(device_s_, actual.data(), output_size * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(initial device_s)"));

        for (int i = 0; i < kWarmupIterations; ++i) {
            naive_attention_qk(device_q_, device_k_, device_s_, shape.length,
                               shape.head_dim);
            ASSERT_TRUE(check_cuda(cudaGetLastError(),
                                   "naive_attention_qk warmup launch"))
                << "warmup launch " << i;
        }
        ASSERT_TRUE(check_cuda(cudaDeviceSynchronize(),
                               "cudaDeviceSynchronize(warmup)"));
        ASSERT_TRUE(
            check_cuda(cudaEventRecord(start_event_), "cudaEventRecord(start)"));
        for (int i = 0; i < kTimedIterations; ++i) {
            naive_attention_qk(device_q_, device_k_, device_s_, shape.length,
                               shape.head_dim);
            ASSERT_TRUE(check_cuda(cudaGetLastError(),
                                   "naive_attention_qk timed launch"))
                << "timed launch " << i;
        }
        ASSERT_TRUE(
            check_cuda(cudaEventRecord(stop_event_), "cudaEventRecord(stop)"));
        ASSERT_TRUE(check_cuda(cudaEventSynchronize(stop_event_),
                               "cudaEventSynchronize(stop)"));

        float elapsed_milliseconds = 0.0f;
        ASSERT_TRUE(check_cuda(cudaEventElapsedTime(
                                   &elapsed_milliseconds, start_event_,
                                   stop_event_),
                               "cudaEventElapsedTime"));
        std::cout << "[ PERF ] naive_attention_qk (length=" << shape.length
                  << ", dk=" << shape.head_dim
                  << "): " << elapsed_milliseconds / kTimedIterations
                  << " ms/iteration (average of " << kTimedIterations
                  << " launches after " << kWarmupIterations
                  << " warmup launches)\n";

        ASSERT_TRUE(check_cuda(
            cudaMemcpy(actual.data(), device_s_, output_size * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(device_s)"));

        float max_absolute_error = 0.0f;
        for (std::size_t i = 0; i < output_size; ++i) {
            const float absolute_error =
                std::fabs(actual[i] - expected[i]);
            max_absolute_error = std::max(max_absolute_error, absolute_error);
            const float tolerance =
                kAbsoluteTolerance +
                kRelativeTolerance * std::fabs(expected[i]);
            EXPECT_LE(absolute_error, tolerance) << "output index " << i;
        }
        std::cout << "QK^T length=" << shape.length
                  << " dk=" << shape.head_dim
                  << " max absolute error=" << max_absolute_error << '\n';
    }

private:
    float* device_q_ = nullptr;
    float* device_k_ = nullptr;
    float* device_s_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;
};

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
}

TEST_P(NaiveAttentionForwardTest, QkMatchesReference) {
    ExpectShapeMatchesReference(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, NaiveAttentionForwardTest,
    ::testing::Values(QkShape{64, 8}, QkShape{70, 12}, QkShape{128, 64}),
    [](const ::testing::TestParamInfo<QkShape>& info) {
        return "length" + std::to_string(info.param.length) + "_dk" +
               std::to_string(info.param.head_dim);
    });

}  // namespace
