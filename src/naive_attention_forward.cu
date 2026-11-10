#include "flash_attention.h"

// 朴素 attention forward 的第一步：S = Q * K^T。
//
// Q: [length, dk]，K: [length, dk]，S: [length, length]，均为行主序。
//
// 这里把 GEMM 的 register tiling 套到 attention 上，对应关系是：
//
//   GEMM:      C[rows, cols] = A[rows, reduction] * B[reduction, cols]
//   attention: S[length, length] = Q[length, dk] * K^T[dk, length]
//
// 对应到索引：
//   Q      读 Q[row * dk + d]
//   K^T    不真正转置 K，直接读 K[col * dk + d]（K 的第 col 行第 d 列）
//   S      写 S[row * length + col]
//   归约维是 dk
//
// 每个 block 计算 S 的一个 64x64 tile，每个线程计算 4x4 个输出。
// 共享内存 tile 为 Qs[64][8] 和 Ks[8][64]：
//   - 64 是序列长度方向的 tile
//   - 8 是 head dimension 方向的 tile
// dk 不是 8 的倍数、length 不是 64 的倍数时，越界元素补 0。
//
// TODO: softmax 与 S * V 两步。

namespace {

constexpr int lengthPerBlock = 64;  // 序列长度方向的 tile
constexpr int headDimPerBlock = 8;  // head dimension 方向的 tile
constexpr int lengthPerThread = 4;  // 每个线程计算 4x4 个输出
constexpr int numThreadsInBlock = (lengthPerBlock / lengthPerThread) *
                                  (lengthPerBlock / lengthPerThread);  // 256

}  // namespace

__global__ void naive_attention_qk_kernel(const float* Q, const float* K,
                                          float* S, int length, int dk) {
    const int tileRow = blockIdx.y;  // S 的行方向 tile
    const int tileCol = blockIdx.x;  // S 的列方向 tile

    // 逻辑线程网格：lengthPerBlock / lengthPerThread = 16 行 × 16 列 = 256 个线程
    const int numThreadsInLength = lengthPerBlock / lengthPerThread;  // 16
    const int threadRow = threadIdx.x / numThreadsInLength;  // 0~15
    const int threadCol = threadIdx.x % numThreadsInLength;  // 0~15

    // 本线程加载 Qs[lengthPerBlock][headDimPerBlock] 的起始索引，
    // 循环中再累加 offset：每轮 32 行，步长 32，共 2 轮
    const int qTileRowStart = threadIdx.x / headDimPerBlock;  // 0~31
    const int qTileHeadStart = threadIdx.x % headDimPerBlock;  // 0~7
    constexpr int qTileRowStride = numThreadsInBlock / headDimPerBlock;  // 32

    // 本线程加载 Ks[headDimPerBlock][lengthPerBlock] 的起始索引，
    // 循环中再累加 offset：每轮 4 行，步长 4，共 2 轮
    const int kTileHeadStart = threadIdx.x / lengthPerBlock;  // 0~3
    const int kTileColStart = threadIdx.x % lengthPerBlock;  // 0~63
    constexpr int kTileHeadStride = numThreadsInBlock / lengthPerBlock;  // 4

    __shared__ float Qs[lengthPerBlock][headDimPerBlock];
    __shared__ float Ks[headDimPerBlock][lengthPerBlock];

    float sAccum[lengthPerThread * lengthPerThread] = {0.0f};
    float regQ[lengthPerThread] = {0.0f};
    float regK[lengthPerThread] = {0.0f};
    float scalingFactor = 1.0f / sqrtf((float)dk);

    for (int headTile = 0;
         headTile < (dk + headDimPerBlock - 1) / headDimPerBlock; headTile++) {
        // 协作加载 Q tile：Qs[localRow][d] = Q[globalRow][globalHead]
        #pragma unroll
        for (int offset = 0; offset < lengthPerBlock; offset += qTileRowStride) {
            int localRow = qTileRowStart + offset;
            int globalRow = tileRow * lengthPerBlock + localRow;
            int globalHead = headTile * headDimPerBlock + qTileHeadStart;
            if (globalRow < length && globalHead < dk)
                Qs[localRow][qTileHeadStart] = Q[globalRow * dk + globalHead];
            else
                Qs[localRow][qTileHeadStart] = 0.0f;
        }

        // 协作加载 K^T tile：Ks[d][localCol] = K[globalCol][globalHead]
        #pragma unroll
        for (int offset = 0; offset < headDimPerBlock;
             offset += kTileHeadStride) {
            int localHead = kTileHeadStart + offset;
            int globalHead = headTile * headDimPerBlock + localHead;
            int globalCol = tileCol * lengthPerBlock + kTileColStart;
            if (globalCol < length && globalHead < dk)
                Ks[localHead][kTileColStart] = K[globalCol * dk + globalHead];
            else
                Ks[localHead][kTileColStart] = 0.0f;
        }

        __syncthreads();

        // 计算：外积累加
        #pragma unroll
        for (int headIdx = 0; headIdx < headDimPerBlock; headIdx++) {
            for (int i = 0; i < lengthPerThread; i++)
                regQ[i] = Qs[threadRow * lengthPerThread + i][headIdx];
            for (int i = 0; i < lengthPerThread; i++)
                regK[i] = Ks[headIdx][threadCol * lengthPerThread + i];
            for (int resIdxRow = 0; resIdxRow < lengthPerThread; resIdxRow++)
                for (int resIdxCol = 0; resIdxCol < lengthPerThread; resIdxCol++)
                    sAccum[resIdxRow * lengthPerThread + resIdxCol] +=
                        regQ[resIdxRow] * regK[resIdxCol];
        }

        __syncthreads();
    }

    // 写回 S
    for (int resIdxRow = 0; resIdxRow < lengthPerThread; resIdxRow++) {
        for (int resIdxCol = 0; resIdxCol < lengthPerThread; resIdxCol++) {
            int globalRow = tileRow * lengthPerBlock +
                            threadRow * lengthPerThread + resIdxRow;
            int globalCol = tileCol * lengthPerBlock +
                            threadCol * lengthPerThread + resIdxCol;
            if (globalRow < length && globalCol < length)
                S[globalRow * length + globalCol] =
                    sAccum[resIdxRow * lengthPerThread + resIdxCol] * scalingFactor;
        }
    }
}

void naive_attention_qk(const float* Q, const float* K, float* S, int length,
                        int dk) {
    dim3 block(numThreadsInBlock, 1);
    dim3 grid((length + lengthPerBlock - 1) / lengthPerBlock,
              (length + lengthPerBlock - 1) / lengthPerBlock);
    naive_attention_qk_kernel<<<grid, block>>>(Q, K, S, length, dk);
}
