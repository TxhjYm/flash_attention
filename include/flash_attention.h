#pragma once

// Computes S = Q * K^T.
// Q: [length, dk] row-major
// K: [length, dk] row-major
// S: [length, length] row-major
void naive_attention_qk(const float* Q, const float* K, float* S, int length,
                        int dk);
