#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct FlashMHAReuseQKParams {
    const half* Q;   // [B, H, N, d]
    const half* K;   // [B, H, N, d]
    const half* V;   // [B, H, N, d]
    half*       O;   // [B, H, N, d]
    int B;
    int H;
    int N;           // padded para múltiplo de 16
    int d;           // esta versão requer d == 64
    float scale;     // 1 / sqrt(d)
};

void flash_mha_reuse_qk_launch(
    const FlashMHAReuseQKParams& p,
    cudaStream_t stream = 0);

