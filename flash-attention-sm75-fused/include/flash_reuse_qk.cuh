#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct FlashReuseQKParams {
    const half* Q;   // [N, d]
    const half* K;   // [N, d]
    const half* V;   // [N, d]
    half*       O;   // [N, d]
    int         N;   // múltiplo de 16 (ou padded)
    int         d;   // ESTA VERSÃO REQUER d == 64
    float       scale;
};

void flash_reuse_qk_launch(
    const FlashReuseQKParams& p,
    cudaStream_t stream = 0);

