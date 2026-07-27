// =============================================================================
// flash_fused.cuh — Declaração da interface pública do kernel
// =============================================================================
#pragma once

#include <cuda_fp16.h>

// Parâmetros do kernel
struct FlashFusedParams {
    const half* Q;   // [N, d] row-major FP16
    const half* K;   // [N, d] row-major FP16
    const half* V;   // [N, d] row-major FP16
    half*       O;   // [N, d] row-major FP16 (saída)
    int         N;   // sequence length (deve ser múltiplo de 16)
    int         d;   // head dimension (deve ser múltiplo de 16)
    float       scale; // 1/sqrt(d)
};

// Lança o kernel fusionado
// Grid:  (N/16) blocos
// Block: 32 threads (1 warp)
// Smem:  calculada internamente
void flash_fused_launch(const FlashFusedParams& p, cudaStream_t stream = 0);

