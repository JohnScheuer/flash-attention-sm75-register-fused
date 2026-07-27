// =============================================================================
// wmma_layout.cuh
//
// Documentação do layout do wmma::accumulator<m16n16k16, float> no SM75.
//
// LAYOUT VERIFICADO VIA probe_layout.cu:
//
//   Thread T (lane 0..31):
//     group_id        = T >> 2   (0..7)
//     thread_in_group = T &  3   (0..3)
//
//   Fragment x[8] -> C[16][16]:
//
//     x[0] -> C[ group_id     ][ thread_in_group*2     ]
//     x[1] -> C[ group_id     ][ thread_in_group*2 + 1 ]
//     x[2] -> C[ group_id + 8 ][ thread_in_group*2     ]
//     x[3] -> C[ group_id + 8 ][ thread_in_group*2 + 1 ]
//     x[4] -> C[ group_id     ][ thread_in_group*2 + 8 ]
//     x[5] -> C[ group_id     ][ thread_in_group*2 + 9 ]
//     x[6] -> C[ group_id + 8 ][ thread_in_group*2 + 8 ]
//     x[7] -> C[ group_id + 8 ][ thread_in_group*2 + 9 ]
//
//   Propriedade crítica para o softmax:
//
//     Cada linha r de C[16][16] pertence a EXATAMENTE 4 threads consecutivas:
//       - Se r < 8:  lanes { r*4, r*4+1, r*4+2, r*4+3 }  (group_id == r)
//                   Cada uma delas tem 4 elementos da linha r: x[0],x[1],x[4],x[5]
//       - Se r >= 8: lanes { (r-8)*4, ..., (r-8)*4+3 }    (group_id == r-8)
//                   Cada uma delas tem 4 elementos da linha r: x[2],x[3],x[6],x[7]
//
//   Consequência: o shuffle para redução de linha usa apenas XOR {1, 2}.
//   Não é necessário comunicar além de 4 lanes.
//
//   Faixas de colunas por índice de fragmento:
//     thread_in_group=0 -> cols { 0, 1, 8,  9  }
//     thread_in_group=1 -> cols { 2, 3, 10, 11 }
//     thread_in_group=2 -> cols { 4, 5, 12, 13 }
//     thread_in_group=3 -> cols { 6, 7, 14, 15 }
//
// =============================================================================
#pragma once

#include <mma.h>

// Retorna o group_id do lane atual (= linha "upper" que o thread possui)
__device__ __forceinline__ int wmma_group_id() {
    return (threadIdx.x & 31) >> 2;
}

// Retorna thread_in_group do lane atual (0..3)
__device__ __forceinline__ int wmma_thread_in_group() {
    return (threadIdx.x & 31) & 3;
}

// Índices de fragmento para a "upper row" (group_id) e "lower row" (group_id+8)
//   upper: x[0], x[1], x[4], x[5]
//   lower: x[2], x[3], x[6], x[7]
static __device__ __constant__ int UPPER_IDX[4] = {0, 1, 4, 5};
static __device__ __constant__ int LOWER_IDX[4] = {2, 3, 6, 7};

