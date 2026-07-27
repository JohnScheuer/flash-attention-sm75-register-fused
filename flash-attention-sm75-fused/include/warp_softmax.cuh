// =============================================================================
// warp_softmax.cuh
//
// Softmax completamente em registradores sobre um wmma::accumulator<m16n16k16>.
//
// Duas variantes:
//   1. warp_softmax_normalized   — divide pelo sum (para uso standalone)
//   2. warp_softmax_unnorm       — NÃO divide (para uso com online rescaling)
//
// Ambas retornam RowMaxSum com os valores de max e sum por linha,
// necessários para o online rescaling do FlashAttention.
//
// Garante: zero acesso a shared memory.
// =============================================================================
#pragma once

#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>

using namespace nvcuda;

// -----------------------------------------------------------------------------
// Estrutura de retorno
// -----------------------------------------------------------------------------
struct RowMaxSum {
    float row_max[2];  // [0] = upper rows (group_id),  [1] = lower rows (group_id+8)
    float row_sum[2];  // idem
};

// -----------------------------------------------------------------------------
// Helpers internos: redução dentro de grupo de 4 threads via shuffle
// -----------------------------------------------------------------------------

// Redução de máximo dentro das 4 threads que compartilham uma linha.
// As 4 threads têm lanes consecutivas (group_id*4 ... group_id*4+3),
// portanto XOR com 1 e XOR com 2 é suficiente.
__device__ __forceinline__
float warp_row_max(float val) {
    float tmp;
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    val = fmaxf(val, tmp);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val = fmaxf(val, tmp);
    return val;  // todas as 4 threads recebem o mesmo resultado
}

// Redução de soma dentro das 4 threads que compartilham uma linha.
__device__ __forceinline__
float warp_row_sum(float val) {
    float tmp;
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    val += tmp;
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += tmp;
    return val;
}

// -----------------------------------------------------------------------------
// Softmax UNNORMALIZADO (para FlashAttention com online rescaling)
//
// Pós-condição:
//   S_acc.x[i] = exp(S_orig.x[i] * scale - row_max)
//   Retorna row_max e row_sum (= soma dos exp acima).
//   NÃO divide pelo sum — isso é feito em finalize_O().
// -----------------------------------------------------------------------------
__device__ __forceinline__
RowMaxSum warp_softmax_unnorm(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    float scale)
{
    RowMaxSum result;

    // ── Passo 1: aplicar escala ──────────────────────────────────────────────
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        S_acc.x[i] *= scale;
    }

    // ── Passo 2: máximo local por thread, separado para upper/lower rows ────
    //   upper (x[0],x[1],x[4],x[5]) → linha group_id
    //   lower (x[2],x[3],x[6],x[7]) → linha group_id+8
    float lm_u = fmaxf(fmaxf(S_acc.x[0], S_acc.x[1]),
                       fmaxf(S_acc.x[4], S_acc.x[5]));
    float lm_l = fmaxf(fmaxf(S_acc.x[2], S_acc.x[3]),
                       fmaxf(S_acc.x[6], S_acc.x[7]));

    // ── Passo 3: redução de máximo pelas 4 threads que compartilham a linha ─
    float gm_u = warp_row_max(lm_u);
    float gm_l = warp_row_max(lm_l);

    result.row_max[0] = gm_u;
    result.row_max[1] = gm_l;

    // ── Passo 4: exp(x - max) in-place ──────────────────────────────────────
    S_acc.x[0] = __expf(S_acc.x[0] - gm_u);
    S_acc.x[1] = __expf(S_acc.x[1] - gm_u);
    S_acc.x[4] = __expf(S_acc.x[4] - gm_u);
    S_acc.x[5] = __expf(S_acc.x[5] - gm_u);

    S_acc.x[2] = __expf(S_acc.x[2] - gm_l);
    S_acc.x[3] = __expf(S_acc.x[3] - gm_l);
    S_acc.x[6] = __expf(S_acc.x[6] - gm_l);
    S_acc.x[7] = __expf(S_acc.x[7] - gm_l);

    // ── Passo 5: soma local ──────────────────────────────────────────────────
    float ls_u = S_acc.x[0] + S_acc.x[1] + S_acc.x[4] + S_acc.x[5];
    float ls_l = S_acc.x[2] + S_acc.x[3] + S_acc.x[6] + S_acc.x[7];

    // ── Passo 6: redução de soma ─────────────────────────────────────────────
    float gs_u = warp_row_sum(ls_u);
    float gs_l = warp_row_sum(ls_l);

    result.row_sum[0] = gs_u;
    result.row_sum[1] = gs_l;

    return result;
}

// -----------------------------------------------------------------------------
// Softmax NORMALIZADO (para uso standalone / validação)
// -----------------------------------------------------------------------------
__device__ __forceinline__
RowMaxSum warp_softmax_normalized(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    float scale)
{
    RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

    float inv_u = 1.0f / ms.row_sum[0];
    float inv_l = 1.0f / ms.row_sum[1];

    S_acc.x[0] *= inv_u;  S_acc.x[1] *= inv_u;
    S_acc.x[4] *= inv_u;  S_acc.x[5] *= inv_u;

    S_acc.x[2] *= inv_l;  S_acc.x[3] *= inv_l;
    S_acc.x[6] *= inv_l;  S_acc.x[7] *= inv_l;

    return ms;
}

