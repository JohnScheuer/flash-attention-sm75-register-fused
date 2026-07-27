// =============================================================================
// online_state.cuh
//
// Estado do online softmax (FlashAttention Algoritmo 1).
//
// Cada thread mantém estado para 2 linhas (upper e lower).
// Todas as operações de rescaling são feitas diretamente nos registradores
// do wmma::accumulator, sem nenhum acesso a shared memory.
//
// Invariante do loop:
//   O_acc contém:  sum_{t'<t} exp(m_{t'} - m_t) * exp(S_{t'}[r,:] - m_{t'}) @ V_{t'}
//                = sum_{t'<t} exp(S_{t'}[r,:] - m_t) @ V_{t'}
//
//   state.l[r] contém:  sum_{t'<=t} exp(m_{t'} - m_t) * rowsum_{t'}
//
// Ao final, finalize_O divide O_acc pela soma total state.l.
// =============================================================================
#pragma once

#include <mma.h>
#include <cfloat>
#include "warp_softmax.cuh"

using namespace nvcuda;

// -----------------------------------------------------------------------------
struct OnlineState {
    float m[2];  // running max  (upper, lower)
    float l[2];  // running sum  (upper, lower)
};

__device__ __forceinline__
void online_state_init(OnlineState& s) {
    s.m[0] = -FLT_MAX;
    s.m[1] = -FLT_MAX;
    s.l[0] = 0.0f;
    s.l[1] = 0.0f;
}

// -----------------------------------------------------------------------------
// online_update:
//
//   Dado o resultado do softmax unnorm do tile atual (ms),
//   atualiza O_acc e o estado online.
//
//   Retorna os fatores beta (para escalar S_acc antes do WMMA P@V).
//
//   Sequência exata (derivada do paper FlashAttention-2, Eq. 4):
//
//     m_new = max(m_old, m_cur)
//     alpha  = exp(m_old - m_new)        ← fator para reescalar O_old
//     beta   = exp(m_cur - m_new)        ← fator para escalar P_cur antes de P@V
//     l_new  = alpha * l_old + beta * l_cur
//
//     O_new  = alpha * O_old + beta * P_cur @ V_cur
//            ←─── alpha aplicado agora em O_acc
//            ←─── beta multiplicado em S_acc ANTES do wmma_sync
// -----------------------------------------------------------------------------
struct BetaFactors {
    float upper;
    float lower;
};

__device__ __forceinline__
BetaFactors online_update(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& O_acc,
    OnlineState& state,
    const RowMaxSum& ms)
{
    BetaFactors beta;

    // ── Upper row ────────────────────────────────────────────────────────────
    {
        float m_new = fmaxf(state.m[0], ms.row_max[0]);
        float alpha  = __expf(state.m[0] - m_new);
        beta.upper   = __expf(ms.row_max[0] - m_new);

        // Reescalar O_acc para as 4 posições da upper row
        O_acc.x[0] *= alpha;
        O_acc.x[1] *= alpha;
        O_acc.x[4] *= alpha;
        O_acc.x[5] *= alpha;

        // Atualizar soma e máximo acumulados
        state.l[0] = alpha * state.l[0] + beta.upper * ms.row_sum[0];
        state.m[0] = m_new;
    }

    // ── Lower row ────────────────────────────────────────────────────────────
    {
        float m_new = fmaxf(state.m[1], ms.row_max[1]);
        float alpha  = __expf(state.m[1] - m_new);
        beta.lower   = __expf(ms.row_max[1] - m_new);

        O_acc.x[2] *= alpha;
        O_acc.x[3] *= alpha;
        O_acc.x[6] *= alpha;
        O_acc.x[7] *= alpha;

        state.l[1] = alpha * state.l[1] + beta.lower * ms.row_sum[1];
        state.m[1] = m_new;
    }

    return beta;
}

// -----------------------------------------------------------------------------
// apply_beta_to_S:
//
//   Multiplica S_acc (= P unnorm) pelos fatores beta por linha.
//   Isso garante que quando wmma_sync acumula em O_acc,
//   o valor adicionado é (beta * P_unnorm) @ V = exp(S - m_new) @ V.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void apply_beta_to_S(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    const BetaFactors& beta)
{
    S_acc.x[0] *= beta.upper;
    S_acc.x[1] *= beta.upper;
    S_acc.x[4] *= beta.upper;
    S_acc.x[5] *= beta.upper;

    S_acc.x[2] *= beta.lower;
    S_acc.x[3] *= beta.lower;
    S_acc.x[6] *= beta.lower;
    S_acc.x[7] *= beta.lower;
}

// -----------------------------------------------------------------------------
// finalize_O:
//
//   Divide O_acc pela soma total acumulada.
//   Chamado UMA VEZ após todos os KV tiles.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void finalize_O(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& O_acc,
    const OnlineState& state)
{
    float inv_u = 1.0f / state.l[0];
    float inv_l = 1.0f / state.l[1];

    O_acc.x[0] *= inv_u;  O_acc.x[1] *= inv_u;
    O_acc.x[4] *= inv_u;  O_acc.x[5] *= inv_u;

    O_acc.x[2] *= inv_l;  O_acc.x[3] *= inv_l;
    O_acc.x[6] *= inv_l;  O_acc.x[7] *= inv_l;
}

