#pragma once

#include <mma.h>
#include <cfloat>
#include "warp_softmax.cuh"

using namespace nvcuda;

// ============================================================================
// Estado online do softmax
// ============================================================================
struct OnlineState {
    float m[2];  // running max    (upper, lower)
    float l[2];  // running sum    (upper, lower)
};

struct OnlineStep {
    float alpha_upper;
    float alpha_lower;
    float beta_upper;
    float beta_lower;
};

__device__ __forceinline__
void online_state_init(OnlineState& s) {
    s.m[0] = -FLT_MAX;
    s.m[1] = -FLT_MAX;
    s.l[0] = 0.0f;
    s.l[1] = 0.0f;
}

// ============================================================================
// Calcula alpha/beta UMA vez por tile de scores e atualiza o estado.
// Isso permite reutilizar o mesmo step para múltiplos O_acc[d_out].
// ============================================================================
__device__ __forceinline__
OnlineStep online_step_update(
    OnlineState& state,
    const RowMaxSum& ms)
{
    OnlineStep step;

    // upper
    {
        float m_new = fmaxf(state.m[0], ms.row_max[0]);
        step.alpha_upper = __expf(state.m[0] - m_new);
        step.beta_upper  = __expf(ms.row_max[0] - m_new);

        state.l[0] = step.alpha_upper * state.l[0]
                   + step.beta_upper  * ms.row_sum[0];
        state.m[0] = m_new;
    }

    // lower
    {
        float m_new = fmaxf(state.m[1], ms.row_max[1]);
        step.alpha_lower = __expf(state.m[1] - m_new);
        step.beta_lower  = __expf(ms.row_max[1] - m_new);

        state.l[1] = step.alpha_lower * state.l[1]
                   + step.beta_lower  * ms.row_sum[1];
        state.m[1] = m_new;
    }

    return step;
}

// ============================================================================
// Aplica alpha ao acumulador O existente
// ============================================================================
__device__ __forceinline__
void apply_alpha_to_O(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& O_acc,
    const OnlineStep& s)
{
    O_acc.x[0] *= s.alpha_upper;
    O_acc.x[1] *= s.alpha_upper;
    O_acc.x[4] *= s.alpha_upper;
    O_acc.x[5] *= s.alpha_upper;

    O_acc.x[2] *= s.alpha_lower;
    O_acc.x[3] *= s.alpha_lower;
    O_acc.x[6] *= s.alpha_lower;
    O_acc.x[7] *= s.alpha_lower;
}

// ============================================================================
// Aplica beta ao tile atual de scores já em exp(x-max)
// ============================================================================
__device__ __forceinline__
void apply_beta_to_S(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    const OnlineStep& s)
{
    S_acc.x[0] *= s.beta_upper;
    S_acc.x[1] *= s.beta_upper;
    S_acc.x[4] *= s.beta_upper;
    S_acc.x[5] *= s.beta_upper;

    S_acc.x[2] *= s.beta_lower;
    S_acc.x[3] *= s.beta_lower;
    S_acc.x[6] *= s.beta_lower;
    S_acc.x[7] *= s.beta_lower;
}

// ============================================================================
// Divide O_acc pela soma final l
// ============================================================================
__device__ __forceinline__
void finalize_O(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& O_acc,
    const OnlineState& state)
{
    float inv_u = 1.0f / state.l[0];
    float inv_l = 1.0f / state.l[1];

    O_acc.x[0] *= inv_u;
    O_acc.x[1] *= inv_u;
    O_acc.x[4] *= inv_u;
    O_acc.x[5] *= inv_u;

    O_acc.x[2] *= inv_l;
    O_acc.x[3] *= inv_l;
    O_acc.x[6] *= inv_l;
    O_acc.x[7] *= inv_l;
}

