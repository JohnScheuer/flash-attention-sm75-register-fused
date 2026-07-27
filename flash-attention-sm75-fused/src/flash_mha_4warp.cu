#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "wmma_layout.cuh"
#include "warp_softmax.cuh"
#include "online_step.cuh"

using namespace nvcuda;

// ============================================================================
// Constantes
// ============================================================================
static constexpr int TM       = 16;
static constexpr int TN       = 16;
static constexpr int TK       = 16;
static constexpr int D_FIXED  = 64;
static constexpr int D_TILES  = D_FIXED / TK;   // 4
static constexpr int N_WARPS  = 4;               // warps por bloco
static constexpr int WARP_SZ  = 32;
static constexpr int BLOCK_SZ = N_WARPS * WARP_SZ; // 128 threads

// ============================================================================
// Shared memory layout (por bloco de 4 warps)
//
//  Região Q  : N_WARPS × TM × D_FIXED halfs = 4 × 16 × 64 = 4096 B
//              Cada warp tem seu próprio tile de Q pré-carregado.
//
//  Região K  : TN × D_FIXED halfs = 16 × 64 = 2048 B
//              Compartilhado entre todos os warps.
//              Carregado UMA vez por kv_tile.
//
//  Região V  : TN × TK halfs = 16 × 16 = 512 B
//              Compartilhado. Recarregado por d_out.
//              (reutilizado no mesmo slot para cada d_out)
//
//  Região P  : N_WARPS × TM × TN halfs = 4 × 16 × 16 = 2048 B
//              Cada warp tem seu próprio tile de P.
//              Slot privado para o scatter do accumulator.
//
//  Total     : 4096 + 2048 + 512 + 2048 = 8704 B ≈ 8.5 KB
//
//  Com 8704 B/bloco e 65536 B/SM → max 7 blocos/SM.
//  Com 255 regs/thread × 128 threads = 32640 regs/bloco → max 2 blocos/SM.
//
//  NOTA: o gargalo aqui ainda é register pressure (255 regs/thread herdado).
//  O ganho vem do reuso de K e V entre os 4 warps, não da ocupação em si.
// ============================================================================
static constexpr int SMEM_Q_HALFS = N_WARPS * TM * D_FIXED;   // 4096
static constexpr int SMEM_K_HALFS = TN * D_FIXED;              // 1024
static constexpr int SMEM_V_HALFS = TN * TK;                   // 256
static constexpr int SMEM_P_HALFS = N_WARPS * TM * TN;         // 1024
static constexpr int SMEM_TOTAL_HALFS =
    SMEM_Q_HALFS + SMEM_K_HALFS + SMEM_V_HALFS + SMEM_P_HALFS;
static constexpr int SMEM_BYTES = SMEM_TOTAL_HALFS * (int)sizeof(half);

// ============================================================================
// Helpers
// ============================================================================
static inline void check_cuda(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error em %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}
static inline float rand_val() {
    return (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
}
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline void print_sep(int w = 110) {
    for (int i = 0; i < w; i++) std::putchar('-');
    std::putchar('\n');
}

// ============================================================================
// Scatter accumulator → smem P tile privado do warp
// ============================================================================
__device__ __forceinline__
void scatter_S_to_half(
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S,
    half* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    const int ru   = gid,       rl   = gid + 8;
    const int c0   = tig * 2,   c1   = tig * 2 + 1;
    const int c4   = tig * 2 + 8, c5 = tig * 2 + 9;

    dst[ru*16+c0] = __float2half(S.x[0]);
    dst[ru*16+c1] = __float2half(S.x[1]);
    dst[ru*16+c4] = __float2half(S.x[4]);
    dst[ru*16+c5] = __float2half(S.x[5]);
    dst[rl*16+c0] = __float2half(S.x[2]);
    dst[rl*16+c1] = __float2half(S.x[3]);
    dst[rl*16+c4] = __float2half(S.x[6]);
    dst[rl*16+c5] = __float2half(S.x[7]);
}

// ============================================================================
// Kernel 4-warp com K/V compartilhado
//
// Grid : (N / (N_WARPS × TM), H, B)
//         Cada bloco processa N_WARPS=4 query tiles de 16 linhas = 64 linhas.
//
// Block: 128 threads = 4 warps
//   warp_id = threadIdx.x / 32   (0..3)
//   lane    = threadIdx.x & 31
//
// Algoritmo:
//   1. Todos os 4 warps carregam cooperativamente K (TN × D_FIXED) na smem.
//      Custo amortizado: 1 load de K por kv_tile para 4 query tiles.
//   2. Cada warp computa seu próprio S_acc = Q_warp @ K^T.
//      Q de cada warp é carregado individualmente (privado).
//   3. Softmax, online rescaling — cada warp independente nos seus regs.
//   4. P scatter → smem P privado do warp.
//   5. V carregado cooperativamente (TN × TK) → todos leem do mesmo slot.
//   6. O_acc acumulado em registradores — cada warp tem os seus.
//   7. Ao final: finalize e store para global.
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SZ, 1)
flash_mha_4warp_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int N,
    int H,
    float scale)
{
    // ── Identificação ─────────────────────────────────────────────────────────
    const int warp_id = threadIdx.x / WARP_SZ;   // 0..3
    const int lane    = threadIdx.x & 31;

    // Cada bloco processa N_WARPS query tiles contíguos.
    // blockIdx.x indexa grupos de N_WARPS tiles.
    const int block_q_base = blockIdx.x * N_WARPS * TM;
    const int warp_q_base  = block_q_base + warp_id * TM;

    const int head_id = blockIdx.y;
    const int batch   = blockIdx.z;

    if (warp_q_base >= N) return;

    const int head_stride = N * D_FIXED;
    const int base_off    = (batch * H + head_id) * head_stride;

    const half* __restrict__ Qh = Q + base_off;
    const half* __restrict__ Kh = K + base_off;
    const half* __restrict__ Vh = V + base_off;
    half*       __restrict__ Oh = O + base_off;

    const int num_kv = N / TN;

    // ── Shared memory ─────────────────────────────────────────────────────────
    extern __shared__ half smem[];

    // Q privado por warp: warp_id × (TM × D_FIXED)
    half* smem_Q_base = smem;
    half* smem_Q      = smem_Q_base + warp_id * TM * D_FIXED;

    // K compartilhado (TN × D_FIXED = 1024 halfs)
    half* smem_K = smem_Q_base + SMEM_Q_HALFS;

    // V compartilhado (TN × TK = 256 halfs) — um slot, recarregado por d_out
    half* smem_V = smem_K + SMEM_K_HALFS;

    // P privado por warp: warp_id × (TM × TN)
    half* smem_P_base = smem_V + SMEM_V_HALFS;
    half* smem_P      = smem_P_base + warp_id * TM * TN;

    // ── Acumuladores persistentes (em registradores, privados por warp) ───────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[D_TILES];
    #pragma unroll
    for (int t = 0; t < D_TILES; t++)
        wmma::fill_fragment(O_acc[t], 0.0f);

    OnlineState state;
    online_state_init(state);

    // =========================================================================
    // PRÉ-CARREGAMENTO DE Q (permanece em smem durante todo o loop KV)
    // Cada warp carrega seu próprio Q tile uma vez.
    // =========================================================================
    #pragma unroll
    for (int d_in = 0; d_in < D_TILES; d_in++) {
        const int db = d_in * TK;
        // 32 threads cobrem TM×TK = 256 elementos; cada thread carrega 8
        #pragma unroll
        for (int i = lane; i < TM * TK; i += WARP_SZ) {
            const int r = i / TK, c = i % TK;
            const int gr = warp_q_base + r;
            smem_Q[r * D_FIXED + db + c] =
                (gr < N) ? Qh[gr * D_FIXED + db + c] : __float2half(0.f);
        }
    }
    // Barreira: garante que Q de todos os warps está pronto antes do loop KV.
    // (Necessário pois smem_Q de diferentes warps não se sobrepõem, mas
    //  usaremos __syncthreads para K e V, então sincronizamos aqui também.)
    __syncthreads();

    // =========================================================================
    // Loop KV tiles
    // =========================================================================
    for (int kv = 0; kv < num_kv; kv++) {
        const int kv_base = kv * TN;

        // ── Carrega K[kv_base:kv_base+16, 0:64] cooperativamente ─────────────
        // 128 threads carregam TN × D_FIXED = 1024 halfs → 8 elementos/thread.
        #pragma unroll
        for (int i = threadIdx.x; i < TN * D_FIXED; i += BLOCK_SZ) {
            const int r = i / D_FIXED, c = i % D_FIXED;
            const int gr = kv_base + r;
            smem_K[i] = (gr < N) ? Kh[gr * D_FIXED + c] : __float2half(0.f);
        }
        __syncthreads();  // K pronto para todos os warps

        // ── Cada warp computa S = Q_warp @ K^T ───────────────────────────────
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
        wmma::fill_fragment(S_acc, 0.0f);

        #pragma unroll
        for (int d_in = 0; d_in < D_TILES; d_in++) {
            const int db = d_in * TK;

            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;

            // Q já está em smem (pré-carregado); K também
            wmma::load_matrix_sync(Qf, smem_Q + db,      D_FIXED);
            wmma::load_matrix_sync(Kf, smem_K + db,      D_FIXED);
            wmma::mma_sync(S_acc, Qf, Kf, S_acc);
            __syncwarp();
        }

        // ── Softmax unnorm em registradores (cada warp independente) ──────────
        RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

        // ── Online step ───────────────────────────────────────────────────────
        OnlineStep step = online_step_update(state, ms);

        // ── Beta em S_acc ─────────────────────────────────────────────────────
        apply_beta_to_S(S_acc, step);

        // ── P scatter → smem privado do warp ─────────────────────────────────
        scatter_S_to_half(S_acc, smem_P);
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Pf;
        wmma::load_matrix_sync(Pf, smem_P, TN);

        // ── Loop d_out: V carregado cooperativamente, reusado por todos warps ─
        #pragma unroll
        for (int d_out = 0; d_out < D_TILES; d_out++) {
            const int vc = d_out * TK;

            // Barreira antes de sobrescrever smem_V
            __syncthreads();

            // 128 threads carregam TN × TK = 256 halfs → 2 elementos/thread
            #pragma unroll
            for (int i = threadIdx.x; i < TN * TK; i += BLOCK_SZ) {
                const int r = i / TK, c = i % TK;
                const int gr = kv_base + r;
                smem_V[i] = (gr < N) ? Vh[gr * D_FIXED + vc + c] : __float2half(0.f);
            }
            __syncthreads();  // V pronto para todos os warps

            // Alpha em O_acc deste d_out
            apply_alpha_to_O(O_acc[d_out], step);

            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> Vf;
            wmma::load_matrix_sync(Vf, smem_V, TK);
            wmma::mma_sync(O_acc[d_out], Pf, Vf, O_acc[d_out]);
            __syncwarp();
        }

        // Barreira após o loop d_out — protege smem_K para a próxima iteração
        __syncthreads();
    }

    // =========================================================================
    // Finaliza e escreve O para global
    // =========================================================================
    #pragma unroll
    for (int d_out = 0; d_out < D_TILES; d_out++) {
        finalize_O(O_acc[d_out], state);

        // Reusa smem_P (privado do warp) como scratch float[256]
        // smem_P tem TM×TN = 256 halfs = 512 B → como float = 1024 B
        // Verificar que 1024 B cabe: smem_P tem 256 halfs = 512 B → NÃO cabe
        // Solução: usar os primeiros 256 floats do smem_Q do próprio warp
        // smem_Q tem TM×D_FIXED = 1024 halfs = 2048 B → como float = 2048 B → ok
        float* scratch = reinterpret_cast<float*>(smem_Q);
        wmma::store_matrix_sync(scratch, O_acc[d_out], 16, wmma::mem_row_major);
        __syncwarp();

        const int oc = d_out * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += WARP_SZ) {
            const int r = i / TK, c = i % TK;
            const int gr = warp_q_base + r;
            if (gr < N)
                Oh[gr * D_FIXED + oc + c] = __float2half(scratch[i]);
        }
        __syncwarp();
    }
}

// ============================================================================
// Launcher
// ============================================================================
static void launch_4warp(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int N,
    float scale,
    cudaStream_t stream = 0)
{
    cudaFuncSetAttribute(flash_mha_4warp_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    // Cada bloco processa N_WARPS×TM = 64 linhas de Q
    const int q_blocks = ceil_div(N, N_WARPS * TM);
    dim3 grid(q_blocks, H, B);
    dim3 block(BLOCK_SZ);

    flash_mha_4warp_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        Q, K, V, O, N, H, scale);
}

// ============================================================================
// CPU reference
// ============================================================================
static void cpu_ref(
    const half* Q, const half* K, const half* V, float* O_ref,
    int B, int H, int N, int d, float scale)
{
    float* S = new float[(size_t)N * N];
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int off = (b*H+h)*N*d;
            const half* q=Q+off; const half* k=K+off;
            const half* v=V+off; float* o=O_ref+off;

            for (int i=0;i<N;i++)
                for (int j=0;j<N;j++) {
                    float acc=0.f;
                    for (int x=0;x<d;x++)
                        acc+=__half2float(q[i*d+x])*__half2float(k[j*d+x]);
                    S[i*N+j]=acc*scale;
                }
            for (int i=0;i<N;i++) {
                float mx=-FLT_MAX;
                for (int j=0;j<N;j++) mx=fmaxf(mx,S[i*N+j]);
                float sm=0.f;
                for (int j=0;j<N;j++){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}
                for (int j=0;j<N;j++) S[i*N+j]/=sm;
            }
            for (int i=0;i<N;i++)
                for (int j=0;j<d;j++) {
                    float acc=0.f;
                    for (int x=0;x<N;x++)
                        acc+=S[i*N+x]*__half2float(v[x*d+j]);
                    o[i*d+j]=acc;
                }
        }
    }
    delete[] S;
}

// ============================================================================
// Validação
// ============================================================================
static bool validate(
    const half* h_O, const float* h_ref,
    int B, int H, int N, int d, float tol=5e-2f)
{
    float max_abs=0.f, max_rel=0.f;
    int wb=0,wh=0,wr=0,wc=0;
    for (int b=0;b<B;b++)
        for (int h=0;h<H;h++) {
            int off=(b*H+h)*N*d;
            for (int i=0;i<N;i++)
                for (int j=0;j<d;j++) {
                    float gpu=__half2float(h_O[off+i*d+j]);
                    float ref=h_ref[off+i*d+j];
                    float err=fabsf(gpu-ref);
                    float rel=fabsf(ref)>1e-6f?err/fabsf(ref):err;
                    if (err>max_abs){max_abs=err;max_rel=rel;wb=b;wh=h;wr=i;wc=j;}
                }
        }
    printf("  Max abs error : %.6f  at [b=%d h=%d r=%d c=%d]\n",
           max_abs,wb,wh,wr,wc);
    printf("  Max rel error : %.4f%%\n", max_rel*100.f);
    return max_abs < tol;
}

// ============================================================================
// Timer e benchmark robusto
// ============================================================================
struct Timer {
    cudaEvent_t t0,t1;
    Timer(){cudaEventCreate(&t0);cudaEventCreate(&t1);}
    ~Timer(){cudaEventDestroy(t0);cudaEventDestroy(t1);}
    void start(){cudaEventRecord(t0);}
    float stop(int r=1){
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms=0.f;cudaEventElapsedTime(&ms,t0,t1);return ms/r;
    }
};

struct BenchResult { double min_ms, med_ms, max_ms; };

static BenchResult bench(
    const half* dQ, const half* dK, const half* dV, half* dO,
    int B, int H, int N, float scale,
    int outer=7, int inner=100)
{
    for (int i=0;i<3;i++) launch_4warp(dQ,dK,dV,dO,B,H,N,scale);
    cudaDeviceSynchronize();
    float samp[32]; if(outer>32)outer=32;
    Timer t;
    for (int r=0;r<outer;r++){
        t.start();
        for(int i=0;i<inner;i++) launch_4warp(dQ,dK,dV,dO,B,H,N,scale);
        samp[r]=t.stop(inner);
    }
    for(int i=0;i<outer-1;i++)
        for(int j=i+1;j<outer;j++)
            if(samp[j]<samp[i]){float tmp=samp[i];samp[i]=samp[j];samp[j]=tmp;}
    return {samp[0], samp[outer/2], samp[outer-1]};
}

// ============================================================================
// Ocupação
// ============================================================================
static void print_occ(const cudaDeviceProp& prop, int regs, int smem)
{
    const int mr = prop.regsPerMultiprocessor / (regs * BLOCK_SZ);
    const int ms = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem);
    const int mw = prop.maxThreadsPerMultiProcessor / BLOCK_SZ;
    int bn=mr; if(ms<bn)bn=ms; if(mw<bn)bn=mw;
    printf("  Regs/thread        : %d\n", regs);
    printf("  Smem/bloco         : %d B (%.1f KB)\n", smem, smem/1024.f);
    printf("  Max blocos/SM regs : %d\n", mr);
    printf("  Max blocos/SM smem : %d\n", ms);
    printf("  Max blocos/SM wrps : %d\n", mw);
    printf("  Blocos ativos/SM   : %d\n", bn);
    printf("  Warps ativos/SM    : %d  (%.0f%% de 32)\n",
           bn*N_WARPS, 100.f*bn*N_WARPS/32.f);
    printf("  Blocos p/ saturar  : %d\n", prop.multiProcessorCount * bn);
}

// ============================================================================
// run_case
// ============================================================================
static void run_case(
    int B, int H, int N_req,
    bool do_val, int outer, int inner,
    int actual_regs, bool header)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int   N     = ceil_div(N_req, N_WARPS*TM) * N_WARPS*TM;
    const float scale = 1.f / sqrtf(64.f);
    const int   total = B * H * N * D_FIXED;
    const size_t bqkv = (size_t)total * sizeof(half);
    const double flops = (double)B * H * 4.0 * N * N * D_FIXED;
    const int q_blocks = ceil_div(N, N_WARPS*TM);
    const int nblocks  = q_blocks * H * B;

    if (header) {
        printf("Device: %s  SM%d%d  %d SMs\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        printf("\n=== flash_mha_4warp  B=%d H=%d N_req=%d N_pad=%d d=64 ===\n",
               B, H, N_req, N);
        printf("Grid = (%d, %d, %d) = %d blocos  "
               "[cada bloco processa %d linhas de Q]\n\n",
               q_blocks, H, B, nblocks, N_WARPS*TM);
        print_occ(prop, actual_regs, SMEM_BYTES);
        printf("\n");
    }

    half *dQ,*dK,*dV,*dO;
    check_cuda(cudaMalloc(&dQ,bqkv),"Q");
    check_cuda(cudaMalloc(&dK,bqkv),"K");
    check_cuda(cudaMalloc(&dV,bqkv),"V");
    check_cuda(cudaMalloc(&dO,bqkv),"O");

    half *hQ=nullptr,*hK=nullptr,*hV=nullptr,*hO=nullptr;
    float *href=nullptr;

    if (do_val && (double)B*H*N*N*D_FIXED > 3e8) {
        printf("[warn] CPU ref pulada\n"); do_val=false;
    }

    if (do_val) {
        hQ=(half*)malloc(bqkv); hK=(half*)malloc(bqkv);
        hV=(half*)malloc(bqkv); hO=(half*)malloc(bqkv);
        href=(float*)malloc((size_t)total*sizeof(float));
        std::srand(42);
        for(int i=0;i<total;i++){
            hQ[i]=__float2half(rand_val());
            hK[i]=__float2half(rand_val());
            hV[i]=__float2half(rand_val());
        }
        printf("[1/3] CPU ref... "); fflush(stdout);
        cpu_ref(hQ,hK,hV,href,B,H,N,D_FIXED,scale);
        printf("OK\n");
        check_cuda(cudaMemcpy(dQ,hQ,bqkv,cudaMemcpyHostToDevice),"H2D Q");
        check_cuda(cudaMemcpy(dK,hK,bqkv,cudaMemcpyHostToDevice),"H2D K");
        check_cuda(cudaMemcpy(dV,hV,bqkv,cudaMemcpyHostToDevice),"H2D V");
        printf("[2/3] GPU... "); fflush(stdout);
        launch_4warp(dQ,dK,dV,dO,B,H,N,scale);
        check_cuda(cudaDeviceSynchronize(),"sync");
        check_cuda(cudaGetLastError(),"launch");
        printf("OK\n");
        check_cuda(cudaMemcpy(hO,dO,bqkv,cudaMemcpyDeviceToHost),"D2H O");
        printf("[3/3] Validacao...\n");
        bool ok=validate(hO,href,B,H,N,D_FIXED,5e-2f);
        printf("  Resultado: %s\n\n", ok?"PASS ✓":"FAIL ✗");
    } else {
        cudaMemset(dQ,0,bqkv);cudaMemset(dK,0,bqkv);cudaMemset(dV,0,bqkv);
    }

    BenchResult br=bench(dQ,dK,dV,dO,B,H,N,scale,outer,inner);

    const int mr=prop.regsPerMultiprocessor/(actual_regs*BLOCK_SZ);
    const int ms=(int)(prop.sharedMemPerMultiprocessor/(size_t)SMEM_BYTES);
    const int mw=prop.maxThreadsPerMultiProcessor/BLOCK_SZ;
    int bn=mr; if(ms<bn)bn=ms; if(mw<bn)bn=mw;
    const int need=prop.multiProcessorCount*bn;
    const float sat=fminf(100.f,100.f*nblocks/(float)need);

    auto tf=[&](double ms_val){ return flops/(ms_val*1e-3)/1e12; };

    if (header) {
        printf("=== Benchmark (outer=%d inner=%d) ===\n",outer,inner);
        printf("  %-10s %-10s %-10s %-10s\n","","min","median","max");
        printf("  %-10s %-10.4f %-10.4f %-10.4f\n","ms",br.min_ms,br.med_ms,br.max_ms);
        printf("  %-10s %-10.4f %-10.4f %-10.4f\n",
               "TFLOPS",tf(br.max_ms),tf(br.med_ms),tf(br.min_ms));
        printf("  Blocos: %d   sat: %.0f%%\n\n",nblocks,sat);
    } else {
        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-7d %-6.0f%%\n",
               B,H,N_req,
               br.min_ms,br.med_ms,br.max_ms,
               tf(br.max_ms),tf(br.med_ms),tf(br.min_ms),
               nblocks,sat);
    }

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    if(hQ)free(hQ);if(hK)free(hK);if(hV)free(hV);
    if(hO)free(hO);if(href)free(href);
}

// ============================================================================
// Sweep comparativo vs flatargs
// ============================================================================
static void run_sweep(int actual_regs)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int mr=prop.regsPerMultiprocessor/(actual_regs*BLOCK_SZ);
    const int ms=(int)(prop.sharedMemPerMultiprocessor/(size_t)SMEM_BYTES);
    const int mw=prop.maxThreadsPerMultiProcessor/BLOCK_SZ;
    int bn=mr; if(ms<bn)bn=ms; if(mw<bn)bn=mw;
    const int need=prop.multiProcessorCount*bn;

    printf("Device: %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("4-warp: regs=%d smem=%dB  Blocos/SM=%d  Para saturar=%d\n\n",
           actual_regs, SMEM_BYTES, bn, need);

    // Medianas do flatargs para comparação direta
    struct Ref { int B,H,N; double flat_med; };
    Ref refs[] = {
        {1,  1,  256, 0.0417},
        {1,  1,  512, 0.0801},
        {1,  1, 1024, 0.1322},
        {1,  4,  256, 0.0376},
        {1,  8,  256, 0.0503},
        {1, 12,  256, 0.0735},
        {1, 16,  256, 0.0736},
        {4,  8,  256, 0.1485},
        {8,  8,  256, 0.3196},
        {4, 12,  512, 0.6556},
        {4, 12, 1024, 2.6381},
        {8, 12,  512, 1.4109},
        {1, 16, 2048, 3.5005},
        {4,  8, 2048, 6.8282},
    };
    const int nref=(int)(sizeof(refs)/sizeof(refs[0]));

    printf("=== Sweep flash_mha_4warp vs flash_mha_flatargs (d=64) ===\n");
    printf("Benchmark: 7 rodadas × 100 iteracoes\n\n");
    printf("%-5s %-5s %-6s | "
           "%-27s | "
           "%-27s | "
           "%-8s %-8s %-6s\n",
           "B","H","N",
           "   4warp ms (min/med/max)   ",
           "  4warp TFLOPS (hi/med/lo)  ",
           "flat-med","speedup","sat%");
    print_sep();

    for (int i=0;i<nref;i++) {
        Ref& r=refs[i];
        const int N=ceil_div(r.N, N_WARPS*TM)*N_WARPS*TM;
        const size_t bqkv=(size_t)r.B*r.H*N*D_FIXED*sizeof(half);
        if (bqkv>(size_t)2*1024*1024*1024ULL) {
            printf("%-5d %-5d %-6d | SKIP\n",r.B,r.H,r.N); continue;
        }
        const float scale=1.f/sqrtf(64.f);
        const double flops=(double)r.B*r.H*4.0*N*N*D_FIXED;
        const int q_blocks=ceil_div(N,N_WARPS*TM);
        const int nblocks=q_blocks*r.H*r.B;
        const float sat=fminf(100.f,100.f*nblocks/(float)need);

        half *dQ,*dK,*dV,*dO;
        check_cuda(cudaMalloc(&dQ,bqkv),"Q");
        check_cuda(cudaMalloc(&dK,bqkv),"K");
        check_cuda(cudaMalloc(&dV,bqkv),"V");
        check_cuda(cudaMalloc(&dO,bqkv),"O");
        cudaMemset(dQ,0,bqkv);cudaMemset(dK,0,bqkv);cudaMemset(dV,0,bqkv);

        BenchResult br=bench(dQ,dK,dV,dO,r.B,r.H,N,scale,7,100);
        auto tf=[&](double ms_v){ return flops/(ms_v*1e-3)/1e12; };
        double speedup=r.flat_med/br.med_ms;

        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.3fx %-6.0f%%\n",
               r.B,r.H,r.N,
               br.min_ms,br.med_ms,br.max_ms,
               tf(br.max_ms),tf(br.med_ms),tf(br.min_ms),
               r.flat_med,speedup,sat);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    }
    print_sep();
    printf("\nspeedup > 1.0x = 4warp MAIS RAPIDO que flatargs (1 warp)\n");
    printf("speedup < 1.0x = reuse de K/V nao compensou a complexidade\n");
    printf("\nNotas de arquitetura:\n");
    printf("  4warp: 1 load de K por kv_tile para 4 query tiles\n");
    printf("  flat : 1 load de K por kv_tile para 1 query tile\n");
    printf("  Ganho teorico maximo de K: 4x menos loads de K\n");
    printf("  V ainda e carregado 4x (uma vez por d_out, compartilhado)\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    // Atualizar apos ver ptxas log
    const int ACTUAL_REGS = 86;

    printf("flash_mha_4warp — SM75  d=64\n");
    printf("Config: %d warps/bloco, %d threads/bloco\n", N_WARPS, BLOCK_SZ);
    printf("Smem/bloco: %d B (%.1f KB)\n", SMEM_BYTES, SMEM_BYTES/1024.f);
    printf("Regs esperados: %d\n\n", ACTUAL_REGS);

    if (argc == 1) {
        run_case(1, 2,  64, true, 5, 100, ACTUAL_REGS, true);
        run_case(1, 4, 128, true, 5, 100, ACTUAL_REGS, true);
        run_sweep(ACTUAL_REGS);
        return 0;
    }

    const int B     = std::atoi(argv[1]);
    const int H     = std::atoi(argv[2]);
    const int N     = std::atoi(argv[3]);
    const int outer = argc>4 ? std::atoi(argv[4]) : 7;
    const int inner = argc>5 ? std::atoi(argv[5]) : 100;
    const bool val  = ((double)B*H*N*N*D_FIXED) < 3e8;

    run_case(B, H, N, val, outer, inner, ACTUAL_REGS, true);
    return 0;
}

