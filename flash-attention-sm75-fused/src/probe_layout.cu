// =============================================================================
// probe_layout.cu  — corrigido
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

// =============================================================================
// Kernel 1: mapeamento do accumulator (float)
// =============================================================================
__global__ void probe_accumulator()
{
    const int lane = threadIdx.x & 31;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

    // Codificar: thread T, elemento i → valor = T*100 + i
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc.x[i] = static_cast<float>(lane * 100 + i);
    }

    __shared__ float mat[16][16];

    for (int i = lane; i < 256; i += 32)
        reinterpret_cast<float*>(mat)[i] = -1.0f;
    __syncwarp();

    wmma::store_matrix_sync(&mat[0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();

    if (lane == 0) {
        printf("\n");
        printf("====================================================\n");
        printf(" WMMA Accumulator Layout  (m16n16k16, float, SM75) \n");
        printf("====================================================\n");
        printf(" Valor = lane*100 + fragment_index\n");
        printf(" Leia como: L<lane>.<frag_idx>\n\n");

        printf("     |");
        for (int c = 0; c < 16; c++) printf(" c%-2d |", c);
        printf("\n-----|");
        for (int c = 0; c < 16; c++) printf("------|");
        printf("\n");

        for (int r = 0; r < 16; r++) {
            printf("r%-3d |", r);
            for (int c = 0; c < 16; c++) {
                float v = mat[r][c];
                if (v < 0) {
                    printf(" ??? |");
                } else {
                    int l = (int)v / 100;
                    int i = (int)v % 100;
                    printf(" L%02d.%d|", l, i);
                }
            }
            printf("\n");
        }

        printf("\n=== Mapeamento Reverso: (thread, x[i]) -> (row, col) ===\n");
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                float v = mat[r][c];
                if (v >= 0) {
                    int l = (int)v / 100;
                    int i = (int)v % 100;
                    printf("  Thread %2d, x[%d] -> C[%2d][%2d]\n", l, i, r, c);
                }
            }
        }

        printf("\n=== Validacao do Modelo Teorico ===\n");
        printf("  group_id = lane>>2,  thread_in_group = lane&3\n\n");

        bool ok = true;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                float v = mat[r][c];
                if (v < 0) {
                    printf("  ERRO: C[%d][%d] nao foi escrito!\n", r, c);
                    ok = false;
                    continue;
                }

                int l  = (int)v / 100;
                int fi = (int)v % 100;

                int group_id        = l >> 2;
                int thread_in_group = l & 3;

                // Modelo esperado baseado no layout documentado
                bool upper   = (r < 8);
                int exp_grp  = upper ? r : (r - 8);
                int half_col = c / 8;          // 0=left cols 0-7, 1=right cols 8-15
                int col_mod2 = c % 2;          // par=0, impar=1
                int exp_tig  = (c % 8) / 2;   // 0..3

                // Indice esperado do fragmento
                int exp_fi;
                if (upper) {
                    exp_fi = (half_col == 0) ? col_mod2 : (4 + col_mod2);
                } else {
                    exp_fi = (half_col == 0) ? (2 + col_mod2) : (6 + col_mod2);
                }

                bool match = (group_id == exp_grp) &&
                             (thread_in_group == exp_tig) &&
                             (fi == exp_fi);

                if (!match) {
                    printf("  MISMATCH C[%d][%d]: "
                           "got L%02d (grp=%d tig=%d) x[%d], "
                           "expected grp=%d tig=%d x[%d]\n",
                           r, c,
                           l, group_id, thread_in_group, fi,
                           exp_grp, exp_tig, exp_fi);
                    ok = false;
                }
            }
        }

        if (ok)
            printf("  PASS: Modelo teorico CONFIRMADO para todos os 256 elementos.\n");
        else
            printf("  FAIL: Verificar layout acima e atualizar wmma_layout.cuh!\n");
    }
}

// =============================================================================
// Kernel 2: mapeamento do matrix_a (half, row_major)
//
// NOTA: wmma::matrix_a NAO tem store_matrix_sync.
// Estrategia: usar mma_sync com B = identidade para que C = A * I = A,
// e inspecionar o accumulator resultante que tem o mesmo layout de A.
// =============================================================================
__global__ void probe_matrix_a()
{
    const int lane = threadIdx.x & 31;

    // Construir identidade 16x16 em shared memory para usar como B
    __shared__ half smem_A[16 * 16];
    __shared__ half smem_I[16 * 16];
    __shared__ float smem_C[16 * 16];

    // Preencher A com valores codificados (lane*100 + i)
    // Precisamos conhecer o layout de A para isso — mas estamos descobrindo.
    // Entao: preenchemos smem_A com valores posicionais A[r][c] = r*16+c
    // e depois inspecionamos o resultado de A @ I para confirmar que A chegou
    // corretamente ao acumulador.

    for (int i = lane; i < 256; i += 32) {
        smem_A[i] = __float2half((float)i);  // A[r][c] = r*16 + c
    }

    // Identidade: I[r][c] = (r == c) ? 1.0 : 0.0
    for (int i = lane; i < 256; i += 32) {
        int r = i / 16, c = i % 16;
        smem_I[i] = __float2half((r == c) ? 1.0f : 0.0f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a,   16, 16, 16, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b,   16, 16, 16, half, wmma::row_major> I_frag;
    wmma::fragment<wmma::accumulator,16, 16, 16, float>                 C_acc;

    wmma::load_matrix_sync(A_frag, smem_A, 16);
    wmma::load_matrix_sync(I_frag, smem_I, 16);
    wmma::fill_fragment(C_acc, 0.0f);

    // C = A @ I = A  (o resultado em C_acc deve ser identico a A)
    wmma::mma_sync(C_acc, A_frag, I_frag, C_acc);
    __syncwarp();

    wmma::store_matrix_sync(smem_C, C_acc, 16, wmma::mem_row_major);
    __syncwarp();

    if (lane == 0) {
        printf("\n");
        printf("================================================\n");
        printf(" Verificacao matrix_a via C = A @ I = A\n");
        printf(" A[r][c] = r*16+c (valor posicional)\n");
        printf("================================================\n\n");

        bool ok = true;
        float max_err = 0.0f;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                float expected = (float)(r * 16 + c);
                float got      = smem_C[r * 16 + c];
                float err      = fabsf(got - expected);
                if (err > max_err) max_err = err;
                if (err > 0.5f) {
                    printf("  MISMATCH C[%d][%d]: got %.1f expected %.1f\n",
                           r, c, got, expected);
                    ok = false;
                }
            }
        }
        printf("  Erro maximo: %.4f\n", max_err);
        if (ok)
            printf("  PASS: A @ I = A confirmado. Load de matrix_a correto.\n");
        else
            printf("  FAIL: Problema no load de matrix_a!\n");

        // Mostrar o layout do accumulator resultante (= layout de A)
        printf("\n  Primeiras 4 linhas de C = A @ I:\n");
        for (int r = 0; r < 4; r++) {
            printf("  row%d: ", r);
            for (int c = 0; c < 16; c++)
                printf("%4.0f ", smem_C[r * 16 + c]);
            printf("\n");
        }
    }
}

// =============================================================================
// Kernel 3: verificar os shuffles de reducao
// Cada thread reporta seus valores de max/sum para validar o shuffle
// =============================================================================
__global__ void probe_shuffle_reduction()
{
    const int lane = threadIdx.x & 31;

    // Simular: cada thread tem um valor de "local_max_upper" = lane
    // Apos o shuffle XOR 1 e 2, todos os 4 threads do grupo devem ter
    // o maximo do grupo (= lane mais alto no grupo, que e group_id*4+3)
    float val = (float)lane;

    float tmp;
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 1); val = fmaxf(val, tmp);
    tmp = __shfl_xor_sync(0xFFFFFFFF, val, 2); val = fmaxf(val, tmp);

    // Reportar: todos os threads do grupo devem ter o mesmo valor
    // O valor esperado para o grupo g (lanes g*4..g*4+3) e g*4+3
    int expected = (lane >> 2) * 4 + 3;
    bool ok_local = ((int)val == expected);

    // Thread 0 coleta via shuffle e reporta
    if (lane == 0) {
        printf("\n=== Teste de Shuffle XOR {1,2} para reducao de grupo de 4 ===\n");
        printf("  (Cada grupo de 4 threads deve ter o mesmo maximo)\n\n");
        printf("  lane | valor_apos_reducao | esperado | ok?\n");
    }
    __syncwarp();

    // Cada thread imprime sua linha (em ordem aproximada)
    for (int t = 0; t < 32; t++) {
        if (lane == t) {
            printf("   %2d  |        %2d          |    %2d    | %s\n",
                   lane, (int)val, expected, ok_local ? "OK" : "FAIL");
        }
        __syncwarp();
    }

    // Checar globalmente
    unsigned int all_ok = __ballot_sync(0xFFFFFFFF, ok_local);
    if (lane == 0) {
        if (all_ok == 0xFFFFFFFF)
            printf("\n  PASS: Todos os 32 lanes com reducao correta.\n");
        else
            printf("\n  FAIL: Lanes com erro: 0x%08X\n", ~all_ok);
    }
}

// =============================================================================
int main()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  (SM%d%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major * 10 + prop.minor < 75) {
        printf("AVISO: Este kernel requer SM75+. Detectado SM%d%d.\n",
               prop.major, prop.minor);
    }

    probe_accumulator<<<1, 32>>>();
    cudaDeviceSynchronize();

    probe_matrix_a<<<1, 32>>>();
    cudaDeviceSynchronize();

    probe_shuffle_reduction<<<1, 32>>>();
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "\nCUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n=== Probe concluido ===\n");
    return 0;
}

