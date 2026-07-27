cmake_minimum_required(VERSION 3.18)
project(flash_attention_sm75_fused LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# SM75 = Turing (RTX 2000, T4)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Flags de compilação
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -Xptxas -v \
    -Xptxas -dlcm=ca \
    --use_fast_math \
    -lineinfo")

include_directories(include)

# ── Binários ────────────────────────────────────────────────────────────────

# 1. Probe: descobre o layout real do WMMA accumulator
add_executable(probe_layout src/probe_layout.cu)
set_target_properties(probe_layout PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# 2. Teste isolado do warp softmax
add_executable(test_warp_softmax src/test_warp_softmax.cu)
set_target_properties(test_warp_softmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# 3. Kernel fusionado + validação
add_executable(flash_fused src/flash_fused_kernel.cu src/main.cu)
set_target_properties(flash_fused PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

