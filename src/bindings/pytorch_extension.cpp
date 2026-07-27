#include <torch/extension.h>
#include <cuda_fp16.h>

extern "C" void launch_naive_attention(
    const half*, const half*, const half*, half*,
    int, int, int, int, float, bool, float*
);
extern "C" void launch_flash_attention_forward(
    const half*, const half*, const half*, half*,
    int, int, int, int, float, bool
);
extern "C" void launch_flash_attention_forward_scalar(
    const half*, const half*, const half*, half*,
    int, int, int, int, float, bool
);
extern "C" void launch_flash_attention_forward_wmma_qk(
    const half*, const half*, const half*, half*,
    int, int, int, int, float, bool
);

static void check_tensor(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(),        std::string(name) + " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat16, std::string(name) + " must be FP16");
    TORCH_CHECK(t.is_contiguous(),  std::string(name) + " must be contiguous");
    TORCH_CHECK(t.dim() == 4,       std::string(name) + " must be [B, H, N, D]");
}

torch::Tensor naive_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, float scale
) {
    check_tensor(Q, "Q"); check_tensor(K, "K"); check_tensor(V, "V");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "naive_attention: Q, K, V must have identical shapes");

    auto O  = torch::empty_like(Q);
    auto ws = torch::empty(
        {Q.size(0)*Q.size(1), Q.size(2), Q.size(2)},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );
    launch_naive_attention(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        Q.size(0), Q.size(1), Q.size(2), Q.size(3),
        scale, causal, ws.data_ptr<float>()
    );
    return O;
}

torch::Tensor flash_attention_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, float scale
) {
    check_tensor(Q, "Q"); check_tensor(K, "K"); check_tensor(V, "V");

    // After GQA expansion in Python, shapes must match
    TORCH_CHECK(Q.size(0) == K.size(0), "batch mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1), "heads mismatch after GQA expansion");
    TORCH_CHECK(Q.size(2) == K.size(2), "seq_len mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3), "head_dim mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3) && K.sizes() == V.sizes(),
                "K and V must have identical shapes");

    const auto head_dim = Q.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "flash_attention_forward supports head_dim=64 or 128, got ", head_dim);

    auto O = torch::empty_like(Q);
    launch_flash_attention_forward(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        Q.size(0), Q.size(1), Q.size(2), Q.size(3),
        scale, causal
    );
    return O;
}

torch::Tensor flash_attention_forward_scalar(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, float scale
) {
    check_tensor(Q, "Q"); check_tensor(K, "K"); check_tensor(V, "V");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have identical shapes");
    TORCH_CHECK(Q.size(3) == 64, "scalar kernel supports head_dim=64 only");

    auto O = torch::empty_like(Q);
    launch_flash_attention_forward_scalar(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        Q.size(0), Q.size(1), Q.size(2), Q.size(3),
        scale, causal
    );
    return O;
}

torch::Tensor flash_attention_forward_wmma_qk(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, float scale
) {
    check_tensor(Q, "Q"); check_tensor(K, "K"); check_tensor(V, "V");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have identical shapes");
    TORCH_CHECK(Q.size(3) == 64, "wmma_qk kernel supports head_dim=64 only");

    auto O = torch::empty_like(Q);
    launch_flash_attention_forward_wmma_qk(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        Q.size(0), Q.size(1), Q.size(2), Q.size(3),
        scale, causal
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention_forward",         &naive_attention_forward);
    m.def("flash_attention_forward",         &flash_attention_forward);
    m.def("flash_attention_forward_scalar",  &flash_attention_forward_scalar);
    m.def("flash_attention_forward_wmma_qk", &flash_attention_forward_wmma_qk);
}
