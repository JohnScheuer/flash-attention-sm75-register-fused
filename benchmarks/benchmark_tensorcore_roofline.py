import torch
import time

def bench_mm(M, N, K, iters=100):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        c = a @ b
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    ms = times[len(times)//2]
    flops = 2 * M * N * K
    tflops = flops / (ms * 1e9)
    return ms, tflops

def main():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("PyTorch GEMM roofline probe\n")
    shapes = [
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (4096, 8192, 4096),
        (8192, 8192, 4096),
        (8192, 8192, 8192),
    ]
    print(f"{'M':>6}{'N':>6}{'K':>6}  {'ms':>10}{'TFLOPS':>10}")
    print("-" * 40)
    for M, N, K in shapes:
        ms, tf = bench_mm(M, N, K)
        print(f"{M:>6}{N:>6}{K:>6}  {ms:>10.3f}{tf:>10.3f}")

if __name__ == "__main__":
    main()
