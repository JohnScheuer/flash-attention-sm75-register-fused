import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import math, csv
from pathlib import Path
from contextlib import nullcontext
import torch, torch.nn.functional as F
from reference.attention_ref import attention_reference
from reference.correctness import check_correctness
import flash_attn_sm75_cuda as _C

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    times=[]
    for _ in range(iters):
        s.record(); fn(); e.record()
        torch.cuda.synchronize(); times.append(s.elapsed_time(e))
    times.sort(); return times[len(times)//2]

def sdpa_eff(Q,K,V,causal,scale):
    try:
        from torch.nn.attention import sdpa_kernel,SDPBackend
        ctx = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False,enable_math=False,enable_mem_efficient=True)
    with ctx:
        return F.scaled_dot_product_attention(Q,K,V,is_causal=causal,scale=scale)

def run(batch,heads,seq_len,causal):
    d=64; scale=1./math.sqrt(d)
    flops=4*batch*heads*seq_len*seq_len*d
    if causal: flops//=2
    torch.manual_seed(42)
    Q=torch.randn(batch,heads,seq_len,d,dtype=torch.float16,device='cuda')
    K=torch.randn(batch,heads,seq_len,d,dtype=torch.float16,device='cuda')
    V=torch.randn(batch,heads,seq_len,d,dtype=torch.float16,device='cuda')
    ref=attention_reference(Q,K,V,causal=causal,scale=scale)

    def measure(fn):
        ms=bench(fn); out=fn()
        err=check_correctness(out,ref)["max_abs_err"]
        return ms, flops/(ms*1e9), err

    se_ms,_,_   = measure(lambda: sdpa_eff(Q,K,V,causal,scale))
    sc_ms,sc_tf,sc_err = measure(lambda: _C.flash_attention_forward_scalar(Q,K,V,causal,scale))
    qk_ms,qk_tf,qk_err = measure(lambda: _C.flash_attention_forward_wmma_qk(Q,K,V,causal,scale))
    pv_ms,pv_tf,pv_err = measure(lambda: _C.flash_attention_forward(Q,K,V,causal,scale))

    return dict(
        batch=batch,heads=heads,seq_len=seq_len,causal=causal,
        sdpa_eff_ms=se_ms,
        scalar_ms=sc_ms, scalar_tflops=sc_tf,
        wmma_qk_ms=qk_ms, wmma_qk_tflops=qk_tf,
        wmma_pv_ms=pv_ms, wmma_pv_tflops=pv_tf,
        speedup_qk_vs_sc=sc_ms/qk_ms,
        speedup_pv_vs_sc=sc_ms/pv_ms,
        speedup_pv_vs_qk=qk_ms/pv_ms,
        wmma_pv_err=pv_err,
    )

configs=[
    (1, 8,  128,False),(1, 8,  256,False),(1, 8,  512,False),
    (1, 8, 1024,False),(1, 8, 2048,False),(1, 8, 4096,False),
    (1, 8,  128,True), (1, 8,  512,True), (1, 8, 1024,True),(1, 8, 2048,True),
    (1,32,  512,False),(1,32, 1024,False),
    (4, 8,  512,False),(4, 8, 1024,False),
]

def main():
    print(f"GPU: {torch.cuda.get_device_name()}\n")
    hdr=(f"{'B':>2}{'H':>4}{'N':>6}{'csl':>5}  "
         f"{'sdpa_eff':>9}{'scalar':>9}{'wmma_qk':>9}{'wmma_pv':>9}  "
         f"{'qk_TFLOPS':>11}{'pv_TFLOPS':>11}  "
         f"{'vs_sc(qk)':>10}{'vs_sc(pv)':>10}{'qk->pv':>8}")
    print(hdr); print("-"*len(hdr))
    rows=[]
    for c in configs:
        r=run(*c)
        print(f"{r['batch']:>2}{r['heads']:>4}{r['seq_len']:>6}{str(r['causal']):>5}  "
              f"{r['sdpa_eff_ms']:>9.3f}{r['scalar_ms']:>9.3f}"
              f"{r['wmma_qk_ms']:>9.3f}{r['wmma_pv_ms']:>9.3f}  "
              f"{r['wmma_qk_tflops']:>11.3f}{r['wmma_pv_tflops']:>11.3f}  "
              f"{r['speedup_qk_vs_sc']:>10.2f}x{r['speedup_pv_vs_sc']:>9.2f}x"
              f"{r['speedup_pv_vs_qk']:>7.2f}x")
        rows.append(r)
    csv_path=RESULTS_DIR/"benchmark_three_kernels.csv"
    with open(csv_path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")

if __name__=="__main__":
    main()
