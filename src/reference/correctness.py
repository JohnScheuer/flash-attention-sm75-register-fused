"""
Correctness checking utilities.
"""

import torch


def check_correctness(output, reference, name="", atol=1e-2, rtol=1e-2):
    """
    Compare output against reference.
    Returns dict with error metrics.
    """
    output_f = output.float()
    reference_f = reference.float()

    abs_err = (output_f - reference_f).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    rel_err = abs_err / (reference_f.abs() + 1e-8)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()

    is_close = torch.allclose(output_f, reference_f, atol=atol, rtol=rtol)

    result = {
        "name": name,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err,
        "mean_rel_err": mean_rel_err,
        "allclose": is_close,
        "atol": atol,
        "rtol": rtol,
    }

    return result


def print_correctness(result):
    status = "PASS" if result["allclose"] else "FAIL"
    print(f"  [{status}] {result['name']}")
    print(f"    max_abs_err: {result['max_abs_err']:.6f}")
    print(f"    mean_abs_err: {result['mean_abs_err']:.6f}")
    print(f"    max_rel_err: {result['max_rel_err']:.6f}")
    print(f"    mean_rel_err: {result['mean_rel_err']:.6f}")
