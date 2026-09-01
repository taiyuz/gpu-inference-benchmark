from __future__ import annotations

import pytest

from gpu_bench.backends import BACKENDS
from gpu_bench.config import BenchConfig
from gpu_bench.report import DASH, markdown_table
from gpu_bench.runner import expand_jobs, run_one, run_suite


def test_tensorrt_unavailable_without_gpu() -> None:
    ok, reason = BACKENDS["tensorrt"].available()
    if ok:
        pytest.skip("TensorRT+CUDA unexpectedly available")
    assert ok is False
    assert reason


def test_pytorch_tiny_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if torch.cuda.is_available():
        pytest.skip("this test is the CPU dummy path")
    result = run_one("pytorch", tiny_cpu_cfg)
    assert not result.skipped
    assert result.n_iter == 3
    assert result.mean_ms > 0
    assert result.timing_backend == "wall_clock"
    assert result.gpu_mem_bytes is None


def test_pytorch_fp16_skipped_on_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tiny_cpu_cfg.precision = "fp16"
    result = run_one("pytorch", tiny_cpu_cfg)
    assert result.skipped


def test_bf16_skipped_on_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tiny_cpu_cfg.precision = "bf16"
    result = run_one("pytorch", tiny_cpu_cfg)
    assert result.skipped


def test_graph_skipped_on_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tiny_cpu_cfg.graph = True
    result = run_one("pytorch", tiny_cpu_cfg)
    assert result.skipped


def test_markdown_skipped_uses_dash(tiny_cpu_cfg: BenchConfig) -> None:
    tiny_cpu_cfg.precision = "fp16"
    tiny_cpu_cfg.graph = True
    results = [run_one("tensorrt", tiny_cpu_cfg)]
    md = markdown_table(results)
    assert DASH in md
    assert results[0].skipped


def test_suite_default_runs_requested_backend(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CPU dummy path")
    results = run_suite(backends=("pytorch",), precisions=("fp32",), batches=(1,), base=tiny_cpu_cfg)
    assert len(results) == 1
    assert results[0].backend == "pytorch"


def test_full_suite_includes_bf16_and_graphs() -> None:
    jobs = expand_jobs(suite="full")
    assert any(p == "bf16" for _, p, _, _ in jobs)
    assert any(g and b == 1 for *_, b, g in jobs)
    assert any(g and b == 8 for *_, b, g in jobs)
    assert {b for _, _, b, _ in jobs} >= {1, 8, 16, 32}


def test_full_suite_cpu_never_invents_gpu_numbers(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import math

    import torch

    if torch.cuda.is_available():
        pytest.skip("CPU dummy path")
    tiny_cpu_cfg.warmup = 0
    tiny_cpu_cfg.iters = 2
    results = run_suite(suite="full", base=tiny_cpu_cfg)
    assert results
    for result in results:
        if result.skipped:
            assert math.isnan(result.mean_ms)
            assert math.isnan(result.stdev_ms)
            assert result.latencies_ms == []
            assert result.timing_backend == "none"
            continue
        # Only PyTorch FP32 eager can measure on CPU, and only as wall-clock.
        assert result.backend == "pytorch"
        assert result.precision == "fp32"
        assert result.graph is False
        assert result.timing_backend == "wall_clock"
        assert result.mean_ms > 0
        assert result.gpu_mem_bytes is None
