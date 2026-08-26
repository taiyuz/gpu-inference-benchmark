from __future__ import annotations

import pytest

from gpu_bench.backends import BACKENDS
from gpu_bench.config import BenchConfig
from gpu_bench.report import DASH, markdown_table
from gpu_bench.runner import run_one, run_suite


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


def test_graph_skipped_on_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tiny_cpu_cfg.graph = True
    result = run_one("pytorch", tiny_cpu_cfg)
    assert result.skipped


def test_onnx_cpu_tiny(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    result = run_one("onnx", tiny_cpu_cfg)
    if result.skipped:
        pytest.skip(result.notes)
    assert result.n_iter == 3
    assert result.mean_ms > 0


def test_markdown_skipped_uses_dash(tiny_cpu_cfg: BenchConfig) -> None:
    tiny_cpu_cfg.precision = "fp16"
    tiny_cpu_cfg.graph = True
    results = [run_one("tensorrt", tiny_cpu_cfg)]
    md = markdown_table(results)
    assert DASH in md
    assert "ms" in md
    assert results[0].skipped


def test_suite_default_runs_requested_backend(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CPU dummy path")
    results = run_suite(backends=("pytorch",), precisions=("fp32",), batches=(1,), base=tiny_cpu_cfg)
    assert len(results) == 1
    assert results[0].backend == "pytorch"


def test_bf16_skipped_on_cpu(tiny_cpu_cfg: BenchConfig) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    tiny_cpu_cfg.precision = "bf16"
    result = run_one("pytorch", tiny_cpu_cfg)
    assert result.skipped
