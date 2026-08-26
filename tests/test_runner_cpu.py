"""CPU-only runner tests. No NVIDIA GPU required."""

from __future__ import annotations

import pytest

from gpu_bench.backends.tensorrt import TensorRTBackend
from gpu_bench.config import RunConfig
from gpu_bench.runner import expand_jobs, run_suite

torch = pytest.importorskip("torch")


def test_pytorch_tiny_cpu_fp32(tiny_cpu_cfg: RunConfig) -> None:
    suite = run_suite([tiny_cpu_cfg])
    assert suite.skips == []
    assert len(suite.results) == 1
    result = suite.results[0]
    assert result.backend == "pytorch"
    assert result.precision == "fp32"
    assert result.n_iter == 3
    assert result.mean_ms > 0.0
    assert len(result.latencies_ms) == 3
    assert result.timing_backend == "wall_clock"
    assert result.gpu_mem_bytes is None
    assert result.graph is False


def test_pytorch_fp16_skipped_on_cpu(artifacts_dir) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA present; FP16 skip is a CPU-only assertion")
    cfg = RunConfig(
        backend="pytorch",
        precision="fp16",
        batch_size=1,
        n_warmup=0,
        n_iter=1,
        model="tiny",
        artifacts_dir=artifacts_dir,
    )
    suite = run_suite([cfg])
    assert suite.results == []
    assert suite.skips
    assert "FP16" in suite.skips[0].reason


def test_onnx_tiny_cpu(artifacts_dir) -> None:
    pytest.importorskip("onnxruntime")
    cfg = RunConfig(
        backend="onnx",
        precision="fp32",
        batch_size=1,
        n_warmup=1,
        n_iter=2,
        model="tiny",
        artifacts_dir=artifacts_dir,
    )
    suite = run_suite([cfg])
    if suite.skips:
        pytest.skip(suite.skips[0].reason)
    assert len(suite.results) == 1
    assert suite.results[0].backend == "onnx"
    assert suite.results[0].mean_ms > 0.0


def test_tensorrt_unavailable_on_cpu() -> None:
    backend = TensorRTBackend()
    if torch.cuda.is_available():
        pytest.skip("CUDA present; TRT available() is not expected to be False")
    assert backend.available() is False
    assert backend.unavailable_reason()


def test_graph_path_skipped_without_cuda(tiny_cpu_cfg: RunConfig) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA Graphs require a GPU; this test is the CPU skip path")
    tiny_cpu_cfg.graph = True
    suite = run_suite([tiny_cpu_cfg])
    assert suite.skips == []
    assert suite.results[0].graph is False
    assert "CUDA" in suite.results[0].notes


def test_full_suite_expands_graphs() -> None:
    jobs = expand_jobs(
        backends=None,
        precisions=None,
        batches=None,
        graph=False,
        suite="full",
        warmup=1,
        iters=1,
        include_transfer=False,
        pinned=False,
        model="tiny",
        require_cuda_events=False,
        artifacts_dir=__import__("pathlib").Path("artifacts"),
    )
    assert any(j.graph and j.batch_size == 1 for j in jobs)
    assert any(j.graph and j.batch_size == 8 for j in jobs)
    assert {j.batch_size for j in jobs} >= {1, 8, 16, 32}
