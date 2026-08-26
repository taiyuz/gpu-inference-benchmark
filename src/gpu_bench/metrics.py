"""Latency percentiles, throughput, and GPU memory accounting.

Throughput is batch_size / (mean_ms / 1000). GPU memory is taken from
torch.cuda.max_memory_allocated() when CUDA is present; otherwise None.
Values are never fabricated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class RunResult:
    backend: str
    precision: str
    batch_size: int
    graph: bool
    n_warmup: int
    n_iter: int
    latencies_ms: list[float]
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_ips: float
    gpu_mem_bytes: int | None
    timing_backend: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(latencies_ms: Sequence[float], q: float) -> float:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("latencies_ms is empty")
    return float(np.percentile(arr, q))


def mean_ms(latencies_ms: Sequence[float]) -> float:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("latencies_ms is empty")
    return float(arr.mean())


def throughput_ips(batch_size: int, mean_latency_ms: float) -> float:
    if mean_latency_ms <= 0:
        raise ValueError("mean_latency_ms must be > 0")
    return float(batch_size) / (mean_latency_ms / 1000.0)


def gpu_mem_bytes() -> int | None:
    """Peak allocated bytes on the current CUDA device, or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def summarize(
    *,
    backend: str,
    precision: str,
    batch_size: int,
    graph: bool,
    n_warmup: int,
    n_iter: int,
    latencies_ms: Sequence[float],
    timing_backend: str,
    notes: str,
    gpu_mem: int | None = None,
) -> RunResult:
    samples = [float(x) for x in latencies_ms]
    mean = mean_ms(samples)
    return RunResult(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        graph=graph,
        n_warmup=n_warmup,
        n_iter=n_iter,
        latencies_ms=samples,
        mean_ms=mean,
        p50_ms=percentile(samples, 50.0),
        p90_ms=percentile(samples, 90.0),
        p99_ms=percentile(samples, 99.0),
        throughput_ips=throughput_ips(batch_size, mean),
        gpu_mem_bytes=gpu_mem if gpu_mem is not None else gpu_mem_bytes(),
        timing_backend=timing_backend,
        notes=notes,
    )
