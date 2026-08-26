"""Latency percentiles, throughput, and GPU memory accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    notes: str = ""
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentiles(latencies_ms: list[float]) -> tuple[float, float, float, float]:
    if not latencies_ms:
        raise ValueError("latencies_ms is empty")
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    p50, p90, p99 = (float(x) for x in np.percentile(arr, [50, 90, 99]))
    return mean, p50, p90, p99


def throughput_ips(batch_size: int, mean_ms: float) -> float:
    if mean_ms <= 0:
        raise ValueError("mean_ms must be > 0")
    return batch_size / (mean_ms / 1000.0)


def gpu_peak_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        return


def summarize(
    *,
    backend: str,
    precision: str,
    batch_size: int,
    graph: bool,
    n_warmup: int,
    latencies_ms: list[float],
    timing_backend: str,
    notes: str = "",
    gpu_mem_bytes: int | None = None,
    extra: dict[str, Any] | None = None,
) -> RunResult:
    mean_ms, p50_ms, p90_ms, p99_ms = percentiles(latencies_ms)
    return RunResult(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        graph=graph,
        n_warmup=n_warmup,
        n_iter=len(latencies_ms),
        latencies_ms=list(latencies_ms),
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        throughput_ips=throughput_ips(batch_size, mean_ms),
        gpu_mem_bytes=gpu_mem_bytes if gpu_mem_bytes is not None else gpu_peak_bytes(),
        timing_backend=timing_backend,
        notes=notes,
        extra=extra or {},
    )


def skipped_result(
    *,
    backend: str,
    precision: str,
    batch_size: int,
    graph: bool,
    reason: str,
) -> RunResult:
    return RunResult(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        graph=graph,
        n_warmup=0,
        n_iter=0,
        latencies_ms=[],
        mean_ms=float("nan"),
        p50_ms=float("nan"),
        p90_ms=float("nan"),
        p99_ms=float("nan"),
        throughput_ips=float("nan"),
        gpu_mem_bytes=None,
        timing_backend="none",
        notes=reason,
        skipped=True,
    )
