"""Latency percentiles, throughput, and GPU memory accounting."""

from __future__ import annotations

import math
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
    stdev_ms: float
    throughput_ips: float
    gpu_mem_bytes: int | None
    timing_backend: str
    notes: str = ""
    skipped: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_latencies(latencies_ms: list[float]) -> list[float]:
    """Reject empty, non-finite, or negative samples. Those are not measurements."""
    if not latencies_ms:
        raise ValueError("latencies_ms is empty")
    out: list[float] = []
    for i, raw in enumerate(latencies_ms):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"latencies_ms[{i}] is not finite: {raw!r}")
        if value < 0.0:
            raise ValueError(f"latencies_ms[{i}] is negative: {value}")
        out.append(value)
    return out


def tail_warning(n_iter: int, q: float = 99.0) -> str | None:
    """p99 of a handful of samples is just the max — do not imply a stable tail."""
    if n_iter < 1:
        return "no samples"
    # Need at least 1/(1-q/100) draws to have one point in the upper tail.
    min_n = max(2, math.ceil(100.0 / max(100.0 - q, 1e-9)))
    if n_iter < min_n:
        return (
            f"p{q:g} is under-sampled with n_iter={n_iter} "
            f"(want n_iter>={min_n}); do not treat as a stable tail"
        )
    return None


def sample_stdev_ms(latencies_ms: list[float]) -> float:
    """Sample standard deviation (ddof=1). Undefined for n<2 → NaN, not 0."""
    arr = np.asarray(validate_latencies(latencies_ms), dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1))


def percentiles(latencies_ms: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(validate_latencies(latencies_ms), dtype=np.float64)
    mean = float(arr.mean())
    p50, p90, p99 = (float(x) for x in np.percentile(arr, [50, 90, 99]))
    return mean, p50, p90, p99


def throughput_ips(batch_size: int, mean_ms: float) -> float:
    if not math.isfinite(mean_ms) or mean_ms <= 0:
        raise ValueError("mean_ms must be > 0 and finite")
    return batch_size / (mean_ms / 1000.0)


def mean_ms(latencies_ms: list[float]) -> float:
    arr = np.asarray(validate_latencies(latencies_ms), dtype=np.float64)
    return float(arr.mean())


def percentile(latencies_ms: list[float], q: float) -> float:
    arr = np.asarray(validate_latencies(latencies_ms), dtype=np.float64)
    return float(np.percentile(arr, q))


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
    n_iter: int | None = None,
    gpu_mem: int | None = None,
) -> RunResult:
    del n_iter  # derived from latencies_ms; accepted for test/call-site compatibility
    if gpu_mem_bytes is None and gpu_mem is not None:
        gpu_mem_bytes = gpu_mem
    samples = validate_latencies(latencies_ms)
    mean_val, p50_ms, p90_ms, p99_ms = percentiles(samples)
    extra_out = dict(extra or {})
    warning = tail_warning(len(samples), 99.0)
    if warning:
        extra_out["tail_warning"] = warning
    return RunResult(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        graph=graph,
        n_warmup=n_warmup,
        n_iter=len(samples),
        latencies_ms=list(samples),
        mean_ms=mean_val,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        stdev_ms=sample_stdev_ms(samples),
        throughput_ips=throughput_ips(batch_size, mean_val),
        gpu_mem_bytes=gpu_mem_bytes if gpu_mem_bytes is not None else gpu_peak_bytes(),
        timing_backend=timing_backend,
        notes=notes,
        extra=extra_out,
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
        stdev_ms=float("nan"),
        throughput_ips=float("nan"),
        gpu_mem_bytes=None,
        timing_backend="none",
        notes=reason,
        skipped=True,
    )
