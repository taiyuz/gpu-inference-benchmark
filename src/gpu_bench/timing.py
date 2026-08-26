"""GPU timing. CUDA events are the source of truth; wall clock is a labeled fallback."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TimingBackend:
    name: str
    notes: str


class Timer(Protocol):
    @property
    def backend(self) -> TimingBackend: ...

    def measure(self, fn: Callable[[], Any]) -> float: ...


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class CudaEventTimer:
    """``torch.cuda.Event.elapsed_time`` in milliseconds (GPU-side)."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot use CUDA event timing")
        self._torch = torch
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    @property
    def backend(self) -> TimingBackend:
        return TimingBackend(
            name="cuda_event",
            notes="torch.cuda.Event.elapsed_time (GPU-side, ms)",
        )

    def measure(self, fn: Callable[[], Any]) -> float:
        torch = self._torch
        torch.cuda.synchronize()
        self._start.record()
        fn()
        self._end.record()
        self._end.synchronize()
        return float(self._start.elapsed_time(self._end))


class WallClockTimer:
    """Host ``perf_counter`` fallback. Not a substitute for CUDA events."""

    @property
    def backend(self) -> TimingBackend:
        return TimingBackend(
            name="wall_clock",
            notes=(
                "time.perf_counter fallback — NOT a CUDA event. "
                "Do not treat as GPU kernel time."
            ),
        )

    def measure(self, fn: Callable[[], Any]) -> float:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0


def make_timer(*, require_cuda_events: bool = False) -> Timer:
    if cuda_is_available():
        return CudaEventTimer()
    if require_cuda_events:
        raise RuntimeError(
            "CUDA events required but no NVIDIA GPU/CUDA runtime is visible. "
            "Run on a CUDA machine or drop --require-cuda-events."
        )
    return WallClockTimer()
