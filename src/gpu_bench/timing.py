"""CUDA event and wall-clock timers.

CUDA events measure device-side elapsed time. Wall-clock is a labeled fallback
only and is NOT a CUDA event.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Timer(Protocol):
    name: str
    notes: str

    def start(self) -> None: ...

    def stop(self) -> float:
        """Elapsed milliseconds since start()."""
        ...


class CudaEventTimer:
    """GPU elapsed time via torch.cuda.Event(enable_timing=True)."""

    name = "cuda_events"
    notes = "CUDA event timing (device-side Event.elapsed_time). Not wall-clock."

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CudaEventTimer requires a CUDA device")
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop(self) -> float:
        self._end.record()
        self._end.synchronize()
        return float(self._start.elapsed_time(self._end))


class WallClockTimer:
    """Host perf_counter timer. This is NOT a CUDA event."""

    name = "wall_clock"
    notes = "Wall-clock timing via time.perf_counter. This is NOT a CUDA event."

    def __init__(self) -> None:
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def make_timer(*, require_cuda_events: bool = False) -> Timer:
    """Return a CUDA event timer when CUDA is available, else wall-clock.

    If require_cuda_events is True and CUDA is missing, raise RuntimeError.
    """
    cuda_ok = _cuda_available()
    if require_cuda_events and not cuda_ok:
        raise RuntimeError("CUDA event timing required but CUDA is not available")
    if cuda_ok:
        return CudaEventTimer()
    return WallClockTimer()
