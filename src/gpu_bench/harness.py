"""Shared timed loop: warmup is discarded; only timed iters become latencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gpu_bench.timing import Timer


@dataclass(frozen=True)
class TimedLoopResult:
    """Outcome of ``warmup_then_measure``. Warmup samples are never included."""

    latencies_ms: list[float]
    n_warmup: int
    n_iter: int
    timing_backend: str
    sync_before_timed: bool
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def warmup_then_measure(
    step: Callable[[], Any],
    timer: Timer,
    *,
    warmup: int,
    iters: int,
    sync_fn: Callable[[], None] | None = None,
) -> TimedLoopResult:
    """Run ``warmup`` discarded steps, optional sync, then ``iters`` timed measures.

    Warmup iterations are never recorded into ``latencies_ms``. Only the timed
    region feeds percentiles / throughput. ``sync_fn`` (typically
    ``torch.cuda.synchronize``) runs once after warmup and before the first
    timed sample so leftover warmup GPU work cannot inflate the first measure.

    Wall-clock timers are labeled honestly: they are not CUDA-event kernel time
    and must not be pasted into the README GPU results table.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iters < 1:
        raise ValueError("iters must be >= 1")

    for _ in range(warmup):
        step()

    synced = False
    if sync_fn is not None:
        sync_fn()
        synced = True

    latencies = [float(timer.measure(step)) for _ in range(iters)]

    notes: list[str] = [
        f"warmup={warmup} discarded; timed_iters={iters} recorded",
    ]
    if synced:
        notes.append("sync after warmup before timed region")
    if timer.name == "wall_clock":
        notes.append(
            "timing_backend=wall_clock \u2014 NOT CUDA-event kernel time; "
            "do not paste into the GPU results table"
        )
    else:
        notes.append(f"timing_backend={timer.name}")

    return TimedLoopResult(
        latencies_ms=latencies,
        n_warmup=warmup,
        n_iter=len(latencies),
        timing_backend=timer.name,
        sync_before_timed=synced,
        notes=tuple(notes),
        extra={
            "warmup_discarded": True,
            "n_warmup_discarded": warmup,
            "n_timed": len(latencies),
            "sync_before_timed": synced,
        },
    )
