"""Warmup vs timed split must hold on CPU without a GPU."""

from __future__ import annotations

import pytest

from gpu_bench.harness import warmup_then_measure
from gpu_bench.timing import WallClockTimer


def test_warmup_calls_are_discarded_not_timed() -> None:
    calls: list[str] = []
    timer = WallClockTimer()

    def step() -> None:
        calls.append("step")

    # Patch measure so we can count timed vs total step invocations.
    measured: list[int] = []
    original = timer.measure

    def counting_measure(fn):
        measured.append(1)
        return original(fn)

    timer.measure = counting_measure  # type: ignore[method-assign]

    result = warmup_then_measure(step, timer, warmup=4, iters=3, sync_fn=None)
    assert result.n_warmup == 4
    assert result.n_iter == 3
    assert len(result.latencies_ms) == 3
    assert len(measured) == 3  # only timed iters call measure
    assert len(calls) == 4 + 3  # warmup + timed
    assert result.extra["warmup_discarded"] is True
    assert result.extra["n_warmup_discarded"] == 4
    assert result.extra["n_timed"] == 3
    assert result.timing_backend == "wall_clock"
    assert any("discarded" in n for n in result.notes)
    assert any("NOT CUDA-event" in n for n in result.notes)


def test_sync_runs_once_between_warmup_and_timed() -> None:
    order: list[str] = []
    timer = WallClockTimer()

    def step() -> None:
        order.append("step")

    def sync() -> None:
        order.append("sync")

    result = warmup_then_measure(step, timer, warmup=2, iters=2, sync_fn=sync)
    assert result.sync_before_timed is True
    # warmup, warmup, sync, timed, timed
    assert order == ["step", "step", "sync", "step", "step"]
    assert any("sync after warmup" in n for n in result.notes)


def test_zero_warmup_ok() -> None:
    timer = WallClockTimer()
    n = {"c": 0}

    def step() -> None:
        n["c"] += 1

    result = warmup_then_measure(step, timer, warmup=0, iters=2)
    assert result.n_warmup == 0
    assert n["c"] == 2
    assert len(result.latencies_ms) == 2


def test_rejects_bad_counts() -> None:
    timer = WallClockTimer()

    def step() -> None:
        return None

    with pytest.raises(ValueError, match="warmup"):
        warmup_then_measure(step, timer, warmup=-1, iters=1)
    with pytest.raises(ValueError, match="iters"):
        warmup_then_measure(step, timer, warmup=0, iters=0)
