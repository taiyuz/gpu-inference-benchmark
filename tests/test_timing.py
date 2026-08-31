"""Wall-clock timer on CPU; CUDA events are not used without CUDA."""

from __future__ import annotations

import time

import pytest

from gpu_bench.timing import WallClockTimer, make_timer


def test_wall_clock_positive_ms() -> None:
    timer = WallClockTimer()
    timer.start()
    time.sleep(0.01)
    elapsed = timer.stop()
    assert elapsed > 0.0
    assert timer.name == "wall_clock"
    assert "NOT a CUDA event" in timer.notes


def test_make_timer_cpu_is_wall_clock() -> None:
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda:
        timer = make_timer()
        assert timer.name == "cuda_events"
        return
    timer = make_timer(require_cuda_events=False)
    assert timer.name == "wall_clock"
    assert "NOT a CUDA event" in timer.notes


def test_measure_runs_fn_and_returns_ms() -> None:
    calls: list[int] = []
    timer = WallClockTimer()

    def fn() -> None:
        calls.append(1)
        time.sleep(0.005)

    ms = timer.measure(fn)
    assert calls == [1]
    assert ms > 0.0


def test_stop_before_start_raises() -> None:
    timer = WallClockTimer()
    with pytest.raises(RuntimeError):
        timer.stop()
