"""Wall-clock timer on CPU; CUDA events are not used without CUDA."""

from __future__ import annotations

import time

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
