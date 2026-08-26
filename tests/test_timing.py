from __future__ import annotations

from gpu_bench.timing import WallClockTimer, cuda_is_available, make_timer


def test_wall_clock_is_labeled_and_positive() -> None:
    timer = WallClockTimer()
    assert timer.backend.name == "wall_clock"
    assert "NOT a CUDA event" in timer.backend.notes
    ms = timer.measure(lambda: sum(range(10000)))
    assert ms >= 0.0


def test_make_timer_falls_back_without_cuda() -> None:
    if cuda_is_available():
        timer = make_timer(require_cuda_events=False)
        assert timer.backend.name == "cuda_event"
        return
    timer = make_timer(require_cuda_events=False)
    assert timer.backend.name == "wall_clock"
