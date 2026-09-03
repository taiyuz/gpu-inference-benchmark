"""Test-facing APIs must import on CPU CI without CUDA or torch being present."""

from __future__ import annotations

import inspect


def test_timer_metrics_export_and_model_factory_import() -> None:
    from gpu_bench.export import export_onnx
    from gpu_bench.harness import warmup_then_measure
    from gpu_bench.metrics import mean_ms, percentile, summarize
    from gpu_bench.models import build_model
    from gpu_bench.timing import WallClockTimer

    assert callable(build_model)
    assert callable(mean_ms)
    assert callable(percentile)
    assert callable(summarize)
    assert callable(export_onnx)
    assert callable(warmup_then_measure)
    timer = WallClockTimer()
    assert timer.name == "wall_clock"
    assert "NOT a CUDA event" in timer.notes


def test_export_onnx_is_keyword_only() -> None:
    from gpu_bench.export import export_onnx

    for name, param in inspect.signature(export_onnx).parameters.items():
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
