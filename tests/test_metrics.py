from __future__ import annotations

import math

import pytest

from gpu_bench.metrics import percentiles, skipped_result, summarize, throughput_ips


def test_percentiles_known_vector() -> None:
    xs = [float(i) for i in range(1, 101)]
    mean, p50, p90, p99 = percentiles(xs)
    assert mean == pytest.approx(50.5)
    assert p50 == pytest.approx(50.5)
    assert p90 == pytest.approx(90.1)
    assert p99 == pytest.approx(99.01)


def test_throughput() -> None:
    assert throughput_ips(8, 2.0) == pytest.approx(4000.0)


def test_summarize_and_skip_do_not_invent_memory() -> None:
    result = summarize(
        backend="pytorch",
        precision="fp32",
        batch_size=2,
        graph=False,
        n_warmup=0,
        latencies_ms=[1.0, 2.0, 3.0],
        timing_backend="wall_clock",
        gpu_mem_bytes=None,
    )
    assert result.gpu_mem_bytes is None
    assert result.mean_ms == pytest.approx(2.0)
    skip = skipped_result(
        backend="tensorrt",
        precision="fp16",
        batch_size=8,
        graph=True,
        reason="no GPU",
    )
    assert skip.skipped
    assert math.isnan(skip.mean_ms)
    assert skip.gpu_mem_bytes is None
    assert skip.latencies_ms == []
