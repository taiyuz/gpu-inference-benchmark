"""Percentile and throughput math against a known latency list."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_bench.metrics import (
    mean_ms,
    percentile,
    sample_stdev_ms,
    skipped_result,
    summarize,
    tail_warning,
    throughput_ips,
    validate_latencies,
)

LATENCIES = [10.0, 20.0, 30.0, 40.0, 50.0]


def test_mean_known_list() -> None:
    assert mean_ms(LATENCIES) == 30.0


def test_p50_p99_match_numpy() -> None:
    assert percentile(LATENCIES, 50.0) == float(np.percentile(LATENCIES, 50.0))
    assert percentile(LATENCIES, 99.0) == float(np.percentile(LATENCIES, 99.0))
    assert percentile(LATENCIES, 50.0) == 30.0


def test_throughput_from_mean() -> None:
    # batch 8, mean 4 ms -> 8 / 0.004 = 2000 img/s
    assert throughput_ips(8, 4.0) == 2000.0


def test_summarize_records_timing_backend() -> None:
    result = summarize(
        backend="pytorch",
        precision="fp32",
        batch_size=8,
        graph=False,
        n_warmup=0,
        n_iter=len(LATENCIES),
        latencies_ms=LATENCIES,
        timing_backend="wall_clock",
        notes="unit test",
        gpu_mem=None,
    )
    assert result.mean_ms == 30.0
    assert result.p50_ms == 30.0
    assert result.p99_ms == float(np.percentile(LATENCIES, 99.0))
    assert result.throughput_ips == throughput_ips(8, 30.0)
    assert result.gpu_mem_bytes is None
    assert result.timing_backend == "wall_clock"
    assert result.stdev_ms == pytest.approx(float(np.std(LATENCIES, ddof=1)))
    assert "tail_warning" in result.extra


def test_empty_latencies_raise() -> None:
    with pytest.raises(ValueError):
        mean_ms([])
    with pytest.raises(ValueError):
        percentile([], 50.0)
    with pytest.raises(ValueError):
        validate_latencies([])


def test_validate_rejects_non_finite_and_negative() -> None:
    with pytest.raises(ValueError, match="not finite"):
        validate_latencies([1.0, float("nan")])
    with pytest.raises(ValueError, match="not finite"):
        validate_latencies([1.0, float("inf")])
    with pytest.raises(ValueError, match="negative"):
        validate_latencies([1.0, -0.01])
    with pytest.raises(ValueError, match="not finite"):
        summarize(
            backend="pytorch",
            precision="fp32",
            batch_size=1,
            graph=False,
            n_warmup=0,
            latencies_ms=[1.0, float("nan")],
            timing_backend="wall_clock",
        )


def test_sample_stdev_known_list() -> None:
    assert sample_stdev_ms(LATENCIES) == pytest.approx(float(np.std(LATENCIES, ddof=1)))
    assert math.isnan(sample_stdev_ms([4.0]))


def test_throughput_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        throughput_ips(8, 0.0)
    with pytest.raises(ValueError):
        throughput_ips(8, float("nan"))


def test_tail_warning_for_small_n() -> None:
    assert tail_warning(0) == "no samples"
    warn = tail_warning(5, 99.0)
    assert warn is not None
    assert "under-sampled" in warn
    assert tail_warning(100, 99.0) is None


def test_skipped_result_is_not_a_measurement() -> None:
    result = skipped_result(
        backend="tensorrt",
        precision="fp16",
        batch_size=1,
        graph=True,
        reason="no CUDA",
    )
    assert result.skipped is True
    assert math.isnan(result.mean_ms)
    assert math.isnan(result.p50_ms)
    assert math.isnan(result.p99_ms)
    assert math.isnan(result.stdev_ms)
    assert math.isnan(result.throughput_ips)
    assert result.gpu_mem_bytes is None
    assert result.n_iter == 0
    assert result.latencies_ms == []
    assert result.notes == "no CUDA"
    assert result.timing_backend == "none"
