"""Percentile and throughput math against a known latency list."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpu_bench.metrics import mean_ms, percentile, skipped_result, summarize, throughput_ips

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


def test_empty_latencies_raise() -> None:
    with pytest.raises(ValueError):
        mean_ms([])
    with pytest.raises(ValueError):
        percentile([], 50.0)


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
    assert math.isnan(result.throughput_ips)
    assert result.gpu_mem_bytes is None
    assert result.n_iter == 0
    assert result.latencies_ms == []
    assert result.notes == "no CUDA"
    assert result.timing_backend == "none"
