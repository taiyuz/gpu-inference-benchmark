"""CSV/JSON schema and env capture work on CPU without CUDA."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from gpu_bench.metrics import skipped_result, summarize
from gpu_bench.schema import COLUMNS, collect_env, json_payload, result_row, write_csv

LATENCIES = [10.0, 20.0, 30.0, 40.0, 50.0]

STUB_ENV = {
    "hostname": "ci",
    "hardware": "cpu",
    "driver": "n/a",
    "cuda": "n/a",
    "pytorch": "not installed",
    "tensorrt": "not installed",
}


def _measured():
    return summarize(
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


def test_schema_has_recruiter_columns() -> None:
    required = {
        "hardware",
        "driver",
        "batch_size",
        "precision",
        "p50_ms",
        "p99_ms",
        "stdev_ms",
        "notes",
    }
    assert required.issubset(set(COLUMNS))
    assert COLUMNS[0] == "timestamp"
    assert COLUMNS.index("stdev_ms") == COLUMNS.index("p99_ms") + 1


def test_collect_env_cpu_safe_no_nvidia_smi() -> None:
    env = collect_env()
    for key in ("hardware", "driver", "cuda", "pytorch", "tensorrt", "hostname"):
        assert key in env
        assert env[key] != ""
    # Driver version would need nvidia-smi; this harness never calls it.
    assert env["driver"] == "n/a"


def test_measured_row_has_p50_p99() -> None:
    row = result_row(_measured(), env=STUB_ENV, model="tiny", timestamp="t0")
    assert row["hardware"] == "cpu"
    assert row["driver"] == "n/a"
    assert row["batch_size"] == "8"
    assert row["precision"] == "fp32"
    assert row["p50_ms"]
    assert row["p99_ms"]
    assert row["stdev_ms"]
    assert row["skipped"] == "false"
    assert float(row["p50_ms"]) == 30.0
    assert row["notes"] == "unit test"
    assert set(row) == set(COLUMNS)


def test_skipped_row_does_not_invent_latency() -> None:
    skipped = skipped_result(
        backend="tensorrt",
        precision="fp16",
        batch_size=8,
        graph=True,
        reason="no CUDA",
    )
    row = result_row(skipped, env=STUB_ENV, model="resnet50", timestamp="t0")
    assert row["skipped"] == "true"
    assert row["mean_ms"] == ""
    assert row["p50_ms"] == ""
    assert row["p99_ms"] == ""
    assert row["stdev_ms"] == ""
    assert row["throughput_ips"] == ""
    assert row["gpu_mem_bytes"] == ""
    assert "no CUDA" in row["notes"]
    assert math.isnan(skipped.mean_ms)


def test_write_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    skipped = skipped_result(
        backend="onnx",
        precision="bf16",
        batch_size=1,
        graph=False,
        reason="ORT bf16 skipped",
    )
    write_csv(path, [_measured(), skipped], env=STUB_ENV, model="tiny")
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS
        got = list(reader)
    assert len(got) == 2
    assert got[0]["p50_ms"]
    assert got[0]["stdev_ms"]
    assert got[1]["p50_ms"] == ""
    assert got[1]["stdev_ms"] == ""
    assert got[1]["skipped"] == "true"


def test_json_payload_keeps_results_and_rows() -> None:
    payload = json_payload([_measured()], env=STUB_ENV, model="tiny")
    assert payload["schema"] == list(COLUMNS)
    assert payload["env"]["hardware"] == "cpu"
    assert "Do not backfill" in payload["disclaimer"]
    dumped = json.dumps(payload, default=str)
    assert "30" in dumped
    assert "stdev_ms" in payload["schema"]
