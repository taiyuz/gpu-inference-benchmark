"""Stable CSV/JSON schema for a run. Hardware fields are environment, not timings."""

from __future__ import annotations

import csv
import math
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_bench.metrics import RunResult

# Column order is the published schema. Latency cells stay empty when skipped.
COLUMNS = [
    "timestamp",
    "hostname",
    "hardware",
    "driver",
    "cuda",
    "pytorch",
    "tensorrt",
    "backend",
    "model",
    "precision",
    "batch_size",
    "graph",
    "include_transfer",
    "timing_backend",
    "n_warmup",
    "n_iter",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p99_ms",
    "throughput_ips",
    "gpu_mem_bytes",
    "skipped",
    "notes",
]


def collect_env() -> dict[str, str]:
    """CPU-safe machine record. Never calls nvidia-smi; missing CUDA is ``n/a``."""
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": "cpu",
        "driver": "n/a",
        "cuda": "n/a",
        "pytorch": "not installed",
        "tensorrt": "not installed",
    }
    try:
        import torch

        env["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            env["hardware"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda or "n/a"
            # Driver version would need nvidia-smi; this harness does not call it.
            env["driver"] = "n/a"
    except Exception:
        pass
    try:
        import tensorrt as trt

        env["tensorrt"] = str(getattr(trt, "__version__", "unknown"))
    except Exception:
        pass
    return env


def _num(value: float | int | None, *, skipped: bool) -> str:
    if skipped or value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def result_row(
    result: RunResult,
    *,
    env: dict[str, str] | None = None,
    model: str = "",
    include_transfer: bool = False,
    timestamp: str | None = None,
) -> dict[str, str]:
    env = env or collect_env()
    skipped = bool(result.skipped)
    return {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "hostname": env.get("hostname", ""),
        "hardware": env.get("hardware", "cpu"),
        "driver": env.get("driver", "n/a"),
        "cuda": env.get("cuda", "n/a"),
        "pytorch": env.get("pytorch", ""),
        "tensorrt": env.get("tensorrt", ""),
        "backend": result.backend,
        "model": model,
        "precision": result.precision,
        "batch_size": str(result.batch_size),
        "graph": str(bool(result.graph)).lower(),
        "include_transfer": str(bool(include_transfer)).lower(),
        "timing_backend": result.timing_backend,
        "n_warmup": str(result.n_warmup),
        "n_iter": str(result.n_iter),
        "mean_ms": _num(result.mean_ms, skipped=skipped),
        "p50_ms": _num(result.p50_ms, skipped=skipped),
        "p90_ms": _num(result.p90_ms, skipped=skipped),
        "p99_ms": _num(result.p99_ms, skipped=skipped),
        "throughput_ips": _num(result.throughput_ips, skipped=skipped),
        "gpu_mem_bytes": _num(result.gpu_mem_bytes, skipped=skipped),
        "skipped": str(skipped).lower(),
        "notes": result.notes,
    }


def write_csv(
    path: Path,
    results: list[RunResult],
    *,
    env: dict[str, str] | None = None,
    model: str = "",
    include_transfer: bool = False,
) -> None:
    env = env or collect_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(
                result_row(
                    result,
                    env=env,
                    model=model,
                    include_transfer=include_transfer,
                    timestamp=stamp,
                )
            )


def json_payload(
    results: list[RunResult],
    *,
    env: dict[str, str] | None = None,
    model: str = "",
    include_transfer: bool = False,
) -> dict[str, Any]:
    env = env or collect_env()
    stamp = datetime.now(timezone.utc).isoformat()
    return {
        "disclaimer": (
            "Values are measurements from this process only. "
            "Empty/NaN/skipped rows are not measurements. Do not backfill."
        ),
        "env": env,
        "schema": list(COLUMNS),
        "model": model,
        "include_transfer": include_transfer,
        "results": [r.to_dict() for r in results],
        "rows": [
            result_row(
                r,
                env=env,
                model=model,
                include_transfer=include_transfer,
                timestamp=stamp,
            )
            for r in results
        ],
    }
