"""JSON + markdown reports. Unmeasured cells stay as em-dash, never invented numbers."""

from __future__ import annotations

import json
import math
from pathlib import Path

from gpu_bench.metrics import RunResult

DASH = "—"


def _fmt(value: float | int | None, *, integer: bool = False) -> str:
    if value is None:
        return DASH
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return DASH
    if integer:
        return str(int(value))
    return f"{value:.3f}"


def _mem_mib(nbytes: int | None) -> str:
    if nbytes is None:
        return DASH
    return f"{nbytes / (1024 ** 2):.1f}"


def markdown_table(results: list[RunResult]) -> str:
    header = (
        "| Backend | Precision | Batch | Graphs | mean ms | p99 ms | img/s | GPU MiB | Timing | Notes |\n"
        "|---|---|---:|:---:|---:|---:|---:|---:|---|---|\n"
    )
    rows = []
    for r in results:
        if r.skipped:
            rows.append(
                f"| {r.backend} | {r.precision} | {r.batch_size} | {r.graph} "
                f"| {DASH} | {DASH} | {DASH} | {DASH} | skipped | {r.notes} |"
            )
            continue
        rows.append(
            f"| {r.backend} | {r.precision} | {r.batch_size} | {r.graph} "
            f"| {_fmt(r.mean_ms)} | {_fmt(r.p99_ms)} | {_fmt(r.throughput_ips)} "
            f"| {_mem_mib(r.gpu_mem_bytes)} | {r.timing_backend} | {r.notes} |"
        )
    return header + "\n".join(rows) + "\n"


def write_json(path: Path, results: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disclaimer": (
            "Values are measurements from this process only. "
            "Empty/NaN/skipped rows are not measurements. Do not backfill."
        ),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_markdown(path: Path, results: list[RunResult], *, title: str = "gpu-bench report") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# {title}",
        "",
        "Unmeasured or skipped rows use `—`. These are not estimates.",
        "",
        markdown_table(results),
    ]
    path.write_text("\n".join(body))
