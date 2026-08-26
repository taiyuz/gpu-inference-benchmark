"""JSON and markdown reports. Unrun cells are an em dash, never a fake number."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_bench.metrics import RunResult
from gpu_bench.runner import Skip

EM = "\u2014"  # —


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return EM
    return f"{value:.3f}"


def _fmt_ips(value: float | None) -> str:
    if value is None:
        return EM
    return f"{value:.1f}"


def _fmt_mem(value: int | None) -> str:
    if value is None:
        return EM
    return str(value)


def render_markdown(results: list[RunResult], skips: list[Skip]) -> str:
    lines = [
        "# GPU inference benchmark report",
        "",
        "Cells are measured values or " + EM + ". Unrun / skipped jobs are never filled with estimates.",
        "",
        "| Backend | Precision | Batch | Graphs | mean ms | p99 ms | img/s | GPU mem |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            "| {backend} | {precision} | {batch} | {graphs} | {mean} | {p99} | {ips} | {mem} |".format(
                backend=r.backend,
                precision=r.precision,
                batch=r.batch_size,
                graphs="yes" if r.graph else "no",
                mean=_fmt_ms(r.mean_ms),
                p99=_fmt_ms(r.p99_ms),
                ips=_fmt_ips(r.throughput_ips),
                mem=_fmt_mem(r.gpu_mem_bytes),
            )
        )
    for s in skips:
        lines.append(
            "| {backend} | {precision} | {batch} | {graphs} | {em} | {em} | {em} | {em} |".format(
                backend=s.backend,
                precision=s.precision,
                batch=s.batch_size,
                graphs="yes" if s.graph else "no",
                em=EM,
            )
        )
    if skips:
        lines.extend(["", "## Skips", ""])
        for s in skips:
            graph = "graph" if s.graph else "eager"
            lines.append(
                f"- `{s.backend}` {s.precision} batch={s.batch_size} ({graph}): {s.reason}"
            )
    if results:
        lines.extend(["", "## Timing backends", ""])
        for r in results:
            lines.append(
                f"- `{r.backend}` {r.precision} batch={r.batch_size} "
                f"graph={r.graph}: {r.timing_backend} — {r.notes}"
            )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, results: list[RunResult], skips: list[Skip]) -> None:
    payload = {
        "results": [r.to_dict() for r in results],
        "skips": [s.to_dict() for s in skips],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, results: list[RunResult], skips: list[Skip]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results, skips))
