"""README GPU results table stays empty until a real NVIDIA run."""

from __future__ import annotations

from pathlib import Path

from gpu_bench.metrics import skipped_result, summarize
from gpu_bench.report import DASH, readme_results_table

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_CELLS = slice(4, 9)  # mean ms, p50 ms, p99 ms, img/s, GPU MiB


def _results_table_body() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Results")
    rest = text[start:]
    next_h = rest.find("\n## ", 1)
    section = rest if next_h < 0 else rest[:next_h]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        if "---" in line or "Backend" in line:
            continue
        rows.append(line)
    return rows


def test_readme_results_table_uses_em_dashes_not_numbers() -> None:
    rows = _results_table_body()
    assert rows, "expected a placeholder GPU results table in README"
    for row in rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert len(cells) >= 9, row
        for cell in cells[NUMERIC_CELLS]:
            assert cell == DASH, f"invented GPU number in README: {row!r}"


def test_readme_does_not_claim_gpu_timings() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Taiyu Zhu" in text
    assert "template, not a measurement" in text
    assert "--require-cuda-events" in text
    assert "Filling the GPU table" in text


def test_readme_helper_skipped_rows_are_dashes() -> None:
    skipped = skipped_result(
        backend="tensorrt",
        precision="fp16",
        batch_size=8,
        graph=True,
        reason="no CUDA",
    )
    table = readme_results_table([skipped])
    assert "| tensorrt | fp16 | 8 | yes |" in table
    body = [ln for ln in table.splitlines() if ln.startswith("| tensorrt")]
    cells = [c.strip() for c in body[0].strip("|").split("|")]
    for cell in cells[NUMERIC_CELLS]:
        assert cell == DASH


def test_readme_helper_measured_row_uses_provided_samples() -> None:
    # Synthetic wall-clock samples in a unit test — not a GPU measurement.
    result = summarize(
        backend="pytorch",
        precision="fp32",
        batch_size=1,
        graph=False,
        n_warmup=0,
        latencies_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
        timing_backend="wall_clock",
        gpu_mem=None,
    )
    table = readme_results_table([result])
    assert "30.000" in table  # mean of the fixture list
    assert "wall_clock" not in table  # README-shaped table has no timing column
