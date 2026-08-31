from __future__ import annotations

from gpu_bench.cli import main


def test_list_backends_exits_zero() -> None:
    rc = main(["--list"])
    assert rc == 0


def test_dry_run_full_suite_prints_jobs(capsys) -> None:
    rc = main(["--dry-run", "--suite", "full"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "hardware=" in out
    assert "pytorch" in out
    assert "onnx" in out
    assert "tensorrt" in out
    assert "graph=True" in out
