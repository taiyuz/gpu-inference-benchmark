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


def test_dry_run_default_prints_jobs(capsys) -> None:
    rc = main(["--dry-run", "--backends", "pytorch", "--precision", "fp32", "--batch", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "pytorch fp32 batch=1" in out
    assert "no inference" in out


def test_readme_table_flag_on_parser() -> None:
    from gpu_bench.cli import build_parser

    ns = build_parser().parse_args(["--dry-run", "--readme-table"])
    assert ns.readme_table is True
    assert ns.dry_run is True


def test_dry_run_batch_sweep_expands_jobs(capsys) -> None:
    rc = main(
        [
            "--dry-run",
            "--backends",
            "pytorch",
            "--precision",
            "fp32",
            "--batch",
            "1,8,16",
            "--warmup",
            "5",
            "--iters",
            "11",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "warmup=5 discarded" in out
    assert "iters=11 timed" in out
    assert "batch-sweep: 1,8,16" in out
    assert "pytorch fp32 batch=1" in out
    assert "pytorch fp32 batch=8" in out
    assert "pytorch fp32 batch=16" in out


def test_dry_run_single_batch_no_sweep_label(capsys) -> None:
    rc = main(["--dry-run", "--backends", "pytorch", "--batch", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "batch-sweep" not in out
    assert "warmup=" in out and "discarded" in out
