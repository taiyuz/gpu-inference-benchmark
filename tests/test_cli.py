from __future__ import annotations

from gpu_bench.cli import main


def test_list_backends_exits_zero() -> None:
    rc = main(["--list"])
    assert rc == 0
