from __future__ import annotations

from pathlib import Path

import pytest

from gpu_bench.config import BenchConfig


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def tiny_cpu_cfg(artifacts_dir: Path) -> BenchConfig:
    return BenchConfig(
        model="tiny",
        precision="fp32",
        batch_size=1,
        warmup=1,
        iters=3,
        graph=False,
        artifacts_dir=artifacts_dir,
    )
