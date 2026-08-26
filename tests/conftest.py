"""Shared fixtures. Artifacts land under tmp_path so tests never touch repo artifacts/."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_bench.config import RunConfig


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def tiny_cpu_cfg(artifacts_dir: Path) -> RunConfig:
    return RunConfig(
        backend="pytorch",
        precision="fp32",
        batch_size=1,
        n_warmup=1,
        n_iter=3,
        graph=False,
        model="tiny",
        artifacts_dir=artifacts_dir,
    )
