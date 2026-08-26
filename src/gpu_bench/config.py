"""Run configuration for a single backend invocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    backend: str
    precision: str
    batch_size: int
    n_warmup: int = 20
    n_iter: int = 100
    graph: bool = False
    include_transfer: bool = False
    pinned: bool = False
    model: str = "resnet50"
    require_cuda_events: bool = False
    artifacts_dir: Path = Path("artifacts")
    workspace_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        self.precision = self.precision.lower()
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.model not in {"resnet50", "tiny"}:
            raise ValueError(f"unsupported model: {self.model}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.artifacts_dir = Path(self.artifacts_dir)
