"""Shared benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    model: str = "resnet50"
    precision: str = "fp32"
    batch_size: int = 1
    warmup: int = 20
    iters: int = 100
    graph: bool = False
    include_transfer: bool = False
    pinned: bool = True
    pretrained: bool = False
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    require_cuda_events: bool = False
    input_size: int = 224
    workspace_bytes: int = 1 << 30  # 1 GiB TensorRT workspace cap
    seed: int = 0
    use_nondefault_stream: bool = True  # required for CUDA Graph capture

    def __post_init__(self) -> None:
        self.precision = self.precision.lower()
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.model not in {"resnet50", "tiny"}:
            raise ValueError(f"unsupported model: {self.model}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.warmup < 0 or self.iters < 1:
            raise ValueError("warmup must be >= 0 and iters must be >= 1")
        self.artifacts_dir = Path(self.artifacts_dir)
