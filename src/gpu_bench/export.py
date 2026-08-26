"""PyTorch → ONNX export. TensorRT engines are built from this ONNX."""

from __future__ import annotations

from pathlib import Path

from gpu_bench.config import BenchConfig
from gpu_bench.models import load_model, random_input, resolve_device, resolve_dtype

ONNX_OPSET = 17


def onnx_path(cfg: BenchConfig) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    name = f"{cfg.model}_{cfg.precision}_b{cfg.batch_size}_op{ONNX_OPSET}.onnx"
    return cfg.artifacts_dir / name


def export_onnx(cfg: BenchConfig, path: Path | None = None) -> Path:
    import torch

    path = path or onnx_path(cfg)
    if path.exists():
        return path

    device = resolve_device()
    if cfg.precision == "fp16" and device.type != "cuda":
        dtype = torch.float32
    else:
        dtype = resolve_dtype(cfg.precision, device) if device.type == "cuda" else torch.float32

    model = load_model(cfg, device=device, dtype=dtype)
    dummy = random_input(cfg, device=device, dtype=dtype, pinned=False)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=ONNX_OPSET,
        dynamo=False,
        do_constant_folding=True,
    )
    return path
