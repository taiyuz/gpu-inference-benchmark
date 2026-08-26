"""ONNX export (opset 17) with an artifacts/ cache. Paths are gitignored."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_bench.models import build_model


def onnx_path(artifacts_dir: Path, model: str, precision: str, batch_size: int) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / f"{model}_{precision}_b{batch_size}.onnx"


def export_onnx(
    *,
    model_name: str,
    precision: str,
    batch_size: int,
    artifacts_dir: Path,
    device: Any | None = None,
) -> Path:
    """Export a static-shape ONNX graph. Reuses the cache when present."""
    import torch

    path = onnx_path(artifacts_dir, model_name, precision, batch_size)
    if path.exists():
        return path

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if precision == "fp16" and device.type != "cuda":
            # Export still needs a fp16 module; keep it on CPU only if CUDA is absent
            # and the caller asked for fp16 — that path is skipped by backends.
            raise RuntimeError("FP16 ONNX export requires CUDA (.half() on device)")

    module = build_model(model_name, device, precision, pretrained=False)
    module.eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "input_names": ["input"],
        "output_names": ["output"],
        "opset_version": 17,
    }
    try:
        torch.onnx.export(module, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, dummy, str(path), **kwargs)
    return path
