"""Model factories. Default ResNet-50; TinyConv for CPU/CI."""

from __future__ import annotations

from typing import Any

from gpu_bench.config import BenchConfig


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def resolve_device():
    torch, _ = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision: str, device) -> Any:
    torch, _ = _torch()
    if precision in {"fp16", "bf16"} and device.type != "cuda":
        raise RuntimeError(f"{precision} is only supported on CUDA in this harness")
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


class TinyConv:
    """Placeholder so tests can import the name; actual module is built below."""


def build_tiny(num_classes: int = 10):
    torch, nn = _torch()

    class _TinyConv(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return _TinyConv()


def build_resnet50(*, pretrained: bool = False):
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    return resnet50(weights=weights)


def load_model(cfg: BenchConfig, device=None, dtype=None):
    torch, _ = _torch()
    device = device or resolve_device()
    if dtype is None:
        dtype = torch.float32 if cfg.precision == "fp32" else resolve_dtype(cfg.precision, device)

    if cfg.model == "tiny":
        model = build_tiny()
    else:
        model = build_resnet50(pretrained=cfg.pretrained)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def random_input(cfg: BenchConfig, device=None, dtype=None, *, pinned: bool | None = None):
    torch, _ = _torch()
    device = device or resolve_device()
    if dtype is None:
        dtype = torch.float32 if device.type != "cuda" or cfg.precision == "fp32" else resolve_dtype(cfg.precision, device)
    pin = cfg.pinned if pinned is None else pinned
    host = torch.randn(
        cfg.batch_size,
        3,
        cfg.input_size,
        cfg.input_size,
        dtype=dtype if device.type == "cpu" else torch.float32,
    )
    if pin and device.type == "cuda":
        host = host.pin_memory()
    if device.type == "cuda":
        return host.to(device=device, dtype=dtype, non_blocking=True)
    return host.to(device=device)
