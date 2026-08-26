"""Model constructors. pretrained defaults to False so CPU/CI never downloads."""

from __future__ import annotations

from typing import Any


def output_features(model_name: str) -> int:
    if model_name == "tiny":
        return 10
    if model_name == "resnet50":
        return 1000
    raise ValueError(f"unsupported model: {model_name}")


def _maybe_half(model: Any, device: Any, precision: str) -> Any:
    import torch

    dev = torch.device(device)
    model = model.to(dev)
    if precision == "fp16":
        if dev.type != "cuda":
            raise RuntimeError("FP16 (.half()) is only supported on CUDA")
        model = model.half()
    return model


_TINY_CONV_CLS: Any = None


def _tiny_conv_class() -> Any:
    """Build (and cache) TinyConv without importing torch at module load."""
    global _TINY_CONV_CLS
    if _TINY_CONV_CLS is not None:
        return _TINY_CONV_CLS
    import torch
    from torch import nn

    class TinyConv(nn.Module):
        """2 conv + pool + linear. 3x224x224 in, 10-class out. CPU test dummy."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    _TINY_CONV_CLS = TinyConv
    return TinyConv


def __getattr__(name: str) -> Any:
    if name == "TinyConv":
        return _tiny_conv_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tiny_conv(device: Any, precision: str) -> Any:
    """2 conv + pool + linear. 3x224x224 in, 10-class out. CPU test dummy."""
    TinyConv = _tiny_conv_class()
    model = TinyConv()
    model.eval()
    return _maybe_half(model, device, precision)


def resnet50(device: Any, precision: str, pretrained: bool = False) -> Any:
    """torchvision ResNet-50. pretrained=False avoids a network fetch on CPU/CI."""
    from torchvision.models import ResNet50_Weights
    from torchvision.models import resnet50 as _resnet50

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = _resnet50(weights=weights)
    model.eval()
    return _maybe_half(model, device, precision)


def build_model(
    name: str,
    device: Any,
    precision: str,
    *,
    pretrained: bool = False,
) -> Any:
    if name == "tiny":
        return tiny_conv(device, precision)
    if name == "resnet50":
        return resnet50(device, precision, pretrained=pretrained)
    raise ValueError(f"unsupported model: {name}")
