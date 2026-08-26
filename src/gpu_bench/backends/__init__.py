"""Backend registry. Missing optional deps register as unavailable, never crash import."""

from __future__ import annotations

from gpu_bench.backends.base import REGISTRY, Backend, get_backend, register
from gpu_bench.backends.pytorch import PyTorchBackend

register(PyTorchBackend())

try:
    from gpu_bench.backends.onnxrt import OnnxRuntimeBackend
except ImportError:
    OnnxRuntimeBackend = None  # type: ignore[misc, assignment]
else:
    register(OnnxRuntimeBackend())

try:
    from gpu_bench.backends.tensorrt import TensorRTBackend
except ImportError:
    TensorRTBackend = None  # type: ignore[misc, assignment]
else:
    register(TensorRTBackend())

__all__ = [
    "REGISTRY",
    "Backend",
    "OnnxRuntimeBackend",
    "PyTorchBackend",
    "TensorRTBackend",
    "get_backend",
    "register",
]
