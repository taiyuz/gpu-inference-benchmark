"""Inference backend registry."""

from __future__ import annotations

from gpu_bench.backends.onnxrt import OnnxRuntimeBackend
from gpu_bench.backends.pytorch import PyTorchBackend
from gpu_bench.backends.tensorrt import TensorRTBackend

BACKENDS = {
    "pytorch": PyTorchBackend(),
    "onnx": OnnxRuntimeBackend(),
    "tensorrt": TensorRTBackend(),
}

__all__ = ["BACKENDS", "OnnxRuntimeBackend", "PyTorchBackend", "TensorRTBackend"]
