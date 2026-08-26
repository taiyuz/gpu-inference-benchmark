#!/usr/bin/env bash
set -euo pipefail
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not on PATH. Install Nsight Systems, or run inside an NGC TensorRT image that ships it." >&2
  exit 1
fi
mkdir -p artifacts
# Trace CUDA + NVTX. This is optional; the harness already times with CUDA events.
nsys profile \
  --output=artifacts/nsys-gpu-bench \
  --force-overwrite=true \
  --stats=true \
  --trace=cuda,nvtx \
  gpu-bench --backends pytorch,tensorrt --precision fp16 --batch 8 --graph --iters 50
