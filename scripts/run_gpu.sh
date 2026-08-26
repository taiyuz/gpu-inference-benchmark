#!/usr/bin/env bash
set -euo pipefail
# GPU run via NVIDIA Container Toolkit. Host must have a driver + nvidia-container-toolkit.
IMAGE="${IMAGE:-gpu-bench:local}"
docker build -t "$IMAGE" "$(dirname "$0")/.."
docker run --rm --gpus all -v "$(pwd)/artifacts:/workspace/artifacts" \
  "$IMAGE" --suite full --require-cuda-events --out artifacts/report.json --md artifacts/report.md
