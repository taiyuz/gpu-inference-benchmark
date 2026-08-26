#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-gpu-inference-benchmark:latest}"
mkdir -p artifacts
exec docker run --rm --gpus all \
  -v "$(pwd)/artifacts:/workspace/artifacts" \
  "$IMAGE" \
  "$@"
