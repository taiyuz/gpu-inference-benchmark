#!/usr/bin/env bash
# Run the suite on a real NVIDIA GPU (needs docker --gpus all).
# This is not CI. After a successful run, copy skipped=false CSV cells into
# the README Results table, or use --readme-table. Leave unmeasured cells as —.
#
#   ./scripts/run_gpu.sh --suite full --require-cuda-events \
#     --csv artifacts/results.csv --out artifacts/results.json --readme-table
set -euo pipefail
IMAGE="${IMAGE:-gpu-inference-benchmark:latest}"
mkdir -p artifacts
exec docker run --rm --gpus all \
  -v "$(pwd)/artifacts:/workspace/artifacts" \
  "$IMAGE" \
  "$@"
