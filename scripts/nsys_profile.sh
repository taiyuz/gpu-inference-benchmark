#!/usr/bin/env bash
set -euo pipefail
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found. Install Nsight Systems or run inside an NGC container that provides it." >&2
  exit 1
fi
mkdir -p artifacts
exec nsys profile -o artifacts/nsys --stat=true -- \
  gpu-bench --backends pytorch --precision fp16 --batch 8 --graph
