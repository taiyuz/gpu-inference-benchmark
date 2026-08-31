# gpu-inference-benchmark

[![ci](https://github.com/taiyuz/gpu-inference-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/taiyuz/gpu-inference-benchmark/actions/workflows/ci.yml)

**Taiyu Zhu** ([`taiyuz`](https://github.com/taiyuz)) — recruiting-portfolio inference harness.

Same ResNet-50, same shapes: **export ONNX → ONNX Runtime → TensorRT 10**. FP16 / BF16, CUDA Graphs on a non-default stream, CUDA-event **p50 / p99**. The GitHub Actions badge is CPU-only pytest (no GPU, no TensorRT, no invented latency). GPU numbers belong in a CSV/JSON produced on an NVIDIA box, never in this README.

## CPU CI vs GPU / TensorRT (honest split)

| | GitHub Actions (this badge) | Local NVIDIA GPU |
|---|---|---|
| `uv run pytest` | yes — CPU, no CUDA | yes |
| PyTorch tiny dummy | wall-clock `perf_counter` | CUDA events |
| ONNX Runtime / TensorRT 10 | skipped (imports missing) | measured |
| FP16 / BF16 / CUDA Graphs | skipped on CPU | measured |
| Latency / throughput in this README | **none** | fill after a real run |
| `nvidia-smi` | never called | optional, not required by the harness |

CI workflow: `.github/workflows/ci.yml` — `uv sync --extra dev` then `uv run pytest -q` on `ubuntu-latest`. It does **not** install CUDA, TensorRT, or GPU torch. Do not read the green badge as a GPU result.

## How to run

CPU tests — this is what the CI badge runs:

```bash
uv sync --extra dev
uv run pytest
```

Same machine, still no GPU required (`ruff` is local; CI does not run it):

```bash
uv run ruff check src tests
uv run gpu-bench --list
uv run gpu-bench --dry-run --suite full
uv run gpu-bench --model tiny --backends pytorch --iters 5 --warmup 1
```

`--dry-run` prints the job list and captured env (hardware / driver / CUDA) without inference. `--list` prints backend availability. Both work without a GPU.

NVIDIA GPU (TensorRT / CUDA Graphs / CUDA-event p50/p99):

```bash
./scripts/run_gpu.sh --suite full --require-cuda-events --csv artifacts/results.csv --out artifacts/results.json
# or
docker run --rm --gpus all gpu-bench:local --suite full --require-cuda-events
```

Nsight Systems is optional. CUDA events are the primary timer:

```bash
./scripts/nsys_profile.sh
```

CSV / JSON schema (`gpu_bench.schema.COLUMNS`): hardware, driver, CUDA, backend, model, precision, batch, graphs, timing backend, mean / p50 / p90 / p99, throughput, GPU memory, skipped, notes. Skipped rows leave numeric cells **empty**. They are not estimates.

## Comparison ladder

`--suite full` walks this in order. Each step changes one thing.

1. PyTorch FP32 eager, compute-only (input already on device).
2. PyTorch FP16 (`.half()` on CUDA). CPU FP16 is skipped.
3. PyTorch BF16 (`torch.bfloat16` on CUDA). CPU BF16 is skipped.
4. ONNX Runtime, opset 17, CUDA EP + IO binding when a GPU is present. ORT BF16 and ORT CUDA Graphs are skipped here rather than silently faked.
5. TensorRT 10 FP32: `EXPLICIT_BATCH`, named I/O, `execute_async_v3`.
6. TensorRT 10 FP16 (`BuilderFlag.FP16`) and BF16 (`BuilderFlag.BF16`).
7. Batching 1 / 8 / 16 / 32.
8. CUDA Graphs at batch 1 and 8 for PyTorch and TensorRT, captured on a non-default stream.

## Results (not yet run)

Numbers appear here only after a real NVIDIA GPU run. This table is a template, not a measurement.

| Backend | Precision | Batch | Graphs | mean ms | p50 ms | p99 ms | img/s | GPU MiB |
|---|---|---:|:---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | 1 | no | — | — | — | — | — |
| pytorch | fp16 | 1 | no | — | — | — | — | — |
| pytorch | bf16 | 1 | no | — | — | — | — | — |
| onnx | fp32 | 1 | no | — | — | — | — | — |
| tensorrt | fp32 | 1 | no | — | — | — | — | — |
| tensorrt | fp16 | 1 | no | — | — | — | — | — |
| tensorrt | bf16 | 1 | no | — | — | — | — | — |
| pytorch | fp16 | 8 | yes | — | — | — | — | — |
| tensorrt | fp16 | 8 | yes | — | — | — | — | — |

Skipped rows print a reason. They are not estimates.

## Timing

On CUDA, latency is `torch.cuda.Event.elapsed_time` (GPU-side). Wall-clock `perf_counter` is a labeled fallback for CPU/CI and is **not** kernel time. `--require-cuda-events` refuses to run without CUDA events.

Default timed region is compute-only. `--include-transfer` copies into a static device buffer (pinned host optional) and D2H's the output.

## TensorRT 10 (explicit batch, named I/O, v3 enqueue)

Written against TensorRT **10.x**, not TRT 8 bindings.

- Implicit batch is **gone**. Networks use `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`. Batch is a tensor dimension; dynamic `N` is an optimization profile, not `max_batch_size`.
- `execute_async_v2(bindings)` is not used. TRT 10 requires `context.set_tensor_address(name, ptr)` for every I/O tensor, then `context.execute_async_v3(stream_handle)`.
- FP16 / BF16 are `BuilderFlag.FP16` and `BuilderFlag.BF16`. Workspace is `MemoryPoolType.WORKSPACE`.
- Engines are `build_serialized_network` → `deserialize_cuda_engine`, cached under `artifacts/`.

Sources used for that API: [Migrating Python Code from TensorRT 8.x to 10.x](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html) (implicit batch removed; `set_tensor_address` + `execute_async_v3`) and [Optimizing TensorRT Performance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html) (batching; CUDA Graphs when enqueue-bound).

## CUDA Graphs and streams

A CUDA Graph records a static DAG of GPU work and replays it with one launch. That is how you cut CPU launch overhead when shapes, pointers, and topology do not change. Capture is stream capture, not “launch kernels on the default stream and hope.”

This harness warms up on a **side stream**, captures with `torch.cuda.graph` (PyTorch) or the same capture around `execute_async_v3` (TensorRT), and uses a **non-default stream** by default. Capture is not valid on the default stream. `--default-stream` opts out. New data is `.copy_()` into the captured input.

Graphs help when CPU launch + TRT enqueue dominate (typical of small-batch ResNet). They do not help if you change shapes, allocate inside the region, or `synchronize()` during capture. Capture failure is reported; the run falls back to enqueue. It does not invent a speedup.

Sources used: [CUDA Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [CUDA Graph best practice for PyTorch](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html), [PyTorch CUDA Graph integration](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html), and the [PyTorch CUDA Graphs note](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (warmup on a side stream, static addresses, `copy_` before replay).

## Bottlenecks this harness is built to show

- **Launch overhead** at batch 1: many small cuDNN/TRT kernels. CUDA Graphs and TRT fusion exist for this (see the TRT optimization guide above).
- **Bandwidth vs compute**: compare compute-only vs `--include-transfer`. If adding H2D/D2H moves the number a lot, you were not looking at Tensor Cores. That is the distinction the [Roofline model](https://dl.acm.org/doi/10.1145/1498765.1498785) (Williams, Waterman, Patterson, CACM 2009) makes between arithmetic intensity and DRAM traffic. This repo does not plot a roofline; it refuses to mix copies into “kernel latency” by default.
- **Warmup**: first iterations allocate, autotune, and pick tactics. `--warmup` is required; do not publish it.
- **Precision**: FP16/BF16 cut bytes moved and enable Tensor Cores. BF16 keeps FP32-range exponents. Measure both. ORT BF16 is skipped here rather than silently running FP16.

## Honesty bar

- No latency, throughput, or memory figure in this README was measured on a GPU for this repo.
- Copyright and package author: Taiyu Zhu (`taiyuz`).
- NGC tag `24.08-py3` is a TRT 10-era image; the patch version is whatever `import tensorrt; tensorrt.__version__` prints, not a number invented here.
