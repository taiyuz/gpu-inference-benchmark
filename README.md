# gpu-inference-benchmark

A recruiting-grade inference harness that compares **PyTorch → ONNX Runtime → TensorRT 10** on ResNet-50, with FP32 / FP16 / BF16, batching, CUDA events, and optional CUDA Graphs on a **non-default stream**.

This repository does **not** ship GPU timings. The results table below is a template. Fill it only after you run on a real NVIDIA GPU.

## Why this comparison

Eager PyTorch pays Python + dispatcher + per-kernel launch overhead on every iteration. ONNX Runtime's CUDA EP fuses some of that and, with IO binding, can keep tensors in device memory you already own. TensorRT 10 goes further: it builds an optimized engine (layer fusion, tactic selection, precision) and enqueues with `execute_async_v3` on named I/O tensors. CUDA Graphs then collapse many kernel launches into one graph launch when the topology is static.

That ladder is the point. It is not "TRT always wins." At batch 1, ResNet-50-224 is often **launch-overhead or bandwidth bound**; at large batch, Tensor Cores and TRT tactics matter more. Measure both. Isolate H2D copies from compute (`--include-transfer` vs default compute-only).

## Run

CPU dummy (CI path):

```bash
uv sync --extra dev --extra torch --extra onnx
uv run pytest
uv run gpu-bench --list
uv run gpu-bench --model tiny --backends pytorch --iters 5 --warmup 1
```

NVIDIA GPU (Docker):

```bash
./scripts/run_gpu.sh
# or, once the image is built:
docker run --rm --gpus all gpu-bench:local --suite full --require-cuda-events
```

Nsight Systems is optional. CUDA events are the primary timer; `nsys` is for traces:

```bash
./scripts/nsys_profile.sh
```

`--suite full` walks PyTorch / ORT / TRT across FP32, FP16, BF16, batches 1/8/16/32, then CUDA Graphs at batch 1 and 8 for PyTorch and TensorRT. ORT skips graphs and BF16 in this harness (honest skip, not a silent fallback).

## Results (not yet run)

Numbers appear here only after a real NVIDIA GPU run. This table is a template, not a measurement.

| Backend | Precision | Batch | Graphs | mean ms | p99 ms | img/s | GPU MiB |
|---|---|---:|:---:|---:|---:|---:|---:|
| pytorch | fp32 | 1 | no | — | — | — | — |
| pytorch | fp16 | 1 | no | — | — | — | — |
| pytorch | bf16 | 1 | no | — | — | — | — |
| onnx | fp32 | 1 | no | — | — | — | — |
| onnx | fp16 | 1 | no | — | — | — | — |
| tensorrt | fp32 | 1 | no | — | — | — | — |
| tensorrt | fp16 | 1 | no | — | — | — | — |
| tensorrt | bf16 | 1 | no | — | — | — | — |
| pytorch | fp16 | 8 | yes | — | — | — | — |
| tensorrt | fp16 | 8 | yes | — | — | — | — |

Skipped backends print a reason (no CUDA, no TensorRT wheel, graph capture failed, …). Skipped rows are not estimates.

## Timing

On CUDA, latency is `torch.cuda.Event.elapsed_time` (GPU-side). Wall-clock `perf_counter` is a labeled fallback for CPU/CI and is **not** kernel time. `--require-cuda-events` refuses to run without CUDA events.

Default timed region is **compute-only** (input already on device). `--include-transfer` copies into a static device buffer (pinned host optional) and D2H's the output, which is how you see whether you are looking at kernels or at PCIe.

## TensorRT 10 (explicit batch, named I/O, v3 enqueue)

This backend is written against TensorRT **10.x**, not the TRT 8 bindings array.

- Implicit batch is **gone**. Networks are created with `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`. The batch dimension is part of the tensor shape; dynamic `N` is an optimization profile, not `max_batch_size`.
- `execute_async_v2(bindings)` is not used. TRT 10 requires `context.set_tensor_address(name, ptr)` for every I/O tensor, then `context.execute_async_v3(stream_handle)`.
- FP16 / BF16 are `BuilderFlag.FP16` and `BuilderFlag.BF16` on `IBuilderConfig`. Workspace is `MemoryPoolType.WORKSPACE`, not the TRT 8 `max_workspace_size` setter.
- Engines are `build_serialized_network` → `deserialize_cuda_engine`, cached under `artifacts/`.

Sources used for that API: NVIDIA's [TRT 8.x → 10.x Python migration](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html) (implicit batch removed; `set_tensor_address` + `execute_async_v3`) and [TRT performance / best practices](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html) (batching, CUDA Graphs when enqueue-bound).

## CUDA Graphs and streams

A CUDA Graph records a static DAG of GPU work and replays it with one launch, which is the current way to cut CPU launch overhead when shapes, pointers, and topology do not change. Capture is stream capture (`cudaStreamBeginCapture` / `EndCapture`), not the 2010s "just launch kernels on the default stream and hope."

This harness:

- Warms up on a **side stream**, then captures with `torch.cuda.graph` (PyTorch) or the same capture around `execute_async_v3` (TensorRT).
- Uses a **non-default stream** by default. Capture is not valid on the default stream (implicit sync). `--default-stream` opts out and will skip or fail graph capture.
- Replays the same device addresses. New data is `.copy_()` into the captured input, matching PyTorch's CUDA Graph contract.

Graphs help when CPU launch + TRT enqueue dominate (typical of small-batch ResNet). They do not help if you change shapes, allocate inside the region, or `synchronize()` on the host during capture. Capture failure is reported and the run falls back to enqueue; it does not invent a speedup.

Sources used: [CUDA Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), [CUDA Graph best practice for PyTorch](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/cuda-graph.html), [PyTorch CUDA Graph integration](https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html), and the [PyTorch CUDA Graphs note](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (warmup on a side stream, static addresses, `copy_` before replay).

## Bottlenecks this harness is built to show

ResNet-50 at 224² is a decent systems probe because it is **not** always math-bound:

- **Launch overhead** at batch 1: many small cuDNN/TRT kernels. CUDA Graphs and TRT fusion exist specifically for this (see the TRT optimization guide above).
- **Bandwidth vs compute**: compare compute-only vs `--include-transfer`. If adding H2D/D2H moves the number a lot, you were not looking at Tensor Cores. That is the same distinction the [Roofline model](https://dl.acm.org/doi/10.1145/1498765.1498785) (Williams, Waterman, Patterson, CACM 2009) makes between arithmetic intensity and DRAM traffic; this repo does not plot a roofline, it just refuses to mix copies into "kernel latency" by default.
- **Warmup**: first iterations allocate, autotune (cuDNN benchmark), and JIT tactics. `--warmup` is required; do not publish it.
- **Precision**: FP16/BF16 cut bytes moved and enable Tensor Cores. BF16 keeps FP32-range exponents; FP16 can be faster on some parts and numerically tighter on others. Measure both. ORT BF16 is skipped here rather than silently running FP16.

## Layout

```
src/gpu_bench/
  timing.py          CUDA events, labeled wall-clock fallback
  metrics.py         mean / p50 / p90 / p99 / throughput / GPU mem
  backends/pytorch.py
  backends/onnxrt.py     CUDA EP + IO binding
  backends/tensorrt.py   TRT 10 explicit batch, v3 enqueue, optional graph
  cli.py             gpu-bench
tests/               CPU dummy tests (no GPU runner)
.github/workflows/ci.yml
Dockerfile           NGC TensorRT 24.08 (TRT 10.x era; confirm with trt.__version__)
```

CI installs CPU torch + onnxruntime and runs pytest. It never calls `nvidia-smi`.

## Honesty bar

- No latency, throughput, or memory figure in this README was measured on a GPU for this repo.
- Optional extras (`torch`, `onnx`, `cuda`, `tensorrt`) keep CI free of TRT/GPU wheels.
- NGC tag `24.08-py3` is a TRT 10-era image; the patch version is whatever `import tensorrt; tensorrt.__version__` prints on that image, not a number invented here.
