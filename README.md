# gpu-bench

A recruiting-grade GPU **inference runtime** comparison: the same model, same shapes, and the same timing protocol across PyTorch eager, PyTorch CUDA Graphs, ONNX Runtime (IO binding), and TensorRT 10. The point is not a leaderboard screenshot — it is to show how you isolate compute from copies, when FP16 and Graphs help, and how TRT 10's API differs from TRT 8. Numbers in this repository appear only after a real NVIDIA GPU run.

## Comparison ladder

Run these in order on one GPU. Each step changes one thing.

1. **PyTorch FP32 eager** — baseline, compute-only timing, input already on device.
2. **PyTorch FP16 eager** — `.half()` on CUDA; same shapes. CPU FP16 is skipped.
3. **ONNX Runtime** — opset-17 export, CUDA EP when present, IO binding to torch CUDA tensors so ORT does not add extra copies.
4. **TensorRT 10 FP32** — `build_serialized_network`, named I/O tensors, `execute_async_v3`.
5. **TensorRT 10 FP16** — `BuilderFlag.FP16` on the same ONNX.
6. **Batching sweep** — 1 / 8 / 16 / 32. Batch 1 on ResNet-50 @ 224 is often launch-overhead or bandwidth bound; larger batches amortize launch and shift toward compute.
7. **CUDA Graphs** — capture once, replay in the timed loop, at batch 1 and 8 (static shapes only).

`--suite full` expands that matrix. Default CLI is all available backends, fp32+fp16, batch 1, no graph.

## How to run

CPU dummy (CI and laptops; wall-clock timer, TinyConv, no NVIDIA GPU):

```bash
uv sync --extra dev --extra torch --extra onnx
uv run gpu-bench --backends pytorch --precision fp32 --batch 1 --model tiny --warmup 2 --iters 5
uv run pytest -q
```

GPU, with uv (CUDA PyTorch / ORT GPU / TensorRT as installed on the machine):

```bash
uv sync --extra torch --extra onnx --extra cuda
# TensorRT is already present on NGC images; otherwise: uv sync --extra tensorrt
uv run gpu-bench --suite full --out report.json --md report.md
```

GPU, Docker (`--gpus all`):

```bash
docker build -t gpu-inference-benchmark:latest .
./scripts/run_gpu.sh --suite full
```

The default container entrypoint is `gpu-bench --suite full`.

## Results (Not yet run)

Numbers appear here only after a real NVIDIA GPU run. This table is a template, not a measurement.

| Backend | Precision | Batch | Graphs | mean ms | p99 ms | img/s | GPU mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — |

`gpu-bench` writes the same columns to `--md` (default `report.md`) using **measured** values or `—` if a job was skipped. It never fills TBD with estimated milliseconds.

## CUDA Graphs

Capture records a static sequence of GPU work onto a graph; replay submits that graph with one launch instead of hundreds of kernel launches. This repo follows the PyTorch recipe: static input tensor, warmup on a **side stream** (so cuDNN autotune is done), `torch.cuda.graph` capture, then `graph.replay()` in the timed loop. TensorRT 10 captures around `execute_async_v3` the same way.

Graphs require **static shapes**. Dynamic batch or changing addresses abort capture; the backend records a skip note and falls back to eager. Graphs also do not help when the GPU is already saturated (large batch, long kernels) or when the timed region is dominated by H2D/D2H — capture is compute-only, copies stay outside the graph unless you opt into `--include-transfer` around replay.

## Honest bottlenecks

- **ResNet-50 224 at batch 1** is often bandwidth- or launch-overhead-sensitive. A "faster runtime" that only cuts launch overhead will look huge at batch 1 and modest at batch 32. Read both.
- **Warmup / cuDNN autotune**: the first iterations choose algorithms (`cudnn.benchmark = True` on CUDA only). They are excluded from percentiles.
- **H2D vs kernel-only**: the default timed region is GPU compute after the input is already on device. `--include-transfer` times H2D + compute + D2H. Mixing the two is how fake speedups happen.
- **Pinned memory**: `--pinned` pins the host tensor so H2D can be asynchronous. It does nothing for compute-only timing.

## TensorRT 10 notes

This backend uses the TRT **10** Python API, not TRT 8:

| TRT 10 | TRT 8 (not used) |
| --- | --- |
| `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` | implicit batch still common in old samples |
| `builder.build_serialized_network(network, config)` | `builder.build_engine(...)` |
| named I/O: `engine.num_io_tensors`, `get_tensor_name`, `get_tensor_mode`, `set_tensor_address` | `bindings[]` + binding indices |
| `context.execute_async_v3(stream_handle)` | `execute_async_v2(bindings, stream)` |
| `BuilderFlag.FP16` | same flag name, different builder config object |
| `config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, n)` | `config.max_workspace_size = n` |

Engines are cached under `artifacts/*.engine` (gitignored). If `import tensorrt` fails, `available()` is False and the runner skips with a reason — it does not crash.

## Timing

On CUDA, every run uses **CUDA events** (`torch.cuda.Event(enable_timing=True)`, record / synchronize, `elapsed_time` in ms). Wall-clock (`time.perf_counter`) is used only when CUDA is absent, and every `RunResult` records `timing_backend` plus a note that wall-clock is **not** a CUDA event. `--require-cuda-events` refuses the wall-clock fallback.

GPU memory is `torch.cuda.max_memory_allocated()` after the run, or `None` without CUDA. It is never invented.

## Nsight Systems

Optional, if `nsys` is on `PATH` (NGC images typically include it):

```bash
./scripts/nsys_profile.sh
```

The script exits 1 with a message when `nsys` is missing; otherwise it profiles `gpu-bench --backends pytorch --precision fp16 --batch 8 --graph` to `artifacts/nsys`.

## CI

GitHub Actions is **CPU tests only** (`ubuntu-latest`, `uv sync --extra dev --extra torch --extra onnx`, `uv run pytest -q`). No `nvidia-smi`, no GPU runners. TinyConv + wall-clock cover metrics, the CPU PyTorch path, and skip reasons for TensorRT / graphs / FP16.
