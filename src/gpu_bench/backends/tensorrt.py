"""TensorRT 10 Python API backend (NOT TRT 8 bindings / execute_async_v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gpu_bench.config import RunConfig
from gpu_bench.export import export_onnx
from gpu_bench.metrics import RunResult, summarize
from gpu_bench.timing import make_timer


class TensorRTBackend:
    name = "tensorrt"

    def available(self) -> bool:
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError:
            return "tensorrt is not installed (NGC image or pip/uv extra: tensorrt)"
        try:
            import torch
        except ImportError:
            return "torch is required to run the TensorRT backend"
        if not torch.cuda.is_available():
            return "TensorRT inference requires CUDA"
        return ""

    def run(self, cfg: RunConfig) -> RunResult:
        if not self.available():
            raise RuntimeError(self.unavailable_reason() or "tensorrt unavailable")
        import torch

        torch.cuda.reset_peak_memory_stats()
        onnx_file = export_onnx(
            model_name=cfg.model,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            artifacts_dir=cfg.artifacts_dir,
        )
        engine_path = _engine_path(cfg.artifacts_dir, cfg.model, cfg.precision, cfg.batch_size)
        engine = _load_or_build_engine(onnx_file, engine_path, cfg)

        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        notes = [
            timer.notes,
            f"TensorRT 10 engine at {engine_path}.",
            "I/O via named tensors + execute_async_v3 (not TRT 8 bindings[]).",
        ]
        latencies, used_graph, extra = _run_engine(engine, cfg, timer)
        notes.extend(extra)

        mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=used_graph,
            n_warmup=cfg.n_warmup,
            n_iter=cfg.n_iter,
            latencies_ms=latencies,
            timing_backend=timer.name,
            notes=" ".join(notes),
            gpu_mem=mem,
        )


def _engine_path(artifacts_dir: Path, model: str, precision: str, batch_size: int) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir / f"{model}_{precision}_b{batch_size}.engine"


def _load_or_build_engine(onnx_file: Path, engine_path: Path, cfg: RunConfig) -> Any:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if engine_path.exists():
        serialized = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is not None:
            return engine

    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_file.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_bytes)
    if cfg.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    engine_path.write_bytes(bytes(serialized))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the built engine")
    return engine


def _run_engine(engine: Any, cfg: RunConfig, timer: Any) -> tuple[list[float], bool, list[str]]:
    import tensorrt as trt
    import torch

    notes: list[str] = []
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()
    dtype = torch.float16 if cfg.precision == "fp16" else torch.float32

    buffers: dict[str, torch.Tensor] = {}
    input_name: str | None = None
    output_names: list[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(int(d) for d in engine.get_tensor_shape(name))
        if any(d < 0 for d in shape):
            if mode == trt.TensorIOMode.INPUT:
                shape = (cfg.batch_size, 3, 224, 224)
                context.set_input_shape(name, shape)
            else:
                shape = tuple(int(d) for d in context.get_tensor_shape(name))
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        buffers[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_names.append(name)

    if input_name is None:
        raise RuntimeError("TensorRT engine has no input tensor")

    cpu_x = torch.randn_like(buffers[input_name], device="cpu")
    if cfg.pinned:
        cpu_x = cpu_x.pin_memory()
    buffers[input_name].copy_(cpu_x)
    torch.cuda.synchronize()

    if cfg.include_transfer:
        notes.append("Timed region includes H2D + execute_async_v3 + D2H.")
    else:
        notes.append("Timed region is execute_async_v3; input already on device.")

    def infer() -> None:
        if cfg.include_transfer:
            buffers[input_name].copy_(cpu_x, non_blocking=bool(cfg.pinned))
        ok = context.execute_async_v3(int(stream.cuda_stream))
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        if cfg.include_transfer:
            for n in output_names:
                _ = buffers[n].to("cpu")

    for _ in range(cfg.n_warmup):
        infer()
    torch.cuda.synchronize()

    used_graph = False
    replay = infer
    if cfg.graph:
        try:
            graph = torch.cuda.CUDAGraph()
            # Capture compute-only: copies stay outside the graph.
            with torch.cuda.graph(graph, stream=stream):
                ok = context.execute_async_v3(int(stream.cuda_stream))
                if not ok:
                    raise RuntimeError("execute_async_v3 failed during capture")

            def replay_graph() -> None:
                if cfg.include_transfer:
                    buffers[input_name].copy_(cpu_x, non_blocking=bool(cfg.pinned))
                graph.replay()
                if cfg.include_transfer:
                    for n in output_names:
                        _ = buffers[n].to("cpu")

            replay = replay_graph
            used_graph = True
            notes.append("CUDA Graph captured around execute_async_v3.")
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"CUDA Graph capture skipped (dynamic shapes or TRT capture limits): {exc}."
            )
            replay = infer

    latencies: list[float] = []
    for _ in range(cfg.n_iter):
        timer.start()
        replay()
        latencies.append(timer.stop())
    return latencies, used_graph, notes
