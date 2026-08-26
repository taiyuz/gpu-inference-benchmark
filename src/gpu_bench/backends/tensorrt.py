"""TensorRT 10 backend: named I/O tensors and execute_async_v3.

TRT 8 used binding indices + execute_async_v2. TRT 10 dropped that path in
favor of named tensors and execute_async_v3. This file is written against 10.x.
"""

from __future__ import annotations

from pathlib import Path

from gpu_bench.config import BenchConfig
from gpu_bench.export import export_onnx
from gpu_bench.metrics import RunResult, reset_gpu_peak, skipped_result, summarize
from gpu_bench.timing import make_timer


class TensorRTBackend:
    name = "tensorrt"

    def available(self) -> tuple[bool, str]:
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as exc:
            return False, f"tensorrt not installed ({exc})"
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "TensorRT backend requires CUDA"
        except ImportError as exc:
            return False, f"torch required for TRT buffers/timing ({exc})"
        return True, ""

    def run(self, cfg: BenchConfig) -> RunResult:
        ok, reason = self.available()
        if not ok:
            return skipped_result(
                backend=self.name,
                precision=cfg.precision,
                batch_size=cfg.batch_size,
                graph=cfg.graph,
                reason=reason,
            )

        import tensorrt as trt
        import torch

        engine_path = _engine_path(cfg)
        engine = _load_or_build_engine(cfg, engine_path)
        context = engine.create_execution_context()
        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        reset_gpu_peak()

        tensors = _allocate_io(engine, context, cfg)
        # Non-default stream: CUDA Graph capture cannot use the default stream.
        stream = torch.cuda.Stream() if cfg.use_nondefault_stream else torch.cuda.current_stream()
        stream_handle = int(stream.cuda_stream)
        notes = [
            f"trt={getattr(trt, '__version__', 'unknown')}",
            f"engine={engine_path.name}",
            "execute_async_v3 + named I/O tensors (TRT 10; implicit batch removed)",
            "non-default CUDA stream" if cfg.use_nondefault_stream else "default stream",
        ]

        def enqueue() -> None:
            ok_exec = context.execute_async_v3(stream_handle)
            if not ok_exec:
                raise RuntimeError("TensorRT execute_async_v3 failed")

        graph = None
        if cfg.graph:
            for _ in range(3):
                enqueue()
            stream.synchronize()
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    enqueue()
                notes.append("CUDA Graph captured around execute_async_v3")
            except Exception as exc:  # capture can fail on some TRT/runtime combos
                graph = None
                notes.append(f"CUDA Graph capture failed ({type(exc).__name__}: {exc}); using enqueue")

        def step() -> None:
            if graph is not None:
                graph.replay()
            else:
                enqueue()

        for _ in range(cfg.warmup):
            step()
        stream.synchronize()

        latencies = [timer.measure(step) for _ in range(cfg.iters)]
        _ = tensors
        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=cfg.graph and graph is not None,
            n_warmup=cfg.warmup,
            latencies_ms=latencies,
            timing_backend=timer.backend.name,
            notes="; ".join(notes + [timer.backend.notes]),
            extra={"engine": str(engine_path)},
        )


def _engine_path(cfg: BenchConfig) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return cfg.artifacts_dir / f"{cfg.model}_{cfg.precision}_b{cfg.batch_size}.engine"


def _load_or_build_engine(cfg: BenchConfig, path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if path.exists():
        return runtime.deserialize_cuda_engine(path.read_bytes())

    onnx_file = export_onnx(cfg)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = onnx_file.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_bytes)
    if cfg.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this GPU does not report fast FP16; refusing TRT FP16 build")
        config.set_flag(trt.BuilderFlag.FP16)
    elif cfg.precision == "bf16":
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise RuntimeError("this TensorRT build has no BuilderFlag.BF16")
        config.set_flag(trt.BuilderFlag.BF16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    path.write_bytes(bytes(serialized))
    return runtime.deserialize_cuda_engine(serialized)


def _allocate_io(engine, context, cfg: BenchConfig) -> dict[str, object]:
    import tensorrt as trt
    import torch

    dtype_map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "bfloat16"):
        dtype_map[trt.bfloat16] = torch.bfloat16
    elif hasattr(trt, "bf16"):
        dtype_map[trt.bf16] = torch.bfloat16
    tensors: dict[str, object] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        fixed = []
        for dim_i, dim in enumerate(shape):
            if dim == -1:
                fixed.append(cfg.batch_size if dim_i == 0 else cfg.input_size)
            else:
                fixed.append(int(dim))
        shape = tuple(fixed)
        if -1 in tuple(engine.get_tensor_shape(name)):
            context.set_input_shape(name, shape)
        trt_dtype = engine.get_tensor_dtype(name)
        torch_dtype = dtype_map.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(f"unsupported TRT dtype {trt_dtype} for {name}")
        buf = torch.empty(shape, device="cuda", dtype=torch_dtype)
        if mode == trt.TensorIOMode.INPUT:
            buf.normal_()
        context.set_tensor_address(name, int(buf.data_ptr()))
        tensors[name] = buf
    return tensors
