"""ONNX Runtime backend with CUDA IO binding when a CUDA EP is active."""

from __future__ import annotations

from typing import Any

from gpu_bench.config import RunConfig
from gpu_bench.export import export_onnx
from gpu_bench.metrics import RunResult, summarize
from gpu_bench.models import output_features
from gpu_bench.timing import make_timer


class OnnxRuntimeBackend:
    name = "onnx"

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        if self.available():
            return ""
        return "onnxruntime is not installed (pip/uv extra: onnx or cuda)"

    def run(self, cfg: RunConfig) -> RunResult:
        if not self.available():
            raise RuntimeError(self.unavailable_reason())
        import numpy as np
        import onnxruntime as ort

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("ONNX export requires torch") from exc

        if cfg.precision == "fp16" and not torch.cuda.is_available():
            raise RuntimeError("FP16 ONNX export uses .half() and requires CUDA")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        onnx_file = export_onnx(
            model_name=cfg.model,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            artifacts_dir=cfg.artifacts_dir,
            device=device,
        )

        providers = _session_providers()
        session = ort.InferenceSession(str(onnx_file), providers=providers)
        use_cuda = "CUDAExecutionProvider" in session.get_providers()
        np_dtype = np.float16 if cfg.precision == "fp16" else np.float32
        torch_dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        n_out = output_features(cfg.model)

        cpu_x = torch.randn(cfg.batch_size, 3, 224, 224, dtype=torch_dtype)
        if cfg.pinned and use_cuda:
            cpu_x = cpu_x.pin_memory()

        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        notes = [timer.notes, f"ONNX cached at {onnx_file}."]
        if cfg.graph:
            notes.append("CUDA Graphs are not used for the ONNX Runtime path.")

        if use_cuda:
            notes.append("CUDAExecutionProvider.")
            latencies, extra = _run_cuda_io_binding(
                session, cpu_x, torch_dtype, np_dtype, n_out, cfg, timer
            )
            notes.extend(extra)
        else:
            notes.append("CPUExecutionProvider (no IO binding).")
            latencies = _run_cpu(session, cpu_x, cfg, timer)

        mem = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=False,
            n_warmup=cfg.n_warmup,
            n_iter=cfg.n_iter,
            latencies_ms=latencies,
            timing_backend=timer.name,
            notes=" ".join(notes),
            gpu_mem=mem,
        )


def _session_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _run_cpu(session: Any, cpu_x: Any, cfg: RunConfig, timer: Any) -> list[float]:
    feed = {"input": cpu_x.numpy()}
    for _ in range(cfg.n_warmup):
        session.run(["output"], feed)
    latencies: list[float] = []
    for _ in range(cfg.n_iter):
        timer.start()
        session.run(["output"], feed)
        latencies.append(timer.stop())
    return latencies


def _run_cuda_io_binding(
    session: Any,
    cpu_x: Any,
    torch_dtype: Any,
    np_dtype: Any,
    n_out: int,
    cfg: RunConfig,
    timer: Any,
) -> tuple[list[float], list[str]]:
    import torch

    notes: list[str] = []
    x = cpu_x.to("cuda")
    y = torch.empty((cfg.batch_size, n_out), dtype=torch_dtype, device="cuda")
    torch.cuda.synchronize()

    io_binding = session.io_binding()
    io_binding.bind_input(
        "input",
        device_type="cuda",
        device_id=0,
        element_type=np_dtype,
        shape=tuple(x.shape),
        buffer_ptr=x.data_ptr(),
    )
    io_binding.bind_output(
        "output",
        device_type="cuda",
        device_id=0,
        element_type=np_dtype,
        shape=tuple(y.shape),
        buffer_ptr=y.data_ptr(),
    )
    notes.append("IO binding uses torch CUDA tensors (no extra ORT copies).")
    if cfg.include_transfer:
        notes.append("Timed region includes H2D + run_with_iobinding + D2H.")
    else:
        notes.append("Timed region is run_with_iobinding; input already on device.")

    def step() -> None:
        if cfg.include_transfer:
            x.copy_(cpu_x, non_blocking=bool(cfg.pinned))
        session.run_with_iobinding(io_binding)
        if cfg.include_transfer:
            _ = y.to("cpu")

    for _ in range(cfg.n_warmup):
        step()
    torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(cfg.n_iter):
        timer.start()
        step()
        latencies.append(timer.stop())
    return latencies, notes
