"""ONNX Runtime backend. CUDA EP + IO binding when a GPU is visible."""

from __future__ import annotations

from gpu_bench.config import BenchConfig
from gpu_bench.export import export_onnx
from gpu_bench.metrics import RunResult, reset_gpu_peak, skipped_result, summarize
from gpu_bench.timing import cuda_is_available, make_timer


class OnnxRuntimeBackend:
    name = "onnx"

    def available(self) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            return False, f"onnxruntime not installed ({exc})"
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"torch required to export ONNX ({exc})"
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
        if cfg.graph:
            return skipped_result(
                backend=self.name,
                precision=cfg.precision,
                batch_size=cfg.batch_size,
                graph=cfg.graph,
                reason="CUDA Graphs are not wired through ONNX Runtime in this harness; use pytorch/tensorrt",
            )
        if cfg.precision == "bf16":
            return skipped_result(
                backend=self.name,
                precision=cfg.precision,
                batch_size=cfg.batch_size,
                graph=cfg.graph,
                reason="ORT path here is fp32/fp16 via CUDA EP; bf16 is pytorch/tensorrt only",
            )

        import numpy as np
        import onnxruntime as ort
        import torch

        onnx_file = export_onnx(
            model_name=cfg.model,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            artifacts_dir=cfg.artifacts_dir,
        )
        available_eps = set(ort.get_available_providers())
        if cuda_is_available() and "CUDAExecutionProvider" in available_eps:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ep = "CUDAExecutionProvider"
        else:
            providers = ["CPUExecutionProvider"]
            ep = "CPUExecutionProvider"
            if cfg.precision == "fp16":
                return skipped_result(
                    backend=self.name,
                    precision=cfg.precision,
                    batch_size=cfg.batch_size,
                    graph=False,
                    reason="ORT fp16 path in this harness expects CUDA EP",
                )

        sess = ort.InferenceSession(str(onnx_file), providers=providers)
        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        reset_gpu_peak()

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        dtype = torch.float16 if cfg.precision == "fp16" and cuda_is_available() else torch.float32
        device = torch.device("cuda" if cuda_is_available() and ep == "CUDAExecutionProvider" else "cpu")

        x = torch.randn(
            cfg.batch_size,
            3,
            cfg.input_size,
            cfg.input_size,
            device=device,
            dtype=dtype,
        )
        notes = [f"ep={ep}", f"onnx={onnx_file.name}"]
        use_io = device.type == "cuda"

        if use_io:
            io = sess.io_binding()
            io.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=0,
                element_type=np.float16 if dtype == torch.float16 else np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            io.bind_output(output_name, "cuda")
            notes.append("IO binding on CUDA tensors (avoids ORT-owned copies)")

            def step() -> None:
                sess.run_with_iobinding(io)

        else:
            feed = {input_name: x.cpu().numpy()}

            def step() -> None:
                sess.run([output_name], feed)

        for _ in range(cfg.warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = [timer.measure(step) for _ in range(cfg.iters)]
        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=False,
            n_warmup=cfg.warmup,
            latencies_ms=latencies,
            timing_backend=timer.backend.name,
            notes="; ".join(notes + [timer.backend.notes]),
        )
