"""PyTorch eager FP32/FP16/BF16, optional CUDA Graphs, optional H2D in the timed region."""

from __future__ import annotations

from contextlib import nullcontext

from gpu_bench.config import BenchConfig
from gpu_bench.metrics import RunResult, reset_gpu_peak, skipped_result, summarize
from gpu_bench.models import load_model, random_input, resolve_device, resolve_dtype
from gpu_bench.timing import make_timer


class PyTorchBackend:
    name = "pytorch"

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            return False, f"PyTorch/torchvision not installed ({exc})"
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

        import torch

        device = resolve_device()
        if cfg.precision in {"fp16", "bf16"} and device.type != "cuda":
            return skipped_result(
                backend=self.name,
                precision=cfg.precision,
                batch_size=cfg.batch_size,
                graph=cfg.graph,
                reason=f"PyTorch {cfg.precision} requires CUDA",
            )
        if cfg.graph and device.type != "cuda":
            return skipped_result(
                backend=self.name,
                precision=cfg.precision,
                batch_size=cfg.batch_size,
                graph=cfg.graph,
                reason="CUDA Graphs require CUDA",
            )

        dtype = resolve_dtype(cfg.precision, device) if device.type == "cuda" else torch.float32
        model = load_model(cfg, device=device, dtype=dtype)
        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        reset_gpu_peak()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        notes = [f"device={device}", f"dtype={dtype}"]
        static_x = random_input(cfg, device=device, dtype=dtype)
        graph = None
        static_y = None

        if cfg.graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    _ = model(static_x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                static_y = model(static_x)
            notes.append(
                "CUDA Graph stream-capture/replay on static NCHW; "
                "torch.cuda.graph() uses a non-default side stream"
            )

        host = None
        if cfg.include_transfer:
            host = torch.randn(
                cfg.batch_size, 3, cfg.input_size, cfg.input_size, dtype=torch.float32
            )
            if cfg.pinned and device.type == "cuda":
                host = host.pin_memory()
            notes.append("timed region includes H2D into static buffer (+ D2H of output)")
        else:
            notes.append("timed region is compute-only (input already on device)")

        work_stream = None
        if device.type == "cuda" and cfg.use_nondefault_stream and graph is None:
            work_stream = torch.cuda.Stream()
            notes.append("eager enqueue on non-default CUDA stream")

        last = {"y": static_y}

        def compute() -> None:
            if graph is not None:
                graph.replay()
                last["y"] = static_y
                return
            stream_ctx = torch.cuda.stream(work_stream) if work_stream is not None else nullcontext()
            with stream_ctx, torch.inference_mode():
                last["y"] = model(static_x)

        def step() -> None:
            if cfg.include_transfer:
                static_x.copy_(
                    host.to(device=device, dtype=dtype, non_blocking=True),
                    non_blocking=True,
                )
            compute()
            if cfg.include_transfer and device.type == "cuda" and last["y"] is not None:
                _ = last["y"].cpu()

        for _ in range(cfg.warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = [timer.measure(step) for _ in range(cfg.iters)]
        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=cfg.graph,
            n_warmup=cfg.warmup,
            latencies_ms=latencies,
            timing_backend=timer.backend.name,
            notes="; ".join(notes + [timer.backend.notes]),
        )
