"""PyTorch eager and CUDA Graphs inference backend.

Default timed region is GPU compute after the input is already on device.
Pass include_transfer=True to time H2D + compute + D2H. Optional pinned host
memory is used for H2D when cfg.pinned is set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpu_bench.config import RunConfig
from gpu_bench.metrics import RunResult, summarize
from gpu_bench.models import build_model
from gpu_bench.timing import Timer, make_timer


class PyTorchBackend:
    name = "pytorch"

    def available(self) -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def unavailable_reason(self) -> str:
        if self.available():
            return ""
        return "torch is not installed (pip/uv extra: torch)"

    def run(self, cfg: RunConfig) -> RunResult:
        if not self.available():
            raise RuntimeError(self.unavailable_reason())
        import torch

        if cfg.precision == "fp16" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch FP16 is CUDA-only; CPU FP16 is skipped")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.backends.cudnn.benchmark = False

        model = build_model(cfg.model, device, cfg.precision, pretrained=False)
        model.eval()

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        cpu_x = torch.randn(cfg.batch_size, 3, 224, 224, dtype=dtype)
        if cfg.pinned and device.type == "cuda":
            cpu_x = cpu_x.pin_memory()

        timer = make_timer(require_cuda_events=cfg.require_cuda_events)
        notes = [timer.notes]
        use_graph = bool(cfg.graph)
        if use_graph and device.type != "cuda":
            notes.append("CUDA Graphs requested but CUDA is unavailable; running eager.")
            use_graph = False

        if cfg.include_transfer:
            notes.append("Timed region includes H2D + compute + D2H.")
        else:
            notes.append("Timed region is GPU compute only; input already on device.")
        if cfg.pinned and device.type == "cuda":
            notes.append("Host tensor is pinned for H2D.")

        if use_graph:
            latencies, graph_ok, extra = _run_cuda_graph(
                model, cpu_x, device, cfg, timer
            )
            notes.extend(extra)
            use_graph = graph_ok
        else:
            latencies = _run_eager(model, cpu_x, device, cfg, timer)

        mem = None
        if device.type == "cuda":
            mem = int(torch.cuda.max_memory_allocated())

        return summarize(
            backend=self.name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=use_graph,
            n_warmup=cfg.n_warmup,
            n_iter=cfg.n_iter,
            latencies_ms=latencies,
            timing_backend=timer.name,
            notes=" ".join(notes),
            gpu_mem=mem,
        )


def _forward(model: Any, x: Any) -> Any:
    import torch

    with torch.inference_mode():
        return model(x)


def _run_eager(
    model: Any,
    cpu_x: Any,
    device: Any,
    cfg: RunConfig,
    timer: Timer,
) -> list[float]:
    import torch

    non_blocking = bool(cfg.pinned and device.type == "cuda")
    if not cfg.include_transfer:
        x = cpu_x.to(device, non_blocking=non_blocking)
        if device.type == "cuda":
            torch.cuda.synchronize()

        def step() -> None:
            _forward(model, x)

    else:

        def step() -> None:
            x = cpu_x.to(device, non_blocking=non_blocking)
            y = _forward(model, x)
            if device.type == "cuda":
                _ = y.to("cpu")

    return _timed_loop(step, cfg, timer, device)


def _run_cuda_graph(
    model: Any,
    cpu_x: Any,
    device: Any,
    cfg: RunConfig,
    timer: Timer,
) -> tuple[list[float], bool, list[str]]:
    """Static-shape CUDA Graph: warmup on a side stream, capture, replay.

    Capture is compute-only (input already on device). include_transfer times
    a host copy into the static buffer plus replay plus D2H around the graph.
    """
    import torch

    notes: list[str] = []
    static_input = cpu_x.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup on a side stream so cuDNN autotune finishes before capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    static_output = None
    with torch.cuda.stream(side):
        for _ in range(max(cfg.n_warmup, 3)):
            static_output = _forward(model, static_input)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            static_output = _forward(model, static_input)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CUDA Graph capture failed; falling back to eager ({exc}).")
        return _run_eager(model, cpu_x, device, cfg, timer), False, notes

    notes.append("CUDA Graph captured (static input); timed loop is graph.replay().")
    non_blocking = bool(cfg.pinned)

    def step() -> None:
        if cfg.include_transfer:
            static_input.copy_(cpu_x, non_blocking=non_blocking)
        graph.replay()
        if cfg.include_transfer and static_output is not None:
            _ = static_output.to("cpu")

    latencies = _timed_loop(step, cfg, timer, device)
    return latencies, True, notes


def _timed_loop(
    step: Callable[[], None],
    cfg: RunConfig,
    timer: Timer,
    device: Any,
) -> list[float]:
    import torch

    for _ in range(cfg.n_warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(cfg.n_iter):
        timer.start()
        step()
        latencies.append(timer.stop())
    return latencies
