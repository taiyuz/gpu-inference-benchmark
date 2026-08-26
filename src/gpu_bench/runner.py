"""Run one backend or the recruiting comparison suite."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from gpu_bench.backends import BACKENDS
from gpu_bench.config import BenchConfig
from gpu_bench.metrics import RunResult, skipped_result

FULL_BACKENDS = ("pytorch", "onnx", "tensorrt")
FULL_PRECISIONS = ("fp32", "fp16", "bf16")
FULL_BATCHES = (1, 8, 16, 32)
GRAPH_BATCHES = (1, 8)


def available_backends() -> dict[str, tuple[bool, str]]:
    return {name: backend.available() for name, backend in BACKENDS.items()}


def run_one(backend_name: str, cfg: BenchConfig) -> RunResult:
    backend = BACKENDS.get(backend_name)
    if backend is None:
        return skipped_result(
            backend=backend_name,
            precision=cfg.precision,
            batch_size=cfg.batch_size,
            graph=cfg.graph,
            reason=f"unknown backend {backend_name}",
        )
    return backend.run(cfg)


def run_suite(
    *,
    backends: Sequence[str] | None = None,
    precisions: Sequence[str] | None = None,
    batches: Sequence[int] | None = None,
    graphs: bool = False,
    suite: str = "default",
    base: BenchConfig,
) -> list[RunResult]:
    if suite == "full":
        jobs = _full_jobs()
    else:
        names = tuple(backends or ("pytorch",))
        precs = tuple(precisions or ("fp32",))
        batch_list = tuple(batches or (base.batch_size,))
        jobs = [
            (name, precision, batch, graphs)
            for name in names
            for precision in precs
            for batch in batch_list
        ]

    results: list[RunResult] = []
    for name, precision, batch, graph in jobs:
        cfg = BenchConfig(
            model=base.model,
            precision=precision,
            batch_size=batch,
            warmup=base.warmup,
            iters=base.iters,
            graph=graph,
            include_transfer=base.include_transfer,
            pinned=base.pinned,
            pretrained=base.pretrained,
            artifacts_dir=base.artifacts_dir,
            require_cuda_events=base.require_cuda_events,
            input_size=base.input_size,
            workspace_bytes=base.workspace_bytes,
            seed=base.seed,
            use_nondefault_stream=base.use_nondefault_stream,
        )
        results.append(run_one(name, cfg))
    return results


def expand_jobs(*, suite: str = "full") -> list[tuple[str, str, int, bool]]:
    if suite == "full":
        return _full_jobs()
    return []


def _full_jobs() -> list[tuple[str, str, int, bool]]:
    """PyTorch FP32→FP16→BF16 → ORT → TRT FP32→FP16→BF16 → batching → CUDA Graphs."""
    jobs: list[tuple[str, str, int, bool]] = []
    for name in FULL_BACKENDS:
        for precision in FULL_PRECISIONS:
            for batch in FULL_BATCHES:
                jobs.append((name, precision, batch, False))
    for name in ("pytorch", "tensorrt"):
        for precision in FULL_PRECISIONS:
            for batch in GRAPH_BATCHES:
                jobs.append((name, precision, batch, True))
    return jobs


def describe_jobs(jobs: Iterable[tuple[str, str, int, bool]]) -> list[str]:
    return [f"{n} {p} batch={b} graph={g}" for n, p, b, g in jobs]
