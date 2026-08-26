"""Job expansion, skip-with-reason, and suite runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from gpu_bench.backends import REGISTRY, get_backend
from gpu_bench.config import RunConfig
from gpu_bench.metrics import RunResult

DEFAULT_BACKENDS = ("pytorch", "onnx", "tensorrt")
DEFAULT_PRECISIONS = ("fp32", "fp16")
FULL_BATCHES = (1, 8, 16, 32)
FULL_GRAPH_BATCHES = (1, 8)


@dataclass
class Skip:
    backend: str
    precision: str
    batch_size: int
    graph: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "graph": self.graph,
            "reason": self.reason,
        }


@dataclass
class SuiteResult:
    results: list[RunResult]
    skips: list[Skip]


def parse_csv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return [p.strip() for p in parts if p.strip()]


def expand_jobs(
    *,
    backends: list[str] | None,
    precisions: list[str] | None,
    batches: list[int] | None,
    graph: bool,
    suite: str | None,
    warmup: int,
    iters: int,
    include_transfer: bool,
    pinned: bool,
    model: str,
    require_cuda_events: bool,
    artifacts_dir: Path,
) -> list[RunConfig]:
    """Build the list of RunConfig jobs.

    Default: all requested/available backends, fp32+fp16, batch 1, no graph.
    --suite full: pytorch/onnx/tensorrt, fp32/fp16, batches 1/8/16/32, and
    CUDA Graphs at batch 1 and 8 for pytorch and tensorrt (in addition to eager).
    """
    if suite == "full":
        names = list(DEFAULT_BACKENDS)
        precs = list(DEFAULT_PRECISIONS)
        batch_list = list(FULL_BATCHES)
        jobs: list[RunConfig] = []
        for name in names:
            for prec in precs:
                for batch in batch_list:
                    jobs.append(
                        _cfg(
                            name, prec, batch, False, warmup, iters,
                            include_transfer, pinned, model,
                            require_cuda_events, artifacts_dir,
                        )
                    )
                for batch in FULL_GRAPH_BATCHES:
                    if name in {"pytorch", "tensorrt"}:
                        jobs.append(
                            _cfg(
                                name, prec, batch, True, warmup, iters,
                                include_transfer, pinned, model,
                                require_cuda_events, artifacts_dir,
                            )
                        )
        return jobs

    names = backends or list(DEFAULT_BACKENDS)
    precs = precisions or list(DEFAULT_PRECISIONS)
    batch_list = batches or [1]
    jobs = []
    for name in names:
        for prec in precs:
            for batch in batch_list:
                jobs.append(
                    _cfg(
                        name, prec, batch, graph, warmup, iters,
                        include_transfer, pinned, model,
                        require_cuda_events, artifacts_dir,
                    )
                )
    return jobs


def _cfg(
    backend: str,
    precision: str,
    batch_size: int,
    graph: bool,
    warmup: int,
    iters: int,
    include_transfer: bool,
    pinned: bool,
    model: str,
    require_cuda_events: bool,
    artifacts_dir: Path,
) -> RunConfig:
    return RunConfig(
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        n_warmup=warmup,
        n_iter=iters,
        graph=graph,
        include_transfer=include_transfer,
        pinned=pinned,
        model=model,
        require_cuda_events=require_cuda_events,
        artifacts_dir=artifacts_dir,
    )


def run_suite(jobs: list[RunConfig]) -> SuiteResult:
    results: list[RunResult] = []
    skips: list[Skip] = []
    for cfg in jobs:
        backend = get_backend(cfg.backend)
        if backend is None:
            skips.append(
                Skip(cfg.backend, cfg.precision, cfg.batch_size, cfg.graph,
                     f"unknown backend {cfg.backend!r}; known: {sorted(REGISTRY)}")
            )
            continue
        if not backend.available():
            skips.append(
                Skip(cfg.backend, cfg.precision, cfg.batch_size, cfg.graph,
                     backend.unavailable_reason() or f"{cfg.backend} unavailable")
            )
            continue
        try:
            results.append(backend.run(cfg))
        except Exception as exc:  # noqa: BLE001
            skips.append(
                Skip(cfg.backend, cfg.precision, cfg.batch_size, cfg.graph, str(exc))
            )
    return SuiteResult(results=results, skips=skips)
