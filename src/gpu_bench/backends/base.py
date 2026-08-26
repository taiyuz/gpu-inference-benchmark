"""Backend protocol and skip helpers."""

from __future__ import annotations

from typing import Protocol

from gpu_bench.config import BenchConfig
from gpu_bench.metrics import RunResult, skipped_result


class Backend(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def run(self, cfg: BenchConfig) -> RunResult: ...


def skip_if_unavailable(backend: Backend, cfg: BenchConfig) -> RunResult | None:
    ok, reason = backend.available()
    if ok:
        return None
    return skipped_result(
        backend=backend.name,
        precision=cfg.precision,
        batch_size=cfg.batch_size,
        graph=cfg.graph,
        reason=reason,
    )
