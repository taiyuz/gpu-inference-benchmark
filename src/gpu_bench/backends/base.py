"""Backend protocol and in-process registry."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gpu_bench.config import RunConfig
from gpu_bench.metrics import RunResult


@runtime_checkable
class Backend(Protocol):
    name: str

    def available(self) -> bool: ...

    def unavailable_reason(self) -> str: ...

    def run(self, cfg: RunConfig) -> RunResult: ...


REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    REGISTRY[backend.name] = backend
    return backend


def get_backend(name: str) -> Backend | None:
    return REGISTRY.get(name)
