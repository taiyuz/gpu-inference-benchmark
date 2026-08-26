"""Inference backend registry. Backends register in a follow-up commit."""

from __future__ import annotations

BACKENDS: dict = {}

__all__ = ["BACKENDS"]
