"""Adapters: one module per format family, all implementing :class:`Adapter`."""

from docweave.adapters.base import (
    Adapter,
    AdapterRegistry,
    ExtractionContext,
    ExtractionResult,
    build_registry,
)

__all__ = [
    "Adapter",
    "AdapterRegistry",
    "ExtractionContext",
    "ExtractionResult",
    "build_registry",
]
