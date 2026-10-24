"""Deterministic identifier generation.

Ids encode the structural path of a node (``p3.h2.t1.r0c2``) so they are stable
across runs for the same input and configuration, which makes diffing and
incremental re-processing possible. No randomness, no counters that survive a
reset, no clock.
"""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 32) -> str:
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return s[:limit] or "untitled"


class IdFactory:
    """Builds hierarchical ids under a fixed document prefix."""

    __slots__ = ("prefix", "_counters")

    def __init__(self, prefix: str = "d") -> None:
        self.prefix = prefix
        self._counters: dict[str, int] = {}

    def child(self, parent: str | None, tag: str) -> str:
        """Return a fresh id ``parent.tagN`` (or ``prefix.tagN`` at the root)."""
        base = parent or self.prefix
        key = f"{base}.{tag}"
        n = self._counters.get(key, 0)
        self._counters[key] = n + 1
        return f"{key}{n}"

    def named(self, parent: str | None, tag: str, *parts: str | int) -> str:
        """Return an id derived from content parts rather than a counter."""
        base = parent or self.prefix
        tail = "-".join(slug(str(p)) for p in parts if p not in (None, ""))
        return f"{base}.{tag}.{tail}" if tail else self.child(parent, tag)

    def reset(self) -> None:
        self._counters.clear()
