"""Asset collection.

Images are *referenced* by default and only written to disk when explicitly
requested, because reading and hashing every embedded image is a real cost on
large documents. Extraction is deterministic: the filename is derived from the
asset id and content hash, never from an incrementing global counter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from docweave.exceptions import LimitExceededError
from docweave.models import AssetRef, Locator

#: Conservative extension map from media type. Unknown types keep no extension.
EXTENSION_BY_MEDIA = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
    "image/webp": ".webp",
    "image/x-emf": ".emf",
    "image/x-wmf": ".wmf",
    "image/svg+xml": ".svg",
}


class AssetCollector:
    """Accumulates asset references and optionally writes bytes to disk."""

    def __init__(
        self,
        directory: Path | None = None,
        enabled: bool = False,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.enabled = bool(enabled and self.directory is not None)
        self.max_bytes = max_bytes
        self._refs: dict[str, AssetRef] = {}
        self._written = 0

    @property
    def referenced(self) -> bool:
        return self.enabled

    def add(
        self,
        data: bytes | None,
        *,
        media_type: str | None,
        locator: Locator | None = None,
        extension: str | None = None,
        description: str | None = None,
        asset_id: str,
    ) -> AssetRef:
        """Register an asset. ``data`` may be ``None`` to record a reference only."""
        ext = extension or EXTENSION_BY_MEDIA.get(media_type or "")
        if data is not None:
            digest = hashlib.sha256(data).hexdigest()
            size = len(data)
            path = None
            if self.enabled and size:
                self._written += size
                if self._written > self.max_bytes:
                    raise LimitExceededError(
                        f"Asset extraction exceeded the configured budget of {self.max_bytes} bytes. "
                        "Re-run with --no-extract-assets or raise max_asset_bytes."
                    )
                path = self._write(asset_id, ext, data, digest)
            ref = AssetRef(
                asset_id=asset_id,
                media_type=media_type,
                extension=ext or None,
                byte_size=size,
                sha256=digest,
                path=str(path) if path else None,
                extracted=path is not None,
                locator=locator or Locator(),
                description=description,
            )
        else:
            ref = AssetRef(
                asset_id=asset_id,
                media_type=media_type,
                extension=ext or None,
                locator=locator or Locator(),
                description=description,
            )
        self._refs[asset_id] = ref
        return ref

    def _write(self, asset_id: str, ext: str, data: bytes, digest: str) -> Path:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"{_safe(asset_id)}{ext}"
        target = self.directory / name
        # Content-addressed collision handling keeps output deterministic.
        if target.exists() and target.read_bytes() != data:
            target = self.directory / f"{_safe(asset_id)}-{digest[:8]}{ext}"
        target.write_bytes(data)
        return target

    def refs(self) -> list[AssetRef]:
        return [self._refs[k] for k in sorted(self._refs)]

    def get(self, asset_id: str) -> AssetRef | None:
        return self._refs.get(asset_id)


def _safe(value: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in value)
    return out.strip("-.") or "asset"
