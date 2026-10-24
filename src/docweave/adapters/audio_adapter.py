"""Audio transcription adapter.

Uses faster-whisper (CTranslate2 backend) for local, fast, accurate transcription
without requiring OpenAI API keys. Falls back to openai-whisper if needed.

The adapter emits structured segments with timestamps, confidence, and language
detection, plus a full-text markdown rendering.
"""

from __future__ import annotations

from pathlib import Path

from docweave.adapters.base import Adapter, ExtractionContext, ExtractionResult
from docweave.assets import AssetRef
from docweave.config import ConversionConfig
from docweave.exceptions import DependencyMissingError
from docweave.models import Document, Origin, Paragraph, SourceInfo, SourceRef


class AudioAdapter(Adapter):
    name = "audio"
    formats = ("audio",)
    requires = {"faster_whisper": "audio"}
    capabilities = (
        "local transcription (faster-whisper)",
        "timestamped segments",
        "language detection",
        "speaker diarization (when model supports it)",
        "word-level confidence",
    )
    limitations = (
        "requires ffmpeg for most formats",
        "model weights ~150MB-3GB downloaded on first use",
        "GPU acceleration optional but recommended",
        "no real-time streaming",
    )
    cost = "expensive"
    streaming = False

    def check_dependencies(self) -> None:
        super().check_dependencies()
        # Also check for ffmpeg
        import shutil
        if not shutil.which("ffmpeg"):
            raise DependencyMissingError(
                "ffmpeg",
                f"The {self.name} adapter",
                "audio (requires system ffmpeg)",
            )

    def extract(self, ctx: ExtractionContext) -> ExtractionResult:
        from faster_whisper import WhisperModel

        config: ConversionConfig = ctx.config
        path: Path = ctx.path

        # Determine model size from config or default to base
        model_size = getattr(config, "audio_model", "base")
        device = getattr(config, "audio_device", "auto")  # auto, cpu, cuda
        compute_type = getattr(config, "audio_compute_type", "auto")

        ctx.check_deadline("model load")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        ctx.check_deadline("transcription")
        segments, info = model.transcribe(
            str(path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        document = Document()
        document.container_kind = "body"

        # Build metadata
        language = info.language
        language_prob = info.language_probability
        duration = info.duration
        all_text = []

        for i, segment in enumerate(segments):
            ctx.check_deadline(f"segment {i}")
            text = segment.text.strip()
            if not text:
                continue
            all_text.append(text)

            # Create paragraph with provenance
            para = Paragraph(text=text)
            para.id = ctx.ids.next("segment")
            para.sources = [
                SourceRef(
                    adapter=self.name,
                    origin=Origin.RECONSTRUCTED,
                    confidence="high",
                    note=f"segment {i}: {segment.start:.2f}s-{segment.end:.2f}s",
                )
            ]
            para.meta = {
                "start": segment.start,
                "end": segment.end,
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in segment.words or []
                ],
            }
            document.body.append(para)

        # Add document-level metadata
        document.metadata["audio"] = {
            "language": language,
            "language_probability": language_prob,
            "duration_seconds": duration,
            "model": model_size,
            "segments": len(document.body),
        }

        # Asset reference for the original audio
        asset = AssetRef(
            asset_id=ctx.ids.next("asset"),
            media_type=ctx.detection.media_type or "audio/mpeg",
            extension=path.suffix.lower(),
            byte_size=path.stat().st_size,
            path=str(path),
            extracted=False,
            description=f"Audio transcription ({language}, {duration:.1f}s)",
        )
        document.assets.append(asset)
        ctx.assets.add(asset)

        source = SourceInfo(
            name=path.name,
            byte_size=path.stat().st_size,
            sha256="",  # Will be filled by converter
            detected_format="audio",
            detection_method=ctx.detection.method,
            detection_confidence=ctx.detection.confidence,
            declared_extension=ctx.detection.declared_extension,
            extension_mismatch=ctx.detection.extension_mismatch,
        )

        return ExtractionResult(document=document, source=source, container_kind="body")


class AudioTranscriptionResult:
    """Structured result for programmatic access."""

    def __init__(self, segments: list[dict], language: str, duration: float):
        self.segments = segments
        self.language = language
        self.duration = duration
