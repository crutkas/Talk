"""Base class for STT engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.deps import ensure_packages


class STTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    # Subclasses override: {import_name: pip_specifier}
    REQUIRED_PACKAGES: dict[str, str] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text."""
        ...

    def is_available(self) -> bool:
        """Check if this engine is ready to use."""
        return True

    def needs_download(self) -> bool:
        """Check if this engine needs to download a model before first use."""
        return False

    def ensure_ready(self, progress_callback: Any | None = None) -> bool:
        """Install dependencies and download model if needed. Returns True if ready."""
        if self.REQUIRED_PACKAGES and not ensure_packages(
            self.REQUIRED_PACKAGES, progress_callback
        ):
            return False

        if self.needs_download():
            self.download_model(progress_callback)

        return True

    def download_model(self, progress_callback: Any | None = None) -> None:  # noqa: B027
        """Download the model if needed."""

    def cleanup(self) -> None:  # noqa: B027
        """Release resources held by the engine."""
