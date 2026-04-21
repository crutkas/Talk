"""Base class for STT engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class STTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text.

        Args:
            audio_bytes: WAV-formatted audio data.

        Returns:
            Transcribed text string.
        """
        ...

    def is_available(self) -> bool:
        """Check if this engine is ready to use."""
        return True

    def needs_download(self) -> bool:
        """Check if this engine needs to download a model before first use."""
        return False

    def download_model(self, progress_callback: Any | None = None) -> None:  # noqa: B027
        """Download the model if needed. Override in subclasses.

        Args:
            progress_callback: Optional callable(status_text: str) for progress updates.
        """

    def cleanup(self) -> None:  # noqa: B027
        """Release resources held by the engine."""
