"""Base class for STT engines."""

from __future__ import annotations

from abc import ABC, abstractmethod


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

    def cleanup(self) -> None:  # noqa: B027
        """Release resources held by the engine."""
