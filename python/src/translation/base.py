"""Base class for translation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class TranslationEngine(ABC):
    """Abstract base class for text translation engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language."""
        ...

    def translate_streaming(
        self, text: str, source_lang: str, target_lang: str
    ) -> Generator[str, None, None]:
        """Translate with streaming output. Yields partial results."""
        yield self.translate(text, source_lang, target_lang)

    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
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
