"""Base class for translation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from src.deps import ensure_packages


class TranslationEngine(ABC):
    """Abstract base class for text translation engines."""

    REQUIRED_PACKAGES: dict[str, str] = {}

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
