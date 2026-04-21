"""Base class for translation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator


class TranslationEngine(ABC):
    """Abstract base class for text translation engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language.

        Args:
            text: Text to translate.
            source_lang: Source language code (e.g., 'en').
            target_lang: Target language code (e.g., 'es').

        Returns:
            Translated text string.
        """
        ...

    def translate_streaming(
        self, text: str, source_lang: str, target_lang: str
    ) -> Generator[str, None, None]:
        """Translate with streaming output. Yields partial results.

        Default implementation returns the full result as a single chunk.
        Override for true streaming support.
        """
        yield self.translate(text, source_lang, target_lang)

    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        ...

    def is_available(self) -> bool:
        """Check if this engine is ready to use."""
        return True

    def cleanup(self) -> None:  # noqa: B027
        """Release resources held by the engine."""
