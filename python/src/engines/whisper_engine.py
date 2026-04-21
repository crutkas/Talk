"""Whisper STT engine using faster-whisper."""

from __future__ import annotations

import io
import logging
import os
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel, download_model

    HAS_FASTER_WHISPER = True
except ImportError:
    try:
        from faster_whisper import WhisperModel

        HAS_FASTER_WHISPER = True
        download_model = None  # type: ignore[assignment,misc]
    except ImportError:
        HAS_FASTER_WHISPER = False


class WhisperEngine(STTEngine):
    """Speech-to-text using faster-whisper (CTranslate2)."""

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "auto") -> None:
        self._model_size = model_size
        self._device = device
        self._model: WhisperModel | None = None  # type: ignore[assignment]

    @property
    def name(self) -> str:
        return f"Whisper ({self._model_size})"

    def needs_download(self) -> bool:
        """Check if the Whisper model needs to be downloaded."""
        if not HAS_FASTER_WHISPER:
            return False
        # Check if model exists in the default cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        # Model names like "large-v3-turbo" map to repo names
        model_repo = f"models--Systran--faster-whisper-{self._model_size}"
        model_path = os.path.join(cache_dir, model_repo)
        return not os.path.exists(model_path)

    def download_model(self, progress_callback: Any | None = None) -> None:
        """Download the Whisper model with progress updates."""
        if not HAS_FASTER_WHISPER:
            raise RuntimeError("faster-whisper is not installed")

        if progress_callback:
            progress_callback(f"⬇️ Downloading Whisper {self._model_size}...")

        logger.info("Downloading Whisper model: %s", self._model_size)

        # Loading the model triggers the download
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type="auto",
        )

        if progress_callback:
            progress_callback(f"✅ Whisper {self._model_size} ready")

        logger.info("Whisper model downloaded and loaded: %s", self._model_size)

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_FASTER_WHISPER:
                raise RuntimeError("faster-whisper is not installed")
            logger.info("Loading Whisper model: %s", self._model_size)
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type="auto",
            )

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()
        assert self._model is not None

        audio_file = io.BytesIO(audio_bytes)
        segments, _info = self._model.transcribe(
            audio_file,
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments)

    def is_available(self) -> bool:
        return HAS_FASTER_WHISPER

    def cleanup(self) -> None:
        self._model = None
