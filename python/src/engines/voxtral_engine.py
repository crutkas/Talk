"""Voxtral Transcribe 2 STT engine (in-process).

Downloads and loads the model automatically on first use.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from src.engines.base import STTEngine

logger = logging.getLogger(__name__)

try:
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    HAS_VOXTRAL = True
except ImportError:
    HAS_VOXTRAL = False

try:
    import torchaudio  # type: ignore[import-untyped]

    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class VoxtralEngine(STTEngine):
    """Speech-to-text using Voxtral Transcribe 2 in-process."""

    REQUIRED_PACKAGES = {
        "transformers": "transformers>=4.40",
        "torchaudio": "torchaudio>=2.3",
    }

    def __init__(
        self, model_name: str = "mistralai/Voxtral-Mini-4B-Realtime-2602", **kwargs: Any
    ) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    @property
    def name(self) -> str:
        return "Voxtral Transcribe 2"

    def needs_download(self) -> bool:
        if not HAS_VOXTRAL:
            return False
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        repo_name = self._model_name.replace("/", "--")
        return not os.path.exists(os.path.join(cache_dir, f"models--{repo_name}"))

    def download_model(self, progress_callback: Any | None = None) -> None:
        if not HAS_VOXTRAL:
            raise RuntimeError(
                "transformers is not installed. "
                "Install with: pip install transformers torch torchaudio"
            )
        if progress_callback:
            progress_callback(f"⬇️ Downloading {self.name}... (~8.9GB, first time only)")
        logger.info("Downloading Voxtral model: %s", self._model_name)
        self._ensure_model()
        if progress_callback:
            progress_callback(f"✅ {self.name} ready")

    def _ensure_model(self) -> None:
        if self._model is None:
            if not HAS_VOXTRAL:
                raise RuntimeError(
                    "transformers is not installed. "
                    "Install with: pip install transformers torch torchaudio"
                )
            logger.info("Loading Voxtral model: %s", self._model_name)
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(self._model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        self._ensure_model()

        if not HAS_TORCHAUDIO:
            raise RuntimeError("torchaudio is not installed. Install with: pip install torchaudio")

        # Write WAV bytes to temp file for torchaudio to load
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            waveform, sample_rate = torchaudio.load(temp_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

            inputs = self._processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
            outputs = self._model.generate(**inputs, max_new_tokens=512)
            return str(self._processor.decode(outputs[0], skip_special_tokens=True)).strip()
        finally:
            os.unlink(temp_path)

    def is_available(self) -> bool:
        return HAS_VOXTRAL and HAS_TORCHAUDIO

    def cleanup(self) -> None:
        self._model = None
        self._processor = None
