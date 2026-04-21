"""Tests for STT engine abstraction and factory."""

import pytest

from src.engines.base import STTEngine
from src.engines.factory import available_engines, create_engine


class TestSTTEngineBase:
    """Tests for the STT engine base class."""

    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            STTEngine()  # type: ignore[abstract]

    def test_base_cleanup_noop(self) -> None:
        """Subclass with minimal implementation."""

        class DummyEngine(STTEngine):
            @property
            def name(self) -> str:
                return "dummy"

            def transcribe(self, audio_bytes: bytes) -> str:
                return "test"

        engine = DummyEngine()
        assert engine.name == "dummy"
        assert engine.is_available() is True
        engine.cleanup()  # should not raise


class TestEngineFactory:
    """Tests for the engine factory."""

    def test_available_engines(self) -> None:
        engines = available_engines()
        assert "whisper" in engines
        assert "canary_qwen" in engines
        assert "voxtral" in engines
        assert "qwen3_asr" in engines

    def test_create_whisper_engine(self) -> None:
        engine = create_engine("whisper", {"model_size": "tiny", "device": "cpu"})
        assert "Whisper" in engine.name

    def test_create_canary_engine(self) -> None:
        engine = create_engine(
            "canary_qwen", {"endpoint": "http://localhost:9999/transcribe"}
        )
        assert "Canary" in engine.name

    def test_create_voxtral_engine(self) -> None:
        engine = create_engine(
            "voxtral", {"endpoint": "http://localhost:9999/transcribe"}
        )
        assert "Voxtral" in engine.name

    def test_create_qwen3_engine(self) -> None:
        engine = create_engine("qwen3_asr", {"model_name": "Qwen/Qwen3-ASR-0.6B"})
        assert "Qwen3" in engine.name

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown STT engine"):
            create_engine("nonexistent")

    def test_create_with_empty_config(self) -> None:
        engine = create_engine("whisper", {})
        assert engine is not None

    def test_create_with_none_config(self) -> None:
        engine = create_engine("whisper", None)
        assert engine is not None


class TestWhisperEngine:
    """Tests for the Whisper engine (without loading model)."""

    def test_name(self) -> None:
        engine = create_engine("whisper", {"model_size": "base"})
        assert engine.name == "Whisper (base)"

    def test_cleanup(self) -> None:
        engine = create_engine("whisper")
        engine.cleanup()  # should not raise


class TestHTTPEngines:
    """Tests for HTTP-based engines (without actual server)."""

    def test_canary_not_available_without_server(self) -> None:
        engine = create_engine(
            "canary_qwen", {"endpoint": "http://localhost:19999/transcribe"}
        )
        assert engine.is_available() is False

    def test_voxtral_not_available_without_server(self) -> None:
        engine = create_engine(
            "voxtral", {"endpoint": "http://localhost:19999/transcribe"}
        )
        assert engine.is_available() is False
