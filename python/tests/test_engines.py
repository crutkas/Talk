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
        engine = create_engine("canary_qwen", {"model_name": "nvidia/canary-qwen-2.5b"})
        assert "Canary" in engine.name

    def test_create_voxtral_engine(self) -> None:
        engine = create_engine("voxtral", {"model_name": "mistralai/Voxtral-Mini-4B-Realtime-2602"})
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


class TestInProcessEngines:
    """Tests for in-process engines (without actual model files)."""

    def test_canary_availability(self) -> None:
        engine = create_engine("canary_qwen")
        assert isinstance(engine.is_available(), bool)

    def test_voxtral_availability(self) -> None:
        engine = create_engine("voxtral")
        assert isinstance(engine.is_available(), bool)

    def test_canary_needs_download(self) -> None:
        engine = create_engine("canary_qwen")
        assert isinstance(engine.needs_download(), bool)

    def test_voxtral_needs_download(self) -> None:
        engine = create_engine("voxtral")
        assert isinstance(engine.needs_download(), bool)

    def test_whisper_needs_download(self) -> None:
        engine = create_engine("whisper", {"model_size": "tiny"})
        assert isinstance(engine.needs_download(), bool)

    def test_qwen3_needs_download(self) -> None:
        engine = create_engine("qwen3_asr")
        assert isinstance(engine.needs_download(), bool)
