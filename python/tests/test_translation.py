"""Tests for the translation engine abstraction and factory."""

import pytest

from src.translation.base import TranslationEngine
from src.translation.factory import available_translation_engines, create_translation_engine


class TestTranslationEngineBase:
    """Tests for the translation engine base class."""

    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            TranslationEngine()  # type: ignore[abstract]

    def test_base_cleanup_noop(self) -> None:
        class DummyTranslation(TranslationEngine):
            @property
            def name(self) -> str:
                return "dummy"

            def translate(self, text: str, source_lang: str, target_lang: str) -> str:
                return f"translated: {text}"

            def supported_languages(self) -> list[str]:
                return ["en", "es"]

        engine = DummyTranslation()
        assert engine.name == "dummy"
        assert engine.translate("hello", "en", "es") == "translated: hello"
        assert engine.is_available() is True
        engine.cleanup()

    def test_streaming_default(self) -> None:
        class DummyTranslation(TranslationEngine):
            @property
            def name(self) -> str:
                return "dummy"

            def translate(self, text: str, source_lang: str, target_lang: str) -> str:
                return f"translated: {text}"

            def supported_languages(self) -> list[str]:
                return ["en", "es"]

        engine = DummyTranslation()
        chunks = list(engine.translate_streaming("hello", "en", "es"))
        assert chunks == ["translated: hello"]


class TestTranslationFactory:
    """Tests for the translation engine factory."""

    def test_available_engines(self) -> None:
        engines = available_translation_engines()
        assert "nllb-200" in engines
        assert "seamless-m4t" in engines
        assert "madlad-400" in engines

    def test_create_nllb(self) -> None:
        engine = create_translation_engine("nllb-200", {"model_name": "facebook/nllb-200-1.3B"})
        assert "NLLB" in engine.name

    def test_create_seamless(self) -> None:
        engine = create_translation_engine(
            "seamless-m4t", {"model_name": "facebook/seamless-m4t-v2-large"}
        )
        assert "Seamless" in engine.name

    def test_create_madlad(self) -> None:
        engine = create_translation_engine("madlad-400", {"model_name": "google/madlad400-3b-mt"})
        assert "Madlad" in engine.name

    def test_unknown_engine_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown translation engine"):
            create_translation_engine("nonexistent")

    def test_empty_config(self) -> None:
        engine = create_translation_engine("nllb-200", {})
        assert engine is not None

    def test_none_config(self) -> None:
        engine = create_translation_engine("nllb-200", None)
        assert engine is not None


class TestNLLBEngine:
    """Tests for NLLB engine (without loading model)."""

    def test_supported_languages(self) -> None:
        engine = create_translation_engine("nllb-200", {"model_name": "test"})
        langs = engine.supported_languages()
        assert "en" in langs
        assert "es" in langs
        assert "fr" in langs

    def test_empty_text(self) -> None:
        engine = create_translation_engine("nllb-200", {"model_name": "test"})
        result = engine.translate("", "en", "es")
        assert result == ""

    def test_whitespace_text(self) -> None:
        engine = create_translation_engine("nllb-200", {"model_name": "test"})
        result = engine.translate("   ", "en", "es")
        assert result == ""


class TestMadladEngine:
    """Tests for Madlad engine (without loading model)."""

    def test_supported_languages(self) -> None:
        engine = create_translation_engine("madlad-400", {"model_name": "test"})
        langs = engine.supported_languages()
        assert "en" in langs
        assert "ja" in langs

    def test_empty_text(self) -> None:
        engine = create_translation_engine("madlad-400", {"model_name": "test"})
        result = engine.translate("", "en", "es")
        assert result == ""
