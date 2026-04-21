"""Tests for the configuration loader."""

import json
import os
import tempfile

from src.config import DEFAULT_CONFIG, _merge_config, load_config


class TestMergeConfig:
    """Tests for deep config merging."""

    def test_simple_override(self) -> None:
        defaults = {"a": 1, "b": 2}
        overrides = {"b": 3}
        result = _merge_config(defaults, overrides)
        assert result == {"a": 1, "b": 3}

    def test_nested_override(self) -> None:
        defaults = {"outer": {"a": 1, "b": 2}}
        overrides = {"outer": {"b": 3}}
        result = _merge_config(defaults, overrides)
        assert result == {"outer": {"a": 1, "b": 3}}

    def test_add_new_key(self) -> None:
        defaults = {"a": 1}
        overrides = {"b": 2}
        result = _merge_config(defaults, overrides)
        assert result == {"a": 1, "b": 2}

    def test_empty_overrides(self) -> None:
        defaults = {"a": 1}
        result = _merge_config(defaults, {})
        assert result == {"a": 1}

    def test_deep_nested(self) -> None:
        defaults = {"level1": {"level2": {"level3": "original"}}}
        overrides = {"level1": {"level2": {"level3": "updated"}}}
        result = _merge_config(defaults, overrides)
        assert result["level1"]["level2"]["level3"] == "updated"


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_from_explicit_path(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"hotkey": "alt+space", "default_model": "voxtral"}, f)
            f.flush()
            path = f.name

        try:
            config = load_config(path)
            assert config["hotkey"] == "alt+space"
            assert config["default_model"] == "voxtral"
            # Defaults should be merged
            assert "audio" in config
            assert "ui" in config
        finally:
            os.unlink(path)

    def test_load_returns_defaults_when_no_file(self) -> None:
        config = load_config("/nonexistent/path/config.json")
        # Should have all required top-level keys from defaults
        assert "hotkey" in config
        assert "default_model" in config
        assert "audio" in config
        assert "translation" in config
        assert "ui" in config

    def test_partial_config_merges_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"hotkey": "f5"}, f)
            f.flush()
            path = f.name

        try:
            config = load_config(path)
            assert config["hotkey"] == "f5"
            assert config["default_model"] == "whisper"
            assert config["audio"]["sample_rate"] == 16000
        finally:
            os.unlink(path)

    def test_default_config_has_required_keys(self) -> None:
        assert "hotkey" in DEFAULT_CONFIG
        assert "default_model" in DEFAULT_CONFIG
        assert "models" in DEFAULT_CONFIG
        assert "audio" in DEFAULT_CONFIG
        assert "translation" in DEFAULT_CONFIG
        assert "ui" in DEFAULT_CONFIG
