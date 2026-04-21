"""Tests for the hotkey manager."""

from unittest.mock import MagicMock

from src.hotkey import AppState, HotkeyManager, parse_hotkey


class TestParseHotkey:
    """Tests for hotkey string parsing."""

    def test_basic_combo(self) -> None:
        result = parse_hotkey("win+ctrl+h")
        assert result == {"win", "ctrl", "h"}

    def test_single_key(self) -> None:
        result = parse_hotkey("f1")
        assert result == {"f1"}

    def test_whitespace_handling(self) -> None:
        result = parse_hotkey("ctrl + shift + a")
        assert result == {"ctrl", "shift", "a"}

    def test_case_insensitive(self) -> None:
        result = parse_hotkey("Win+CTRL+H")
        assert result == {"win", "ctrl", "h"}


class TestAppState:
    """Tests for the application state enum."""

    def test_states_exist(self) -> None:
        assert AppState.IDLE
        assert AppState.RECORDING
        assert AppState.PROCESSING
        assert AppState.TRANSLATING


class TestHotkeyManager:
    """Tests for the hotkey manager state machine."""

    def test_initial_state_is_idle(self) -> None:
        manager = HotkeyManager("win+ctrl+h")
        assert manager.state == AppState.IDLE

    def test_state_setter(self) -> None:
        manager = HotkeyManager("win+ctrl+h")
        manager.state = AppState.PROCESSING
        assert manager.state == AppState.PROCESSING

    def test_callbacks_stored(self) -> None:
        on_start = MagicMock()
        on_stop = MagicMock()
        on_cancel = MagicMock()
        manager = HotkeyManager(
            "win+ctrl+h", on_start=on_start, on_stop=on_stop, on_cancel=on_cancel
        )
        assert manager._on_start is on_start
        assert manager._on_stop is on_stop
        assert manager._on_cancel is on_cancel

    def test_trigger_key_identification(self) -> None:
        manager = HotkeyManager("win+ctrl+h")
        assert manager._trigger_key == "h"

    def test_trigger_key_letter(self) -> None:
        manager = HotkeyManager("ctrl+shift+a")
        assert manager._trigger_key == "a"

    def test_trigger_key_single(self) -> None:
        manager = HotkeyManager("f1")
        assert manager._trigger_key == "f1"

    def test_state_transition_recording_to_processing(self) -> None:
        manager = HotkeyManager("win+ctrl+h")
        manager.state = AppState.RECORDING
        assert manager.state == AppState.RECORDING
        manager.state = AppState.PROCESSING
        assert manager.state == AppState.PROCESSING

    def test_state_transition_processing_to_idle(self) -> None:
        manager = HotkeyManager("win+ctrl+h")
        manager.state = AppState.PROCESSING
        manager.state = AppState.IDLE
        assert manager.state == AppState.IDLE

    def test_cancel_callback_stored(self) -> None:
        on_cancel = MagicMock()
        manager = HotkeyManager("win+ctrl+h", on_cancel=on_cancel)
        assert manager._on_cancel is on_cancel
