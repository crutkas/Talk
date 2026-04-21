"""Tests for the paste manager."""

from unittest.mock import patch

from src.paste import PasteManager


class TestPasteManager:
    """Tests for clipboard and paste operations."""

    def test_create_paste_manager(self) -> None:
        pm = PasteManager()
        assert pm._target_hwnd == 0

    def test_capture_target_window(self) -> None:
        pm = PasteManager()
        pm.capture_target_window()
        # On Windows this captures the real window, on CI it's 0
        assert isinstance(pm._target_hwnd, int)

    def test_paste_empty_text_returns_false(self) -> None:
        pm = PasteManager()
        result = pm.paste_text("")
        assert result is False

    def test_paste_none_text_returns_false(self) -> None:
        pm = PasteManager()
        result = pm.paste_text("")
        assert result is False

    @patch("src.paste.pyperclip")
    def test_paste_clipboard_only(self, mock_pyperclip: object) -> None:
        pm = PasteManager()
        result = pm.paste_text("hello world", force_paste=False)
        assert result is False
