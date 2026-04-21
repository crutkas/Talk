"""Floating overlay UI with waveform visualization.

Shows a frameless, always-on-top, non-activating overlay near the cursor
with real-time audio waveform during recording and status indicators.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.audio import AudioRingBuffer

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class OverlayState:
    """Visual states for the overlay."""

    RECORDING = "recording"
    PROCESSING = "processing"
    TRANSLATING = "translating"
    DOWNLOADING = "downloading"
    DONE = "done"
    HIDDEN = "hidden"


if HAS_PYQT6:

    class WaveformWidget(QWidget):
        """Draws a real-time audio waveform from a ring buffer."""

        def __init__(
            self,
            ring_buffer: AudioRingBuffer | None = None,
            color: str = "#4CAF50",
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._ring_buffer = ring_buffer
            self._color = QColor(color)
            self._frozen_data: np.ndarray | None = None
            self.setMinimumHeight(50)

        def set_ring_buffer(self, ring_buffer: AudioRingBuffer) -> None:
            self._ring_buffer = ring_buffer

        def set_color(self, color: str) -> None:
            self._color = QColor(color)

        def freeze(self) -> None:
            """Freeze the waveform display (stop updating from ring buffer)."""
            if self._ring_buffer is not None:
                self._frozen_data = self._ring_buffer.snapshot(self.width())

        def unfreeze(self) -> None:
            """Resume live updates from ring buffer."""
            self._frozen_data = None

        def paintEvent(self, event: object) -> None:  # noqa: N802
            if not HAS_PYQT6:
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            w = self.width()
            h = self.height()
            mid_y = h // 2

            # Get waveform data
            if self._frozen_data is not None:
                data = self._frozen_data
            elif self._ring_buffer is not None:
                data = self._ring_buffer.snapshot(w)
            else:
                data = np.zeros(w, dtype=np.float32)

            # Downsample if needed
            if len(data) > w:
                indices = np.linspace(0, len(data) - 1, w, dtype=int)
                data = data[indices]
            elif len(data) < w:
                data = np.pad(data, (0, w - len(data)))

            # Draw waveform
            pen = QPen(self._color, 2)
            painter.setPen(pen)

            for x in range(w - 1):
                y1 = int(mid_y - data[x] * mid_y * 0.9)
                y2 = int(mid_y - data[x + 1] * mid_y * 0.9)
                painter.drawLine(x, y1, x + 1, y2)

            painter.end()

    class OverlayWindow(QWidget):
        """Floating overlay window that appears near the cursor."""

        # Signals for thread-safe state updates
        show_signal = pyqtSignal()
        hide_signal = pyqtSignal()
        set_state_signal = pyqtSignal(str, str)  # state, extra_text

        def __init__(
            self,
            width: int = 320,
            height: int = 120,
            waveform_color: str = "#4CAF50",
            processing_color: str = "#FF9800",
            translating_color: str = "#2196F3",
            opacity: float = 0.9,
        ) -> None:
            super().__init__()
            self._width = width
            self._height = height
            self._waveform_color = waveform_color
            self._processing_color = processing_color
            self._translating_color = translating_color
            self._state = OverlayState.HIDDEN

            # Window flags: frameless, always on top, tool window (no taskbar),
            # don't take focus
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setFixedSize(width, height)
            self.setWindowOpacity(opacity)

            # Layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            # Waveform widget
            self._waveform = WaveformWidget(color=waveform_color)
            layout.addWidget(self._waveform)

            # Status label
            self._status_label = QLabel("Ready")
            self._status_label.setStyleSheet(
                "color: white; font-size: 11px; background: transparent;"
            )
            self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._status_label)

            # Translation label (hidden by default)
            self._translation_label = QLabel("")
            self._translation_label.setStyleSheet(
                "color: #90CAF9; font-size: 11px; background: transparent;"
            )
            self._translation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._translation_label.setWordWrap(True)
            self._translation_label.hide()
            layout.addWidget(self._translation_label)

            # Refresh timer for waveform (~30fps)
            self._refresh_timer = QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh)
            self._refresh_timer.setInterval(33)

            # Done timer (auto-hide after brief flash)
            self._done_timer = QTimer(self)
            self._done_timer.setSingleShot(True)
            self._done_timer.timeout.connect(self._auto_hide)

            # Connect signals
            self.show_signal.connect(self._do_show)
            self.hide_signal.connect(self._do_hide)
            self.set_state_signal.connect(self._do_set_state)

        def set_ring_buffer(self, ring_buffer: AudioRingBuffer) -> None:
            """Set the ring buffer for waveform visualization."""
            self._waveform.set_ring_buffer(ring_buffer)

        def _position_near_cursor(self) -> None:
            """Position the overlay near the current cursor position."""
            cursor_pos = QCursor.pos()
            screen = QGuiApplication.screenAt(cursor_pos)
            if screen is None:
                screen = QGuiApplication.primaryScreen()

            if screen is None:
                self.move(cursor_pos.x() + 20, cursor_pos.y() + 20)
                return

            screen_geo = screen.availableGeometry()

            # Position below and to the right of cursor
            x = cursor_pos.x() + 20
            y = cursor_pos.y() + 20

            # Clamp to screen bounds
            if x + self._width > screen_geo.right():
                x = cursor_pos.x() - self._width - 20
            if y + self._height > screen_geo.bottom():
                y = cursor_pos.y() - self._height - 20

            x = max(x, screen_geo.left())
            y = max(y, screen_geo.top())

            self.move(x, y)

        def _do_show(self) -> None:
            self._position_near_cursor()
            self.show()
            self._refresh_timer.start()

        def _do_hide(self) -> None:
            self._refresh_timer.stop()
            self._done_timer.stop()
            self._waveform.unfreeze()
            self._translation_label.hide()
            self.hide()
            self._state = OverlayState.HIDDEN

        def _do_set_state(self, state: str, extra_text: str) -> None:
            self._state = state

            if state == OverlayState.RECORDING:
                self._waveform.set_color(self._waveform_color)
                self._waveform.unfreeze()
                self._status_label.setText("🎤 Recording...")
                self._status_label.setStyleSheet(
                    "color: #4CAF50; font-size: 11px; background: transparent;"
                )
                self._translation_label.hide()

            elif state == OverlayState.PROCESSING:
                self._waveform.set_color(self._processing_color)
                self._waveform.freeze()
                suffix = f" ({extra_text})" if extra_text else ""
                self._status_label.setText(f"⏳ Transcribing...{suffix}")
                self._status_label.setStyleSheet(
                    "color: #FF9800; font-size: 11px; background: transparent;"
                )

            elif state == OverlayState.TRANSLATING:
                suffix = f" → {extra_text}" if extra_text else ""
                self._status_label.setText(f"🌐 Translating...{suffix}")
                self._status_label.setStyleSheet(
                    "color: #2196F3; font-size: 11px; background: transparent;"
                )
                self._translation_label.show()

            elif state == OverlayState.DOWNLOADING:
                self._waveform.set_color(self._processing_color)
                self._status_label.setText(extra_text or "⬇️ Downloading model...")
                self._status_label.setStyleSheet(
                    "color: #FF9800; font-size: 11px; background: transparent;"
                )

            elif state == OverlayState.DONE:
                self._status_label.setText("✅ Done")
                self._status_label.setStyleSheet(
                    "color: #4CAF50; font-size: 11px; background: transparent;"
                )
                self._done_timer.start(800)

        def _auto_hide(self) -> None:
            self.hide_signal.emit()

        def _refresh(self) -> None:
            if self._state == OverlayState.RECORDING:
                self._waveform.update()

        def update_translation_text(self, text: str) -> None:
            """Update the translation preview text."""
            self._translation_label.setText(text)
            self._translation_label.show()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            """Draw the rounded background."""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor(30, 30, 30, 230))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(self.rect(), 12, 12)
            painter.end()

else:
    # Stubs when PyQt6 is not available
    class WaveformWidget:  # type: ignore[no-redef]
        pass

    class OverlayWindow:  # type: ignore[no-redef]
        pass
