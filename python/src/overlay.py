"""Compact voice toolbar overlay — Windows Voice Typing style.

A small, pill-shaped floating toolbar with a mic indicator, status text,
and close button. Matches the compact form factor of Win+H voice typing.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.audio import AudioRingBuffer

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QColor,
        QCursor,
        QFont,
        QGuiApplication,
        QPainter,
        QPen,
    )
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

FONT_FAMILY = "Segoe UI Variable, Segoe UI, sans-serif"


class OverlayState:
    RECORDING = "recording"
    PROCESSING = "processing"
    TRANSLATING = "translating"
    DOWNLOADING = "downloading"
    ERROR = "error"
    DONE = "done"
    HIDDEN = "hidden"


if HAS_PYQT6:

    class MicIndicator(QWidget):
        """Animated microphone indicator — pulses when recording."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setFixedSize(48, 48)
            self._state = "idle"
            self._pulse = 0.0
            self._spin_angle = 0.0
            self._ring_buffer: AudioRingBuffer | None = None

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.setInterval(33)
            self._timer.start()

        def set_ring_buffer(self, rb: AudioRingBuffer) -> None:
            self._ring_buffer = rb

        def set_state(self, state: str) -> None:
            self._state = state

        def _tick(self) -> None:
            self._pulse = (self._pulse + 0.08) % (2 * math.pi)
            self._spin_angle = (self._spin_angle + 6) % 360
            self.update()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            cx, cy = self.width() / 2, self.height() / 2
            r = 20.0

            if self._state == "recording":
                # Pulsing ring based on audio level
                level = 0.3
                if self._ring_buffer is not None:
                    samples = self._ring_buffer.snapshot(512)
                    level = min(float(np.abs(samples).mean()) * 12, 1.0)

                pulse_r = r + 2 + level * 6 + math.sin(self._pulse) * 2
                ring_color = QColor(96, 205, 255, int(80 + level * 120))
                p.setPen(QPen(ring_color, 2.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QRectF(cx - pulse_r, cy - pulse_r, pulse_r * 2, pulse_r * 2))

                # Filled circle
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(96, 205, 255))
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

                # Mic icon (white)
                self._draw_mic(p, cx, cy, QColor(255, 255, 255))

            elif self._state in ("processing", "downloading", "translating"):
                # Spinning arc
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(70, 70, 70))
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

                arc_color = (
                    QColor(96, 205, 255) if self._state != "downloading" else QColor(252, 225, 0)
                )
                pen = QPen(arc_color, 3)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                rect = QRectF(cx - r + 3, cy - r + 3, (r - 3) * 2, (r - 3) * 2)
                p.drawArc(rect, int(self._spin_angle * 16), 240 * 16)

            elif self._state == "done":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(108, 203, 95))
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
                # Checkmark
                p.setPen(QPen(QColor(255, 255, 255), 3, cap=Qt.PenCapStyle.RoundCap))
                p.drawLine(int(cx - 7), int(cy + 1), int(cx - 2), int(cy + 6))
                p.drawLine(int(cx - 2), int(cy + 6), int(cx + 8), int(cy - 5))

            elif self._state == "error":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 153, 164))
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
                # X mark
                p.setPen(QPen(QColor(255, 255, 255), 3, cap=Qt.PenCapStyle.RoundCap))
                p.drawLine(int(cx - 6), int(cy - 6), int(cx + 6), int(cy + 6))
                p.drawLine(int(cx + 6), int(cy - 6), int(cx - 6), int(cy + 6))

            else:
                # Idle
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(70, 70, 70))
                p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
                self._draw_mic(p, cx, cy, QColor(180, 180, 180))

            p.end()

        def _draw_mic(self, p: QPainter, cx: float, cy: float, color: QColor) -> None:
            """Draw a simple microphone icon."""
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            # Mic body
            p.drawRoundedRect(QRectF(cx - 4, cy - 10, 8, 14), 4, 4)
            # Mic stand arc
            pen = QPen(color, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - 8, cy - 6, 16, 16), 0, -180 * 16)
            # Mic stem
            p.drawLine(int(cx), int(cy + 10), int(cx), int(cy + 14))

    class OverlayWindow(QWidget):
        """Compact pill-shaped toolbar — matches Windows Voice Typing style."""

        show_signal = pyqtSignal()
        hide_signal = pyqtSignal()
        set_state_signal = pyqtSignal(str, str)

        def __init__(
            self,
            width: int = 280,
            height: int = 64,
            **kwargs: object,
        ) -> None:
            super().__init__()
            self._width = width
            self._height = height
            self._state = OverlayState.HIDDEN
            self._bg = QColor(43, 43, 43, 245)
            self._border = QColor(255, 255, 255, 18)

            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.setFixedSize(width, height)
            self.setWindowOpacity(0.97)

            # Layout: [status text] [mic indicator] [close btn]
            layout = QHBoxLayout(self)
            layout.setContentsMargins(16, 8, 12, 8)
            layout.setSpacing(8)

            # Status text (left side)
            text_col = QVBoxLayout()
            text_col.setSpacing(0)
            self._status = QLabel("Listening...")
            self._status.setFont(QFont(FONT_FAMILY, 10))
            self._status.setStyleSheet("color: #FFFFFF; background: transparent;")
            text_col.addWidget(self._status)

            self._detail = QLabel("Enter to send · Esc to cancel")
            self._detail.setFont(QFont(FONT_FAMILY, 8))
            self._detail.setStyleSheet("color: rgba(255,255,255,0.45); background: transparent;")
            text_col.addWidget(self._detail)
            layout.addLayout(text_col, 1)

            # Mic indicator (center)
            self._mic = MicIndicator()
            layout.addWidget(self._mic)

            # Close button
            self._close_btn = QLabel("✕")
            self._close_btn.setFont(QFont(FONT_FAMILY, 11))
            self._close_btn.setFixedSize(24, 24)
            self._close_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._close_btn.setStyleSheet("color: rgba(255,255,255,0.45); background: transparent;")
            layout.addWidget(self._close_btn)

            # Timers
            self._done_timer = QTimer(self)
            self._done_timer.setSingleShot(True)
            self._done_timer.timeout.connect(self._auto_hide)

            self.show_signal.connect(self._do_show)
            self.hide_signal.connect(self._do_hide)
            self.set_state_signal.connect(self._do_set_state)

        def set_ring_buffer(self, rb: AudioRingBuffer) -> None:
            self._mic.set_ring_buffer(rb)

        def _position_near_cursor(self) -> None:
            pos = QCursor.pos()
            screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
            if screen is None:
                self.move(pos.x() - self._width // 2, pos.y() - 80)
                return
            geo = screen.availableGeometry()
            # Center horizontally on cursor, above cursor
            x = pos.x() - self._width // 2
            y = pos.y() - self._height - 20
            if y < geo.top():
                y = pos.y() + 20
            x = max(geo.left(), min(x, geo.right() - self._width))
            self.move(x, y)

        def _do_show(self) -> None:
            self._position_near_cursor()
            self.show()

        def _do_hide(self) -> None:
            self._done_timer.stop()
            self.hide()
            self._state = OverlayState.HIDDEN

        def _do_set_state(self, state: str, extra: str) -> None:
            self._state = state
            self._mic.set_state(state)

            if state == OverlayState.RECORDING:
                self._status.setText("Listening...")
                self._status.setStyleSheet("color: #60CDFF; background: transparent;")
                self._detail.setText("Enter to send · Esc to cancel")
                self._detail.show()

            elif state == OverlayState.PROCESSING:
                model = f" · {extra}" if extra else ""
                self._status.setText("Processing...")
                self._status.setStyleSheet("color: #FFFFFF; background: transparent;")
                self._detail.setText(f"Transcribing{model}")
                self._detail.show()

            elif state == OverlayState.TRANSLATING:
                self._status.setText(f"Translating → {extra}" if extra else "Translating...")
                self._status.setStyleSheet("color: #60CDFF; background: transparent;")
                self._detail.setText("Converting language")
                self._detail.show()

            elif state == OverlayState.DOWNLOADING:
                self._status.setText(extra or "Downloading...")
                self._status.setStyleSheet("color: #FCE100; background: transparent;")
                self._detail.setText("One-time setup")
                self._detail.show()

            elif state == OverlayState.DONE:
                self._status.setText("Pasted")
                self._status.setStyleSheet("color: #6CCB5F; background: transparent;")
                self._detail.hide()
                self._done_timer.start(1000)

            elif state == OverlayState.ERROR:
                self._status.setText(extra or "Error")
                self._status.setStyleSheet("color: #FF99A4; background: transparent;")
                self._detail.setText("Check tray for details")
                self._detail.show()
                self._done_timer.start(3500)

        def _auto_hide(self) -> None:
            self.hide_signal.emit()

        def update_translation_text(self, text: str) -> None:
            self._detail.setText(text)
            self._detail.show()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setBrush(self._bg)
            p.setPen(QPen(self._border, 1.0))
            # Pill shape — radius = half height
            p.drawRoundedRect(
                self.rect().adjusted(1, 1, -1, -1),
                self._height // 2,
                self._height // 2,
            )
            p.end()

else:

    class MicIndicator:  # type: ignore[no-redef]
        pass

    class WaveformWidget:  # type: ignore[no-redef]
        pass

    class OverlayWindow:  # type: ignore[no-redef]
        pass
