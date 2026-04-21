"""Audio capture module with ring buffer for waveform visualization.

Captures microphone audio at device-native sample rate and resamples to 16kHz
for STT. Uses a lock-free circular buffer for waveform visualization.
"""

from __future__ import annotations

import io
import threading
import wave
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import sounddevice, but allow graceful fallback for testing
try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]
    HAS_SOUNDDEVICE = False


TARGET_SAMPLE_RATE = 16000
RING_BUFFER_SECONDS = 3
CHANNELS = 1


class AudioRingBuffer:
    """Lock-free single-producer single-consumer circular buffer for audio samples."""

    def __init__(self, max_samples: int) -> None:
        self._buffer: NDArray[np.float32] = np.zeros(max_samples, dtype=np.float32)
        self._max_samples = max_samples
        self._write_pos = 0
        self._total_written = 0

    @property
    def max_samples(self) -> int:
        return self._max_samples

    def write(self, samples: NDArray[np.float32]) -> None:
        """Write samples into the ring buffer. Called from audio callback thread."""
        n = len(samples)
        if n == 0:
            return

        if n >= self._max_samples:
            # If more samples than buffer, take the last max_samples
            samples = samples[-self._max_samples :]
            n = self._max_samples

        end_pos = self._write_pos + n
        if end_pos <= self._max_samples:
            self._buffer[self._write_pos : end_pos] = samples
        else:
            first_chunk = self._max_samples - self._write_pos
            self._buffer[self._write_pos :] = samples[:first_chunk]
            self._buffer[: n - first_chunk] = samples[first_chunk:]

        self._write_pos = end_pos % self._max_samples
        self._total_written += n

    def snapshot(self, num_samples: int | None = None) -> NDArray[np.float32]:
        """Get a snapshot of the most recent samples. Safe to call from UI thread."""
        if num_samples is None:
            num_samples = self._max_samples

        num_samples = min(num_samples, self._max_samples)
        result = np.empty(num_samples, dtype=np.float32)

        # Read backwards from write position
        start = (self._write_pos - num_samples) % self._max_samples
        if start + num_samples <= self._max_samples:
            result[:] = self._buffer[start : start + num_samples]
        else:
            first_chunk = self._max_samples - start
            result[:first_chunk] = self._buffer[start:]
            result[first_chunk:] = self._buffer[: num_samples - first_chunk]

        return result

    def reset(self) -> None:
        """Reset the buffer."""
        self._buffer[:] = 0
        self._write_pos = 0
        self._total_written = 0


def resample_audio(
    audio: NDArray[np.float32], source_rate: int, target_rate: int
) -> NDArray[np.float32]:
    """Resample audio from source_rate to target_rate using linear interpolation."""
    if source_rate == target_rate:
        return audio

    duration = len(audio) / source_rate
    target_length = int(duration * target_rate)
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_wav_bytes(audio: NDArray[np.float32], sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Convert float32 audio array to WAV bytes."""
    # Convert float32 [-1, 1] to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue()


class AudioRecorder:
    """Manages microphone recording with ring buffer for visualization."""

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._stream: sd.InputStream | None = None  # type: ignore[union-attr]
        self._recording = False
        self._recorded_chunks: list[NDArray[np.float32]] = []
        self._device_sample_rate: int = TARGET_SAMPLE_RATE
        self._lock = threading.Lock()

        # Determine device sample rate
        if HAS_SOUNDDEVICE:
            try:
                device_info = sd.query_devices(device, "input")
                self._device_sample_rate = int(device_info["default_samplerate"])  # type: ignore[index]
            except Exception:
                self._device_sample_rate = TARGET_SAMPLE_RATE

        # Ring buffer sized for visualization at device rate
        buffer_samples = self._device_sample_rate * RING_BUFFER_SECONDS
        self.ring_buffer = AudioRingBuffer(buffer_samples)

    @property
    def device_sample_rate(self) -> int:
        return self._device_sample_rate

    @property
    def is_recording(self) -> bool:
        return self._recording

    def _audio_callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Audio stream callback — runs in real-time thread. Keep minimal work here."""
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        self.ring_buffer.write(mono)

        with self._lock:
            if self._recording:
                self._recorded_chunks.append(mono.copy())

    def start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice is not available")

        with self._lock:
            self._recorded_chunks = []
            self._recording = True

        self.ring_buffer.reset()

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._device_sample_rate,
            channels=CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop_recording(self) -> NDArray[np.float32]:
        """Stop recording and return the captured audio resampled to 16kHz."""
        with self._lock:
            self._recording = False
            chunks = self._recorded_chunks.copy()
            self._recorded_chunks = []

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks)
        return resample_audio(audio, self._device_sample_rate, TARGET_SAMPLE_RATE)

    def get_wav_bytes(self, audio: NDArray[np.float32]) -> bytes:
        """Convert recorded audio to WAV bytes ready for STT."""
        return audio_to_wav_bytes(audio, TARGET_SAMPLE_RATE)
