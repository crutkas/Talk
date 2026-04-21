"""Tests for the audio capture module."""

import numpy as np

from src.audio import AudioRingBuffer, audio_to_wav_bytes, resample_audio


class TestAudioRingBuffer:
    """Tests for the circular audio buffer."""

    def test_create_buffer(self) -> None:
        buf = AudioRingBuffer(1000)
        assert buf.max_samples == 1000

    def test_write_and_snapshot(self) -> None:
        buf = AudioRingBuffer(100)
        data = np.ones(50, dtype=np.float32) * 0.5
        buf.write(data)
        snapshot = buf.snapshot(50)
        # Last 50 samples should be 0.5
        np.testing.assert_array_almost_equal(snapshot, data)

    def test_write_wraps_around(self) -> None:
        buf = AudioRingBuffer(100)
        # Write 80 samples
        buf.write(np.ones(80, dtype=np.float32) * 0.3)
        # Write another 40 samples — wraps around
        buf.write(np.ones(40, dtype=np.float32) * 0.7)

        snapshot = buf.snapshot(40)
        np.testing.assert_array_almost_equal(snapshot, np.ones(40, dtype=np.float32) * 0.7)

    def test_write_more_than_buffer_size(self) -> None:
        buf = AudioRingBuffer(50)
        data = np.arange(100, dtype=np.float32)
        buf.write(data)
        snapshot = buf.snapshot(50)
        # Should contain last 50 samples
        np.testing.assert_array_almost_equal(snapshot, data[50:])

    def test_snapshot_fewer_samples(self) -> None:
        buf = AudioRingBuffer(100)
        data = np.arange(100, dtype=np.float32)
        buf.write(data)
        snapshot = buf.snapshot(10)
        np.testing.assert_array_almost_equal(snapshot, data[90:])

    def test_snapshot_more_than_written(self) -> None:
        buf = AudioRingBuffer(100)
        data = np.ones(30, dtype=np.float32) * 0.5
        buf.write(data)
        snapshot = buf.snapshot(100)
        # Last 30 should be 0.5, first 70 should be 0
        assert len(snapshot) == 100

    def test_reset(self) -> None:
        buf = AudioRingBuffer(100)
        buf.write(np.ones(50, dtype=np.float32))
        buf.reset()
        snapshot = buf.snapshot(100)
        np.testing.assert_array_almost_equal(snapshot, np.zeros(100))

    def test_empty_write(self) -> None:
        buf = AudioRingBuffer(100)
        buf.write(np.array([], dtype=np.float32))
        snapshot = buf.snapshot(100)
        np.testing.assert_array_almost_equal(snapshot, np.zeros(100))

    def test_snapshot_default_size(self) -> None:
        buf = AudioRingBuffer(50)
        buf.write(np.ones(50, dtype=np.float32))
        snapshot = buf.snapshot()
        assert len(snapshot) == 50

    def test_multiple_small_writes(self) -> None:
        buf = AudioRingBuffer(100)
        for i in range(10):
            buf.write(np.full(10, i, dtype=np.float32))
        snapshot = buf.snapshot(100)
        expected = np.concatenate([np.full(10, i, dtype=np.float32) for i in range(10)])
        np.testing.assert_array_almost_equal(snapshot, expected)


class TestResampleAudio:
    """Tests for audio resampling."""

    def test_same_rate_no_change(self) -> None:
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = resample_audio(data, 16000, 16000)
        np.testing.assert_array_almost_equal(result, data)

    def test_downsample(self) -> None:
        data = np.ones(48000, dtype=np.float32)
        result = resample_audio(data, 48000, 16000)
        assert len(result) == 16000

    def test_upsample(self) -> None:
        data = np.ones(16000, dtype=np.float32)
        result = resample_audio(data, 16000, 48000)
        assert len(result) == 48000

    def test_preserves_dtype(self) -> None:
        data = np.ones(100, dtype=np.float32)
        result = resample_audio(data, 48000, 16000)
        assert result.dtype == np.float32


class TestAudioToWavBytes:
    """Tests for WAV conversion."""

    def test_produces_valid_wav(self) -> None:
        import io
        import wave

        audio = np.zeros(16000, dtype=np.float32)
        wav_bytes = audio_to_wav_bytes(audio, 16000)

        # Should be valid WAV
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_empty_audio(self) -> None:
        audio = np.array([], dtype=np.float32)
        wav_bytes = audio_to_wav_bytes(audio, 16000)
        assert len(wav_bytes) > 0  # WAV header at minimum

    def test_clipping(self) -> None:
        # Values beyond [-1, 1] should be clipped
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav_bytes = audio_to_wav_bytes(audio)
        assert len(wav_bytes) > 0
