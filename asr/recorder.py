"""Microphone command recorder that stores captured audio as an in-memory WAV buffer."""

from __future__ import annotations

import argparse
import io
import math
import wave
from typing import Iterable, Optional

import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM (paInt16)
CHUNK_SAMPLES = 1280  # 80 ms frame
DEFAULT_DURATION_SECONDS = 4.0


def pcm_frames_to_wav_buffer(
    frames: Iterable[bytes],
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH,
) -> io.BytesIO:
    """Wrap raw 16-bit PCM byte chunks into an in-memory .wav BytesIO object."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        for frame in frames:
            wav_file.writeframes(frame)
    wav_buffer.seek(0)
    return wav_buffer


def record_command(
    duration: float = DEFAULT_DURATION_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    chunk_samples: int = CHUNK_SAMPLES,
    device_index: Optional[int] = None,
    stream: Optional[pyaudio.Stream] = None,
) -> io.BytesIO:
    """Capture `duration` seconds of microphone audio and return an in-memory WAV buffer.

    If an active PyAudio `stream` is passed (for example from WakeWordListener),
    it is reused directly to avoid stream reopen latency or missed syllables.
    Otherwise, a temporary PyAudio stream is opened and cleanly closed.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    total_chunks = max(1, math.ceil((sample_rate * duration) / chunk_samples))
    frames: list[bytes] = []

    owns_stream = stream is None
    pa: Optional[pyaudio.PyAudio] = None
    active_stream = stream

    try:
        if owns_stream:
            pa = pyaudio.PyAudio()
            active_stream = pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_samples,
            )

        for _ in range(total_chunks):
            data = active_stream.read(chunk_samples, exception_on_overflow=False)
            frames.append(data)
    finally:
        if owns_stream:
            if active_stream is not None:
                try:
                    if active_stream.is_active():
                        active_stream.stop_stream()
                    active_stream.close()
                except Exception:
                    pass
            if pa is not None:
                pa.terminate()

    return pcm_frames_to_wav_buffer(
        frames,
        sample_rate=sample_rate,
        channels=CHANNELS,
        sample_width=SAMPLE_WIDTH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a short microphone command into an in-memory WAV buffer."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_SECONDS}).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional file path to dump the recorded in-memory WAV for verification.",
    )
    args = parser.parse_args()

    print(f"Recording for {args.duration:.1f} seconds... Speak now.")
    wav_buffer = record_command(duration=args.duration, device_index=args.device)
    num_bytes = len(wav_buffer.getvalue())
    print(f"Recorded {num_bytes} bytes into in-memory WAV buffer.")

    if args.save:
        with open(args.save, "wb") as f:
            f.write(wav_buffer.getvalue())
        print(f"Saved copy to {args.save}")


if __name__ == "__main__":
    main()
