"""Real-time wake word detection using openWakeWord ('hey_jarvis') and PyAudio."""

from __future__ import annotations

import argparse
import time
from typing import Callable, Optional

import numpy as np
import pyaudio
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 1280  # 80 ms frame at 16 kHz (optimal step for openWakeWord)
DEFAULT_MODEL_NAME = "hey_jarvis"
DEFAULT_THRESHOLD = 0.5


def list_input_devices() -> list[dict]:
    """Return available audio input devices with their PyAudio indices."""
    pa = pyaudio.PyAudio()
    devices: list[dict] = []
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) > 0:
                devices.append(
                    {
                        "index": idx,
                        "name": info.get("name", ""),
                        "max_input_channels": int(info.get("maxInputChannels", 0)),
                        "default_sample_rate": int(info.get("defaultSampleRate", 0)),
                    }
                )
    finally:
        pa.terminate()
    return devices


class WakeWordListener:
    """Continuously listens to microphone audio until the wake word triggers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        threshold: float = DEFAULT_THRESHOLD,
        inference_framework: str = "onnx",
        vad_threshold: float = 0.0,
        enable_speex_noise_suppression: bool = False,
        device_index: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples

        self.model = Model(
            wakeword_models=[model_name],
            inference_framework=inference_framework,
            vad_threshold=vad_threshold,
            enable_speex_noise_suppression=enable_speex_noise_suppression,
        )

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    @property
    def stream(self) -> Optional[pyaudio.Stream]:
        """Expose the active PyAudio input stream for seamless handoff to recording."""
        return self._stream

    def open_stream(self) -> pyaudio.Stream:
        """Open (or return) the PyAudio microphone input stream."""
        if self._stream is not None and self._stream.is_active():
            return self._stream

        if self._pa is None:
            self._pa = pyaudio.PyAudio()

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_samples,
        )
        return self._stream

    def close(self) -> None:
        """Cleanly close the PyAudio stream and terminate PyAudio."""
        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None

    def reset(self) -> None:
        """Reset openWakeWord internal audio and prediction buffers."""
        self.model.reset()

    def __enter__(self) -> "WakeWordListener":
        self.open_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def predict_frame(self, pcm_bytes: bytes) -> float:
        """Run wake word inference on a single 16-bit PCM audio frame."""
        audio_frame = np.frombuffer(pcm_bytes, dtype=np.int16)
        predictions = self.model.predict(audio_frame)
        return float(predictions.get(self.model_name, 0.0))

    def wait_for_wakeword(
        self,
        threshold: Optional[float] = None,
        timeout: Optional[float] = None,
        reset_before_listen: bool = True,
        on_score: Optional[Callable[[float], None]] = None,
    ) -> Optional[float]:
        """Block in a lightweight loop until the wake word score meets threshold.

        Returns the triggering confidence score, or None if `timeout` elapsed.
        """
        target_threshold = self.threshold if threshold is None else threshold
        stream = self.open_stream()

        if reset_before_listen:
            self.reset()

        start_time = time.monotonic()
        while True:
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                return None

            raw_bytes = stream.read(self.chunk_samples, exception_on_overflow=False)
            score = self.predict_frame(raw_bytes)

            if on_score is not None:
                on_score(score)

            if score >= target_threshold:
                return score


def wait_for_wakeword_and_record(
    record_seconds: float = 4.0,
    threshold: float = DEFAULT_THRESHOLD,
    status_callback: Optional[Callable[[str], None]] = None,
    device_index: Optional[int] = None,
    timeout: Optional[float] = None,
) -> np.ndarray:
    """Wait for 'hey_jarvis', record `record_seconds` of 16 kHz mono audio, and return float32 [-1, 1]."""
    if record_seconds <= 0:
        raise ValueError(f"record_seconds must be positive, got {record_seconds}")

    def _notify(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    _notify("Listening for 'Hey Jarvis'...")
    with WakeWordListener(
        model_name=DEFAULT_MODEL_NAME,
        threshold=threshold,
        inference_framework="onnx",
        device_index=device_index,
    ) as listener:
        score = listener.wait_for_wakeword(
            threshold=threshold,
            timeout=timeout,
            reset_before_listen=True,
        )
        if score is None:
            raise TimeoutError("Timed out waiting for wake word 'hey_jarvis'.")

        _notify(
            f"Wake word detected (score={score:.2f}). Recording for {record_seconds:.1f}s..."
        )
        total_chunks = max(
            1, int(np.ceil((listener.sample_rate * record_seconds) / listener.chunk_samples))
        )
        frames: list[bytes] = []
        stream = listener.open_stream()
        for _ in range(total_chunks):
            frames.append(stream.read(listener.chunk_samples, exception_on_overflow=False))

    pcm_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
    return (pcm_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test real-time openWakeWord ('hey_jarvis') detection."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection confidence threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available microphone input devices and exit.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep listening after each detection instead of exiting after the first hit.",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Print live frame scores above 0.05 for threshold tuning.",
    )
    args = parser.parse_args()

    if args.list_devices:
        for dev in list_input_devices():
            print(
                f"[{dev['index']}] {dev['name']} "
                f"(channels={dev['max_input_channels']}, rate={dev['default_sample_rate']})"
            )
        return

    def _log_score(score: float) -> None:
        if args.show_scores and score >= 0.05:
            print(f"  score={score:.3f}")

    with WakeWordListener(
        threshold=args.threshold,
        device_index=args.device,
    ) as listener:
        print(
            f"Listening for '{listener.model_name}' "
            f"(threshold={args.threshold:.2f}). Press Ctrl+C to stop."
        )
        try:
            while True:
                score = listener.wait_for_wakeword(
                    on_score=_log_score if args.show_scores else None
                )
                print(f"[WAKE WORD DETECTED] '{listener.model_name}' (score={score:.3f})")
                if not args.continuous:
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped wake word listener.")


if __name__ == "__main__":
    main()
