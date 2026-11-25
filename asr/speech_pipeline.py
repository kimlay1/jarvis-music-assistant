"""Step 3 combined Speech & Wake Word pipeline (Wake Word -> In-Memory WAV -> faster-whisper)."""

from __future__ import annotations

import argparse
from typing import Callable, Optional

from asr.recorder import DEFAULT_DURATION_SECONDS, record_command
from asr.transcriber import SpeechTranscriber
from wakeword.listener import DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD, WakeWordListener


class SpeechPipeline:
    """Combines openWakeWord detection, in-memory command recording, and faster-whisper ASR."""

    def __init__(
        self,
        wakeword_model: str = DEFAULT_MODEL_NAME,
        wakeword_threshold: float = DEFAULT_THRESHOLD,
        whisper_model: str = "base.en",
        whisper_device: str = "auto",
        record_duration: float = DEFAULT_DURATION_SECONDS,
        device_index: Optional[int] = None,
    ) -> None:
        self.record_duration = record_duration
        self.listener = WakeWordListener(
            model_name=wakeword_model,
            threshold=wakeword_threshold,
            device_index=device_index,
        )
        self.transcriber = SpeechTranscriber(
            model_size=whisper_model,
            device=whisper_device,
        )

    def listen_and_transcribe_once(self) -> dict:
        """Wait for wake word, record command into memory, and return transcription details."""
        score = self.listener.wait_for_wakeword(reset_before_listen=True)
        wav_buffer = record_command(
            duration=self.record_duration,
            stream=self.listener.stream,
        )
        result = self.transcriber.transcribe_with_metadata(wav_buffer)
        result["wakeword_score"] = score
        return result

    def run_forever(self, on_transcript: Optional[Callable[[dict], None]] = None) -> None:
        """Continuously listen for the wake word and transcribe spoken commands."""
        with self.listener:
            print(
                f"SpeechPipeline ready: wakeword='{self.listener.model_name}' "
                f"(threshold={self.listener.threshold:.2f}), "
                f"whisper='{self.transcriber.model_size}' ({self.transcriber.device}), "
                f"record={self.record_duration:.1f}s. Press Ctrl+C to stop."
            )
            try:
                while True:
                    print("\nListening for 'hey jarvis'...")
                    score = self.listener.wait_for_wakeword(reset_before_listen=True)
                    print(f"Wake word detected (score={score:.3f}). Recording command ({self.record_duration:.1f}s)...")

                    wav_buffer = record_command(
                        duration=self.record_duration,
                        stream=self.listener.stream,
                    )
                    result = self.transcriber.transcribe_with_metadata(wav_buffer)
                    result["wakeword_score"] = score

                    if on_transcript is not None:
                        on_transcript(result)
                    else:
                        print(
                            f"Transcript: {result['text']!r} "
                            f"(ASR {result['latency_ms']:.1f} ms)"
                        )
            except KeyboardInterrupt:
                print("\nSpeechPipeline stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Step 3 Speech & Wake Word pipeline (hey_jarvis -> WAV -> faster-whisper)."
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--whisper-model", type=str, default="base.en")
    parser.add_argument("--whisper-device", type=str, default="auto")
    parser.add_argument("--mic-device", type=int, default=None)
    args = parser.parse_args()

    pipeline = SpeechPipeline(
        wakeword_threshold=args.threshold,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        record_duration=args.duration,
        device_index=args.mic_device,
    )
    pipeline.run_forever()


if __name__ == "__main__":
    main()
