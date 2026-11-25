"""Local Speech-to-Text (ASR) transcription using faster-whisper."""

from __future__ import annotations

import argparse
import io
import time
from typing import Any, Union

import numpy as np
import torch
from faster_whisper import WhisperModel

from asr.recorder import DEFAULT_DURATION_SECONDS, record_command

AudioInput = Union[io.BytesIO, str, np.ndarray]


class SpeechTranscriber:
    """Wraps faster-whisper ('base.en' / 'small.en') for low-latency command transcription."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_size = model_size
        self.device, self.compute_type = self._resolve_runtime(device, compute_type)

        try:
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=local_files_only,
            )
        except Exception:
            if self.device == "cuda":
                # Fallback to CPU int8 if CUDA libraries fail at runtime
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=local_files_only,
                )
            else:
                raise

    @staticmethod
    def _resolve_runtime(device: str, compute_type: str) -> tuple[str, str]:
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved_compute = compute_type
        if resolved_compute == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        return resolved_device, resolved_compute

    def transcribe_with_metadata(
        self,
        audio_input: AudioInput,
        beam_size: int = 5,
        language: str = "en",
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        """Transcribe an in-memory WAV BytesIO, file path, or float32 numpy array."""
        if isinstance(audio_input, io.BytesIO):
            audio_input.seek(0)

        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_input,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "text": text,
            "duration_s": float(getattr(info, "duration", 0.0)),
            "latency_ms": latency_ms,
            "language": getattr(info, "language", language),
        }

    def transcribe(
        self,
        audio_input: AudioInput,
        beam_size: int = 5,
        language: str = "en",
        vad_filter: bool = True,
    ) -> str:
        """Return only the transcribed text string."""
        result = self.transcribe_with_metadata(
            audio_input,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
        )
        return str(result["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Speech-to-Text transcription with faster-whisper."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        help="Whisper model size (e.g., 'base.en' or 'small.en').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to an existing audio file to transcribe.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record live microphone audio into memory and transcribe it.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Recording duration in seconds when --record is used (default: {DEFAULT_DURATION_SECONDS}).",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="Optional PyAudio input device index when --record is used.",
    )
    args = parser.parse_args()

    transcriber = SpeechTranscriber(model_size=args.model, device=args.device)
    print(f"Loaded faster-whisper '{transcriber.model_size}' on {transcriber.device} ({transcriber.compute_type}).")

    if args.file:
        result = transcriber.transcribe_with_metadata(args.file)
    elif args.record:
        print(f"Recording for {args.duration:.1f} seconds... Speak now.")
        wav_buffer = record_command(duration=args.duration, device_index=args.mic_device)
        result = transcriber.transcribe_with_metadata(wav_buffer)
    else:
        parser.error("Provide either --file <path> or --record.")
        return

    print(f"Transcript : {result['text']!r}")
    print(f"Audio len  : {result['duration_s']:.2f}s")
    print(f"ASR latency: {result['latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
