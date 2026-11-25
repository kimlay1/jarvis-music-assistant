from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asr.recorder import (
        CHANNELS,
        CHUNK_SAMPLES,
        DEFAULT_DURATION_SECONDS,
        SAMPLE_RATE,
        SAMPLE_WIDTH,
        pcm_frames_to_wav_buffer,
        record_command,
    )
    from asr.transcriber import SpeechTranscriber

__all__ = [
    "CHANNELS",
    "CHUNK_SAMPLES",
    "DEFAULT_DURATION_SECONDS",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "SpeechTranscriber",
    "pcm_frames_to_wav_buffer",
    "record_command",
]

_RECORDER_EXPORTS = {
    "CHANNELS",
    "CHUNK_SAMPLES",
    "DEFAULT_DURATION_SECONDS",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "pcm_frames_to_wav_buffer",
    "record_command",
}


def __getattr__(name: str):
    if name in _RECORDER_EXPORTS:
        from asr import recorder

        return getattr(recorder, name)
    if name == "SpeechTranscriber":
        from asr import transcriber

        return transcriber.SpeechTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
