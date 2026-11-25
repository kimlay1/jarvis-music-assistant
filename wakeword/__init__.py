from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wakeword.listener import (
        CHANNELS,
        CHUNK_SAMPLES,
        DEFAULT_MODEL_NAME,
        DEFAULT_THRESHOLD,
        SAMPLE_RATE,
        WakeWordListener,
        list_input_devices,
    )

__all__ = [
    "CHANNELS",
    "CHUNK_SAMPLES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_THRESHOLD",
    "SAMPLE_RATE",
    "WakeWordListener",
    "list_input_devices",
]


def __getattr__(name: str):
    if name in __all__:
        from wakeword import listener

        return getattr(listener, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
