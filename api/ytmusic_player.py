"""YouTube Music search/playback, in-app player state, and desktop media-key execution."""

from __future__ import annotations

import webbrowser
from typing import Any, Optional

import pyautogui
from ytmusicapi import YTMusic

_ytmusic: Optional[YTMusic] = None


def _get_ytmusic() -> YTMusic:
    global _ytmusic
    if _ytmusic is None:
        _ytmusic = YTMusic()
    return _ytmusic


def _format_track(item: dict) -> str:
    title = item.get("title", "Unknown Track")
    artists = ", ".join(
        a.get("name", "") for a in item.get("artists", []) if a.get("name")
    )
    return f"{title} - {artists}" if artists else title


def search_tracks(query: str, limit: int = 5) -> list[dict[str, str]]:
    """Search YouTube Music for songs (fallback to videos) and return normalized track dicts."""
    yt = _get_ytmusic()
    cleaned_query = query.strip() or "top hits"
    results = yt.search(cleaned_query, filter="songs", limit=limit)
    if not results:
        results = yt.search(cleaned_query, filter="videos", limit=limit)

    tracks: list[dict[str, str]] = []
    for item in results or []:
        video_id = item.get("videoId")
        if not video_id:
            continue
        tracks.append(
            {
                "videoId": str(video_id),
                "title": str(item.get("title", "Unknown Track")),
                "artists": ", ".join(
                    a.get("name", "") for a in item.get("artists", []) if a.get("name")
                ),
                "display": _format_track(item),
            }
        )
    return tracks


def _search_top_track(query: str) -> Optional[dict[str, str]]:
    tracks = search_tracks(query, limit=1)
    return tracks[0] if tracks else None


def default_player_state() -> dict[str, Any]:
    """Return initial in-app player state dictionary."""
    return {
        "queue": [],
        "current_index": 0,
        "is_playing": False,
        "volume": 80,
        "is_muted": False,
        "playlist": [],
    }


def apply_intent_to_player_state(
    intent: str,
    query: str,
    state: dict[str, Any],
) -> str:
    """Apply an intent to the in-app player state and return a user-facing status message."""
    if intent == "PlayMusic":
        tracks = search_tracks(query, limit=5)
        if not tracks:
            return f"No YouTube Music results found for '{query}'."
        state["queue"] = tracks
        state["current_index"] = 0
        state["is_playing"] = True
        return f"Playing in app: {tracks[0]['display']}"

    if intent == "AddToPlaylist":
        track = _search_top_track(query)
        if not track:
            return f"Could not find a matching track to add for '{query}'."
        if not any(t["videoId"] == track["videoId"] for t in state["playlist"]):
            state["playlist"].append(track)
        if not any(t["videoId"] == track["videoId"] for t in state["queue"]):
            state["queue"].append(track)
        return f"Added to in-app playlist: {track['display']}"

    if intent == "PauseMusic":
        if not state["queue"]:
            return "Nothing is currently loaded in the player."
        state["is_playing"] = False
        current = state["queue"][state["current_index"]]
        return f"Paused: {current['display']}"

    if intent == "ResumeMusic":
        if not state["queue"]:
            return "Nothing is currently loaded in the player."
        state["is_playing"] = True
        current = state["queue"][state["current_index"]]
        return f"Resumed: {current['display']}"

    if intent == "NextTrack":
        if not state["queue"]:
            return "Queue is empty. Play a song first."
        state["current_index"] = (state["current_index"] + 1) % len(state["queue"])
        state["is_playing"] = True
        current = state["queue"][state["current_index"]]
        return f"Skipped to next track: {current['display']}"

    if intent == "PreviousTrack":
        if not state["queue"]:
            return "Queue is empty. Play a song first."
        state["current_index"] = (state["current_index"] - 1) % len(state["queue"])
        state["is_playing"] = True
        current = state["queue"][state["current_index"]]
        return f"Returned to previous track: {current['display']}"

    if intent == "VolumeControl":
        q_lower = query.lower()
        if any(w in q_lower for w in ("unmute", "restore")):
            state["is_muted"] = False
            return f"Unmuted (volume {state['volume']}%)."
        if any(w in q_lower for w in ("mute", "silence")):
            state["is_muted"] = True
            return "Muted in-app player."
        if any(w in q_lower for w in ("down", "lower", "quieter", "decrease", "soften", "drop", "minimum")):
            state["is_muted"] = False
            state["volume"] = max(0, int(state["volume"]) - 20)
            return f"Volume lowered to {state['volume']}%."
        state["is_muted"] = False
        state["volume"] = min(100, int(state["volume"]) + 20)
        return f"Volume raised to {state['volume']}%."

    if intent == "OutOfScope":
        return "Ignored (OutOfScope): no media action taken."

    return f"Unknown intent '{intent}': no action taken."


def execute_intent(intent: str, query: str = "") -> str:
    """Standalone/CLI intent executor using browser + OS media keys."""
    if intent == "PlayMusic":
        item = _search_top_track(query)
        if not item:
            return f"No YouTube Music results found for '{query}'."
        url = f"https://music.youtube.com/watch?v={item['videoId']}"
        webbrowser.open(url)
        return f"Playing on YouTube Music: {item['display']}"

    if intent == "AddToPlaylist":
        item = _search_top_track(query)
        if not item:
            return f"Could not find a matching track to add for '{query}'."
        return f"Matched track for playlist: {item['display']}"

    if intent in {"PauseMusic", "ResumeMusic"}:
        pyautogui.press("playpause")
        action = "Paused" if intent == "PauseMusic" else "Resumed"
        return f"{action} media playback (play/pause)."

    if intent == "NextTrack":
        pyautogui.press("nexttrack")
        return "Skipped to next track."

    if intent == "PreviousTrack":
        pyautogui.press("prevtrack")
        return "Returned to previous track."

    if intent == "VolumeControl":
        q_lower = query.lower()
        if any(w in q_lower for w in ("mute", "silence")):
            pyautogui.press("volumemute")
            return "Toggled volume mute."
        if any(w in q_lower for w in ("down", "lower", "quieter", "decrease", "soften", "drop", "minimum")):
            pyautogui.press("volumedown", presses=5)
            return "Turned volume down."
        pyautogui.press("volumeup", presses=5)
        return "Turned volume up."

    if intent == "OutOfScope":
        return "Ignored (OutOfScope): no media action taken."

    return f"Unknown intent '{intent}': no action taken."
