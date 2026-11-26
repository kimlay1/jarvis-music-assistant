"""Interactive Streamlit control UI for the Hey Jarvis Voice Music Assistant."""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from api.ytmusic_player import (
    apply_intent_to_player_state,
    default_player_state,
)
from asr.transcriber import SpeechTranscriber
from nlp.predictor import IntentEngine, extract_music_query
from wakeword.listener import wait_for_wakeword_and_record

REPORT_PATH = "models/comparison_report.json"
WAKEWORD_TIMEOUT_S = 15.0

st.set_page_config(
    page_title="Hey Jarvis — Voice Music Assistant",
    page_icon=":material/music_note:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Speech-to-Text (faster-whisper)...")
def load_transcriber() -> SpeechTranscriber:
    return SpeechTranscriber(model_size="base.en", device="cpu", compute_type="int8")


@st.cache_resource(show_spinner="Loading NLP Intent Models...")
def load_intent_engine() -> IntentEngine:
    return IntentEngine()


def _models_available() -> bool:
    return os.path.exists("models/logreg_intent.joblib") and os.path.isdir(
        "models/distilbert_intent"
    )


def _init_session_state() -> None:
    if "player_state" not in st.session_state:
        st.session_state.player_state = default_player_state()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _render_in_app_player(state: dict[str, Any]) -> None:
    """Render the embedded YouTube IFrame Player synced to session state."""
    st.subheader("Player")
    queue = state["queue"]
    if not queue:
        st.info("No track loaded yet. Say or type a command like **'play blinding lights'**.")
        return

    idx = state["current_index"] % len(queue)
    current = queue[idx]
    status_badge = "Playing" if state["is_playing"] else "Paused"
    vol_badge = "Muted" if state["is_muted"] else f"Volume {state['volume']}%"

    info_col, ctrl_col = st.columns([3, 2])
    with info_col:
        st.markdown(f"**{status_badge}:** {current['display']}  \n{vol_badge}")
    with ctrl_col:
        b1, b2, b3 = st.columns(3)
        if b1.button("Prev", icon=":material/skip_previous:", use_container_width=True):
            apply_intent_to_player_state("PreviousTrack", "", state)
            st.rerun()
        if state["is_playing"]:
            toggle_label, toggle_icon = "Pause", ":material/pause:"
        else:
            toggle_label, toggle_icon = "Resume", ":material/play_arrow:"
        if b2.button(toggle_label, icon=toggle_icon, use_container_width=True):
            apply_intent_to_player_state(
                "PauseMusic" if state["is_playing"] else "ResumeMusic", "", state
            )
            st.rerun()
        if b3.button("Next", icon=":material/skip_next:", use_container_width=True):
            apply_intent_to_player_state("NextTrack", "", state)
            st.rerun()

    video_id = current["videoId"]
    is_playing_js = "true" if state["is_playing"] else "false"
    is_muted_js = "true" if state["is_muted"] else "false"
    volume_js = int(state["volume"])
    # Same video + same token = same "listen"; pause/resume/volume reruns resume there.
    position_key = f"{video_id}:{int(state.get('play_token', 0))}"

    # Embed YouTube IFrame API so Play/Pause/Volume/Next/Prev stay inside the app.
    # Streamlit rebuilds this iframe whenever player state changes, so the playback
    # position is saved to localStorage and restored on reload.
    html = f"""
    <div id="yt-player-container" style="border-radius:10px;overflow:hidden;">
      <div id="yt-player"></div>
    </div>
    <script>
      var player;
      var POS_STORE = 'jarvis_player_position';
      var POS_KEY = '{position_key}';
      function loadPosition() {{
        try {{
          var saved = JSON.parse(window.localStorage.getItem(POS_STORE) || 'null');
          return (saved && saved.key === POS_KEY) ? Math.floor(saved.t) : 0;
        }} catch (e) {{ return 0; }}
      }}
      function savePosition(t) {{
        try {{
          window.localStorage.setItem(POS_STORE, JSON.stringify({{key: POS_KEY, t: t}}));
        }} catch (e) {{}}
      }}
      function createPlayer() {{
        if (player) {{ return; }}
        player = new YT.Player('yt-player', {{
          height: '220',
          width: '100%',
          videoId: '{video_id}',
          playerVars: {{
            autoplay: {1 if state["is_playing"] else 0},
            controls: 1,
            rel: 0,
            start: loadPosition()
          }},
          events: {{
            'onStateChange': function(event) {{
              if (event.data === YT.PlayerState.ENDED) {{
                savePosition(0);
                return;
              }}
              // A freshly reloaded player reports 0:00 before it seeks; never
              // let that overwrite the real saved position.
              var t = player.getCurrentTime ? player.getCurrentTime() : 0;
              if (t > 0 && (event.data === YT.PlayerState.PLAYING ||
                            event.data === YT.PlayerState.PAUSED)) {{
                savePosition(t);
              }}
            }},
            'onReady': function(event) {{
              setInterval(function() {{
                if (player.getPlayerState && player.getPlayerState() === YT.PlayerState.PLAYING &&
                    player.getCurrentTime() > 0) {{
                  savePosition(player.getCurrentTime());
                }}
              }}, 1000);
              event.target.setVolume({volume_js});
              if ({is_muted_js}) {{
                event.target.mute();
              }} else {{
                event.target.unMute();
              }}
              if ({is_playing_js}) {{
                event.target.playVideo();
              }} else {{
                event.target.pauseVideo();
              }}
            }}
          }}
        }});
      }}
      // Define the ready hook BEFORE loading the API. On reloads the API script can
      // come from cache and fire before a later-defined hook exists, leaving the
      // player blank, so also poll briefly in case the hook was missed.
      window.onYouTubeIframeAPIReady = createPlayer;
      var apiPolls = 0;
      var apiPoll = setInterval(function() {{
        if (player || ++apiPolls > 50) {{ clearInterval(apiPoll); return; }}
        if (window.YT && YT.loaded && YT.Player) {{ createPlayer(); }}
      }}, 200);
    </script>
    <script src="https://www.youtube.com/iframe_api"></script>
    """
    components.html(html, height=235)

    with st.expander(f"Up Next Queue ({len(queue)} tracks)", expanded=False):
        for i, track in enumerate(queue):
            if i == idx:
                st.markdown(f"**{i + 1}. {track['display']}** (now playing)")
            else:
                st.write(f"{i + 1}. {track['display']}")

    if state["playlist"]:
        with st.expander(f"In-App Playlist ({len(state['playlist'])} tracks)", expanded=False):
            for i, track in enumerate(state["playlist"], start=1):
                st.write(f"{i}. {track['display']}")


def _process_command(
    transcript: str,
    engine: IntentEngine,
    active_model_label: str,
) -> None:
    """Classify transcript with both models, execute selected intent in-app, and store result."""
    preds = engine.predict_both(transcript)
    chosen: dict[str, Any] = (
        preds["distilbert"]
        if active_model_label == "Fine-Tuned DistilBERT"
        else preds["logreg"]
    )
    intent = chosen["intent"]
    query = (
        extract_music_query(transcript)
        if intent in {"PlayMusic", "AddToPlaylist"}
        else transcript
    )
    try:
        status_msg = apply_intent_to_player_state(
            intent, query, st.session_state.player_state
        )
    except Exception as exc:  # network / YouTube Music API failures
        status_msg = f"YouTube Music request failed ({type(exc).__name__}: {exc}). Check your internet connection and try again."
        intent_failed = True
    else:
        intent_failed = False
    st.session_state.last_result = {
        "transcript": transcript,
        "preds": preds,
        "active_model": active_model_label,
        "intent": intent,
        "query": query,
        "status_msg": status_msg,
        "failed": intent_failed,
    }


def _render_last_result() -> None:
    res = st.session_state.last_result
    if not res:
        return

    st.markdown(f"**Command:** `{res['transcript']}`")
    col1, col2 = st.columns(2)
    logreg_res = res["preds"]["logreg"]
    bert_res = res["preds"]["distilbert"]

    with col1:
        st.subheader("TF-IDF + Logistic Regression")
        st.metric("Predicted Intent", logreg_res["intent"])
        st.caption(
            f"Confidence: **{logreg_res['confidence'] * 100:.1f}%** | "
            f"Latency: **{logreg_res['latency_ms']:.2f} ms**"
        )

    with col2:
        st.subheader("Fine-Tuned DistilBERT")
        st.metric("Predicted Intent", bert_res["intent"])
        st.caption(
            f"Confidence: **{bert_res['confidence'] * 100:.1f}%** | "
            f"Latency: **{bert_res['latency_ms']:.2f} ms**"
        )

    st.markdown(f"**Active Model:** `{res['active_model']}`")
    if res["intent"] in {"PlayMusic", "AddToPlaylist"}:
        st.markdown(f"**Extracted Query:** `{res['query']}`")
    if res.get("failed"):
        st.error(res["status_msg"])
    elif res["intent"] == "OutOfScope":
        st.info(res["status_msg"])
    else:
        st.success(res["status_msg"])


def main() -> None:
    _init_session_state()

    st.title("Hey Jarvis — Voice-Controlled YouTube Music Assistant")
    st.caption(
        "openWakeWord (`hey_jarvis`) + faster-whisper (`base.en`) + "
        "TF-IDF/LogReg vs. DistilBERT + In-App YouTube Player"
    )

    with st.sidebar:
        st.header("Controls")
        active_model = st.radio(
            "Execution Model",
            options=["Fine-Tuned DistilBERT", "TF-IDF + Logistic Regression"],
            index=0,
        )
        record_seconds = st.slider(
            "Recording Window (seconds)",
            min_value=2.5,
            max_value=6.0,
            value=4.0,
            step=0.5,
        )
        wakeword_threshold = st.slider(
            "Wake-Word Threshold",
            min_value=0.2,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )

        if os.path.exists(REPORT_PATH):
            with st.expander("Test Set Benchmark", expanded=False):
                with open(REPORT_PATH, "r", encoding="utf-8") as f:
                    report = json.load(f)
                for key in ("logreg", "distilbert"):
                    m = report["models"][key]
                    st.markdown(
                        f"**{m['model']}**\n"
                        f"- Accuracy: `{m['accuracy']:.4f}`\n"
                        f"- Macro F1: `{m['macro_f1']:.4f}`\n"
                        f"- Avg Latency: `{m['avg_latency_ms']:.2f} ms`"
                    )

    if not _models_available():
        st.error(
            "Trained models not found in `models/`. Run `nlp/train_baseline.py` "
            "and `nlp/train_transformer.py` (or `notebooks/train_models.ipynb`) first."
        )
        st.stop()

    engine = load_intent_engine()

    voice_tab, sandbox_tab = st.tabs(["Voice Mode", "Text Mode"])

    with voice_tab:
        st.write(
            "Click the button below, say **'Hey Jarvis'**, then speak your music command."
        )
        if st.button("Press to Listen for 'Hey Jarvis'", type="primary"):
            transcriber = load_transcriber()
            with st.status("Initializing microphone...", expanded=True) as status:
                try:
                    audio_float32 = wait_for_wakeword_and_record(
                        record_seconds=record_seconds,
                        threshold=wakeword_threshold,
                        status_callback=lambda msg: status.update(label=msg),
                        timeout=WAKEWORD_TIMEOUT_S,
                    )
                    status.update(label="Transcribing command with faster-whisper...")
                    transcript = transcriber.transcribe(
                        audio_float32, beam_size=3, vad_filter=True
                    )
                    status.update(label="Transcription complete!", state="complete")
                except TimeoutError:
                    status.update(
                        label=f"Didn't hear 'Hey Jarvis' within {WAKEWORD_TIMEOUT_S:.0f}s. "
                        "Press the button to try again.",
                        state="error",
                    )
                    st.stop()
                except Exception as exc:
                    status.update(label=f"Audio capture failed: {exc}", state="error")
                    st.stop()

            if transcript.strip():
                _process_command(transcript, engine, active_model)
            else:
                st.warning("No speech detected in the recording window.")

    with sandbox_tab:
        st.write(
            "Test intent classification and in-app playback controls without a microphone."
        )
        sample_text = st.text_input(
            "Command",
            placeholder="e.g. play blinding lights, pause music, next track, volume down",
        )
        if st.button("Test Text Command"):
            if not sample_text.strip():
                st.warning("Please enter a command to test.")
            else:
                _process_command(sample_text, engine, active_model)

    st.divider()
    _render_in_app_player(st.session_state.player_state)
    if st.session_state.last_result:
        st.divider()
        _render_last_result()


if __name__ == "__main__":
    main()
