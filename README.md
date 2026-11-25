# Hey Jarvis — Voice-Controlled YouTube Music Assistant

An end-to-end NLP & speech-controlled YouTube Music assistant built with Python 3.11 and Streamlit. It combines wake-word detection, local speech-to-text, side-by-side intent classification (**TF-IDF + Logistic Regression** vs. **Fine-Tuned DistilBERT**), and an embedded in-app YouTube player.

---

## What It Can Do

- **Wake-Word Detection (`openWakeWord`)**: Listens for `"Hey Jarvis"` in real time (`16 kHz` mono PCM via `PyAudio`, ONNX inference) and captures a configurable command window (`2.5s–6.0s`).
- **Local Speech-to-Text (`faster-whisper`)**: Transcribes recorded voice commands locally using `base.en` (`int8`) with VAD filtering.
- **Side-by-Side Intent Classification**: Runs both a classical baseline (`TF-IDF + Logistic Regression`) and a transformer (`distilbert-base-uncased`) on every command, displaying predicted intent, confidence, and inference latency (ms).
- **8 Supported Intents & In-App Player Controls**:
  - `PlayMusic` — Extracts the song/artist query, searches YouTube Music via `ytmusicapi`, loads the top 5 matches into the queue, and plays the top track inside the embedded Streamlit YouTube IFrame Player.
  - `AddToPlaylist` — Searches YouTube Music for the matching track and adds it to the in-app playlist and queue.
  - `PauseMusic` — Pauses the in-app YouTube player.
  - `ResumeMusic` — Resumes playback in the in-app YouTube player.
  - `NextTrack` / `PreviousTrack` — Cycles forward/backward through the loaded track queue.
  - `VolumeControl` — Raises/lowers volume in `20%` steps (`0–100%`) or mutes/unmutes the in-app player.
  - `OutOfScope` — Safely ignores non-music or non-playback queries (e.g., weather, restaurant booking, *"find the song..."*).
- **Text Sandbox Mode**: Test intent classification, query extraction, and player controls without a microphone.

---

## Model Benchmark (`data/processed/test.csv`, 731 samples)

Dataset: Merged Snips NLU (`PlayMusic`, `AddToPlaylist`, and 5 other intents mapped to `OutOfScope`) + custom playback commands (`60+` per playback class) across `13,324` train / `730` val / `731` test rows.

| Metric | TF-IDF + LogisticRegression | Fine-Tuned DistilBERT |
| :--- | :---: | :---: |
| **Accuracy** | `0.9891` | `0.9918` |
| **Macro F1** | `0.9740` | `0.9845` |
| **Weighted F1** | `0.9891` | `0.9918` |
| **`PlayMusic` <-> `OutOfScope` Confusions** | `4` | `2` |
| **Avg Single-Query Latency (ms)** | `0.211` | `3.333` |

---

## Project Structure

```text
final-project/
├── api/
│   ├── __init__.py
│   └── ytmusic_player.py       # YouTube Music search + in-app player state & CLI media keys
├── asr/
│   ├── __init__.py
│   ├── recorder.py             # In-memory WAV microphone capture
│   ├── speech_pipeline.py      # Standalone CLI wakeword -> ASR loop
│   └── transcriber.py          # faster-whisper SpeechTranscriber
├── data/
│   ├── custom_playback.csv
│   └── processed/              # train.csv, val.csv, test.csv
├── models/
│   ├── logreg_intent.joblib
│   ├── distilbert_intent/
│   └── comparison_report.json
├── nlp/
│   ├── __init__.py
│   ├── generate_custom_data.py # Custom playback command generator
│   ├── dataset_builder.py      # Snips + custom dataset merger
│   ├── train_baseline.py       # TF-IDF + LogisticRegression trainer
│   ├── train_transformer.py    # DistilBERT fine-tuning script
│   ├── evaluate_compare.py     # Side-by-side test benchmark
│   └── predictor.py            # IntentEngine + regex query extraction
├── notebooks/
│   └── train_models.ipynb      # End-to-end training & evaluation notebook
├── wakeword/
│   ├── __init__.py
│   └── listener.py             # openWakeWord ('hey_jarvis') listener
├── app.py                      # Streamlit control UI + in-app YouTube player
└── requirements.txt
```

---

## Getting Started

1. **Install dependencies** (Python 3.11 recommended):
   ```powershell
   pip install -r requirements.txt
   ```
2. **(Optional) Rebuild dataset & retrain models**:
   Run [notebooks/train_models.ipynb](notebooks/train_models.ipynb) or run the scripts in order:
   ```powershell
   python nlp/generate_custom_data.py
   python nlp/dataset_builder.py
   python nlp/train_baseline.py
   python nlp/train_transformer.py
   python nlp/evaluate_compare.py
   ```
3. **Launch the Streamlit app**:
   ```powershell
   streamlit run app.py
   ```

---

## Ideas for Groupmates to Improve

- **Authenticated YouTube Music Playlists**: Connect `ytmusicapi` OAuth/browser headers so `AddToPlaylist` saves songs to a real YouTube Music account playlist instead of only the in-app session playlist.
- **Hands-Free Continuous Listening**: Add an always-on background listener mode in `app.py` (using `streamlit` threading/polling or WebRTC) so users don't have to click the listen button for each command.
- **Entity / Slot Extraction (NER)**: Replace regex-based `extract_music_query()` with a sequence-labeling / NER model to separately extract `song`, `artist`, `playlist_name`, and `volume_level` (e.g., *"set volume to 35%"*).
- **Automated Playback Progression**: Hook the YouTube IFrame `onStateChange` (`YT.PlayerState.ENDED`) event to automatically advance to the next track in the queue.
- **Expanded Evaluation & Unit Tests**: Add a `tests/` suite with `pytest` covering edge cases, noisy ASR transcripts, and out-of-scope adversarial queries.
