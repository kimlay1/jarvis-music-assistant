"""Runtime intent prediction and music query extraction."""

from __future__ import annotations

import re
import time
from typing import Any

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGREG_PATH = "models/logreg_intent.joblib"
DISTILBERT_DIR = "models/distilbert_intent"

_LEADING_PHRASES_RE = re.compile(
    r"^(?:"
    r"(?:hey\s+jarvis|jarvis)[,\s]+|"
    r"(?:please|can\s+you|could\s+you|would\s+you|i\s+want\s+to|i'd\s+like\s+to)\s+|"
    r"(?:put\s+on|play\s+me|play\s+some|play|start\s+playing|listen\s+to|add)\s+"
    r")+",
    flags=re.IGNORECASE,
)

_TRAILING_PHRASES_RE = re.compile(
    r"\s+(?:"
    r"(?:on\s+youtube\s+music|on\s+yt\s+music|on\s+youtube|on\s+spotify|on\s+pandora)|"
    r"(?:to\s+(?:my|the)\s+playlist(?:\s+named\s+.*)?)|"
    r"please"
    r")\s*$",
    flags=re.IGNORECASE,
)


def extract_music_query(text: str) -> str:
    """Strip conversational command wrappers to isolate the song/artist query."""
    cleaned = text.strip()
    # Apply leading stripping iteratively in case of stacked prefixes ("can you please play ...")
    for _ in range(3):
        updated = _LEADING_PHRASES_RE.sub("", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated

    cleaned = _TRAILING_PHRASES_RE.sub("", cleaned).strip(" .,!?")
    return cleaned if cleaned else text.strip()


class IntentEngine:
    """Loads both TF-IDF + LogisticRegression and Fine-Tuned DistilBERT for side-by-side inference."""

    def __init__(
        self,
        logreg_path: str = LOGREG_PATH,
        distilbert_dir: str = DISTILBERT_DIR,
    ) -> None:
        self.logreg = joblib.load(logreg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
        self.distilbert = AutoModelForSequenceClassification.from_pretrained(
            distilbert_dir
        ).to(self.device)
        self.distilbert.eval()

    def predict_logreg(self, text: str) -> dict[str, Any]:
        """Predict intent with TF-IDF + LogisticRegression."""
        normalized = text.lower().strip()
        t0 = time.perf_counter()
        probs = self.logreg.predict_proba([normalized])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_idx = int(probs.argmax())
        return {
            "model": "TF-IDF + Logistic Regression",
            "intent": str(self.logreg.classes_[best_idx]),
            "confidence": float(probs[best_idx]),
            "latency_ms": float(latency_ms),
        }

    def predict_distilbert(self, text: str) -> dict[str, Any]:
        """Predict intent with Fine-Tuned DistilBERT."""
        normalized = text.lower().strip()
        inputs = self.tokenizer(
            normalized, return_tensors="pt", truncation=True, max_length=32
        ).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.distilbert(**inputs).logits
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            probs = torch.softmax(logits, dim=-1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_idx = int(torch.argmax(probs).item())
        return {
            "model": "Fine-Tuned DistilBERT",
            "intent": str(self.distilbert.config.id2label[best_idx]),
            "confidence": float(probs[best_idx].item()),
            "latency_ms": float(latency_ms),
        }

    def predict_both(self, text: str) -> dict[str, dict[str, Any]]:
        """Run both models and return their predictions keyed by short name."""
        return {
            "logreg": self.predict_logreg(text),
            "distilbert": self.predict_distilbert(text),
        }
