"""Train and evaluate the TF-IDF + Logistic Regression intent classifier."""

from __future__ import annotations

import os
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/logreg_intent.joblib"


def train_baseline(
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    model_path: str = MODEL_PATH,
) -> Pipeline:
    """Fit TF-IDF + LogisticRegression on train.csv, evaluate on test.csv, and save."""
    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print(f"Training TF-IDF + LogisticRegression on {len(train_df)} samples...")
    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(str))

    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].astype(str).tolist()
    preds = pipeline.predict(test_texts)

    print("\n=== TF-IDF + Logistic Regression Test Classification Report ===")
    print(classification_report(test_labels, preds, digits=4))

    # Measure average single-query inference latency
    latencies_ms: list[float] = []
    for text in test_texts:
        t0 = time.perf_counter()
        pipeline.predict_proba([text])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = sum(latencies_ms) / max(1, len(latencies_ms))
    print(f"Average single-command inference latency: {avg_latency:.3f} ms")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved baseline model to {model_path}")
    return pipeline


if __name__ == "__main__":
    train_baseline()
