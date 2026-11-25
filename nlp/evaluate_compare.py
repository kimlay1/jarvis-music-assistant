"""Benchmark and compare TF-IDF + LogisticRegression vs. Fine-Tuned DistilBERT."""

from __future__ import annotations

import json
import os
import time

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEST_PATH = "data/processed/test.csv"
LOGREG_PATH = "models/logreg_intent.joblib"
DISTILBERT_DIR = "models/distilbert_intent"
REPORT_PATH = "models/comparison_report.json"


def _play_oos_confusions(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    play_as_oos = (
        int(cm[idx["PlayMusic"], idx["OutOfScope"]])
        if "PlayMusic" in idx and "OutOfScope" in idx
        else 0
    )
    oos_as_play = (
        int(cm[idx["OutOfScope"], idx["PlayMusic"]])
        if "PlayMusic" in idx and "OutOfScope" in idx
        else 0
    )
    return {
        "PlayMusic_as_OutOfScope": play_as_oos,
        "OutOfScope_as_PlayMusic": oos_as_play,
        "total_PlayMusic_OutOfScope_confusions": play_as_oos + oos_as_play,
    }


def evaluate_and_compare(
    test_path: str = TEST_PATH,
    logreg_path: str = LOGREG_PATH,
    distilbert_dir: str = DISTILBERT_DIR,
    report_path: str = REPORT_PATH,
) -> dict:
    """Run both saved models on test.csv, save comparison_report.json, and print table."""
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])
    texts = test_df["text"].astype(str).tolist()
    y_true = test_df["label"].astype(str).tolist()

    # 1. TF-IDF + Logistic Regression
    logreg = joblib.load(logreg_path)
    logreg_preds: list[str] = []
    logreg_latencies: list[float] = []
    for text in texts:
        t0 = time.perf_counter()
        pred = logreg.predict([text])[0]
        logreg_latencies.append((time.perf_counter() - t0) * 1000.0)
        logreg_preds.append(str(pred))

    # 2. Fine-Tuned DistilBERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
    distilbert = AutoModelForSequenceClassification.from_pretrained(distilbert_dir).to(device)
    distilbert.eval()

    bert_preds: list[str] = []
    bert_latencies: list[float] = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=32
            ).to(device)
            t0 = time.perf_counter()
            logits = distilbert(**inputs).logits
            if device.type == "cuda":
                torch.cuda.synchronize()
            bert_latencies.append((time.perf_counter() - t0) * 1000.0)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            bert_preds.append(str(distilbert.config.id2label[pred_id]))

    def _summarize(name: str, preds: list[str], latencies: list[float]) -> dict:
        conf = _play_oos_confusions(y_true, preds)
        return {
            "model": name,
            "accuracy": round(float(accuracy_score(y_true, preds)), 4),
            "macro_f1": round(float(f1_score(y_true, preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_true, preds, average="weighted")), 4),
            "avg_latency_ms": round(float(sum(latencies) / max(1, len(latencies))), 3),
            **conf,
        }

    report = {
        "test_samples": len(texts),
        "device": device.type,
        "models": {
            "logreg": _summarize("TF-IDF + LogisticRegression", logreg_preds, logreg_latencies),
            "distilbert": _summarize("Fine-Tuned DistilBERT", bert_preds, bert_latencies),
        },
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    m_lr = report["models"]["logreg"]
    m_db = report["models"]["distilbert"]
    md_table = "\n".join(
        [
            "| Metric | TF-IDF + LogisticRegression | Fine-Tuned DistilBERT |",
            "| :--- | :---: | :---: |",
            f"| Accuracy | {m_lr['accuracy']:.4f} | {m_db['accuracy']:.4f} |",
            f"| Macro F1 | {m_lr['macro_f1']:.4f} | {m_db['macro_f1']:.4f} |",
            f"| Weighted F1 | {m_lr['weighted_f1']:.4f} | {m_db['weighted_f1']:.4f} |",
            f"| PlayMusic <-> OutOfScope Confusions | {m_lr['total_PlayMusic_OutOfScope_confusions']} | {m_db['total_PlayMusic_OutOfScope_confusions']} |",
            f"| Avg Inference Latency (ms) | {m_lr['avg_latency_ms']:.3f} | {m_db['avg_latency_ms']:.3f} |",
        ]
    )
    print("\n" + md_table + "\n")
    print(f"Saved comparison report to {report_path}")
    return report


if __name__ == "__main__":
    evaluate_and_compare()
