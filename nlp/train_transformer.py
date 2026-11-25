"""Fine-tune DistilBERT for 8-class voice command intent classification."""

from __future__ import annotations

import inspect
import os
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
OUTPUT_DIR = "models/distilbert_intent"
MODEL_NAME = "distilbert-base-uncased"


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def train_transformer(
    train_path: str = TRAIN_PATH,
    val_path: str = VAL_PATH,
    test_path: str = TEST_PATH,
    output_dir: str = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    max_length: int = 32,
    learning_rate: float = 3e-5,
    batch_size: int = 32,
    num_epochs: int = 4,
) -> None:
    """Fine-tune DistilBERT on train.csv with val.csv selection, then evaluate on test.csv."""
    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    val_df = pd.read_csv(val_path).dropna(subset=["text", "label"])
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    labels_sorted = sorted(train_df["label"].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict(
            {
                "text": df["text"].astype(str).tolist(),
                "label": [label2id[lbl] for lbl in df["label"].astype(str)],
            }
        )
        return ds.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=max_length
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_ds = _to_hf_dataset(train_df)
    val_ds = _to_hf_dataset(val_df)
    test_ds = _to_hf_dataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "_checkpoints")

    ta_kwargs: dict = {
        "output_dir": checkpoint_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": num_epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": "none",
        "seed": 42,
        "fp16": torch.cuda.is_available(),
    }
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )

    print(
        f"Fine-tuning '{model_name}' on {len(train_df)} samples "
        f"(device={'cuda' if torch.cuda.is_available() else 'cpu'})..."
    )
    trainer.train()

    # Evaluate on test set
    test_output = trainer.predict(test_ds)
    pred_ids = np.argmax(test_output.predictions, axis=-1)
    pred_labels = [id2label[int(i)] for i in pred_ids]
    true_labels = test_df["label"].astype(str).tolist()

    print("\n=== Fine-Tuned DistilBERT Test Classification Report ===")
    print(classification_report(true_labels, pred_labels, digits=4))

    # Measure single-query inference latency on CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sample_texts = test_df["text"].astype(str).tolist()[:200]
    latencies_ms: list[float] = []
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            t0 = time.perf_counter()
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = sum(latencies_ms) / max(1, len(latencies_ms))
    print(f"Average single-command inference latency ({device.type}): {avg_latency:.3f} ms")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned DistilBERT model & tokenizer to {output_dir}")


if __name__ == "__main__":
    train_transformer()
