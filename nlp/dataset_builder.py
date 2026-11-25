import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

MUSIC_INTENTS = {"PlayMusic", "AddToPlaylist"}

def build_full_dataset():
    print("Downloading Snips dataset from Hugging Face...")
    snips = load_dataset("benayas/snips")

    # Dynamically detect the intent column name ('category', 'label', or 'intent')
    sample_cols = snips["train"].column_names
    intent_col = next(c for c in ["category", "label", "intent"] if c in sample_cols)
    label_feature = snips["train"].features[intent_col]

    def process_split(hf_split):
        df = pd.DataFrame(hf_split)
        df["text"] = df["text"].astype(str).str.lower().str.strip()

        # Convert integer class IDs to string names if needed
        if hasattr(label_feature, "int2str"):
            raw_intents = df[intent_col].apply(label_feature.int2str)
        else:
            raw_intents = df[intent_col].astype(str)

        df["label"] = raw_intents.apply(lambda x: x if x in MUSIC_INTENTS else "OutOfScope")
        return df[["text", "label"]]

    train_df = process_split(snips["train"])
    test_full_df = process_split(snips["test"])

    # Split the 1,400 test rows into 700 validation and 700 test rows
    val_df, test_df = train_test_split(
        test_full_df, test_size=0.5, stratify=test_full_df["label"], random_state=42
    )

    # Load your generated custom playback dataset
    custom_df = pd.read_csv("data/custom_playback.csv")
    c_train, c_temp = train_test_split(custom_df, test_size=0.2, stratify=custom_df["label"], random_state=42)
    c_val, c_test = train_test_split(c_temp, test_size=0.5, stratify=c_temp["label"], random_state=42)

    # Merge Snips + Custom
    final_train = pd.concat([train_df, c_train], ignore_index=True).sample(frac=1, random_state=42)
    final_val = pd.concat([val_df, c_val], ignore_index=True).sample(frac=1, random_state=42)
    final_test = pd.concat([test_df, c_test], ignore_index=True).sample(frac=1, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    final_train.to_csv("data/processed/train.csv", index=False)
    final_val.to_csv("data/processed/val.csv", index=False)
    final_test.to_csv("data/processed/test.csv", index=False)

    print("\nDataset successfully built!")
    print(f"Train size: {len(final_train)} | Val size: {len(final_val)} | Test size: {len(final_test)}")
    print("\nTraining Class Distribution:")
    print(final_train["label"].value_counts())

if __name__ == "__main__":
    build_full_dataset()