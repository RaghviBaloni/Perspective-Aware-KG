import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================
TACO_CSV = "/home/baloni/Perspective-Aware-KG/data/processed/merged_taco_data.csv"
MODEL_DIR = "/home/baloni/Perspective-Aware-KG/data/baseline_models"

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv(TACO_CSV, dtype={"tweet_id": str})
df = df[df["class"] != "undecided"].copy()

# IMPORTANT: sort for deterministic alignment
df = df.sort_values("tweet_id").reset_index(drop=True)

print(f"Loaded {len(df)} tweets for baseline embedding extraction.")

# ============================================
# EXTRACTION FUNCTION
# ============================================
def extract_baseline_embeddings(fold):
    print(f"\n=== Extracting baseline embeddings for fold {fold} ===")

    model_path = f"{MODEL_DIR}/cv_fold_{fold}"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    texts = df["text"].tolist()
    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32)):
            batch = texts[i:i+32]

            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to("cuda")

            # Extract penultimate layer (CLS embedding)
            outputs = model.base_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )

            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            all_embs.append(cls_emb.cpu())

    all_embs = torch.cat(all_embs, dim=0).numpy()

    # Save embeddings + tweet IDs
    np.save(f"{MODEL_DIR}/embeddings_fold_{fold}.npy", all_embs)
    np.save(f"{MODEL_DIR}/tweet_ids_fold_{fold}.npy", df["tweet_id"].values)

    print(f"Saved embeddings for fold {fold}: {all_embs.shape}")


# ============================================
# RUN FOR ALL FOLDS
# ============================================
if __name__ == "__main__":
    for fold in range(10):
        extract_baseline_embeddings(fold)
