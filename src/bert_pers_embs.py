import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os

from compute_embeddings import normalize_tweet

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "/home/baloni/Perspective-Aware-KG/data/training"
PERSPECTIVE_CSV = "/home/baloni/Perspective-Aware-KG/data/processed/perspective_enriched.csv"
OUTPUT_DIR = "/home/baloni/Perspective-Aware-KG/data/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(PERSPECTIVE_CSV)
df = df[df["taco_class"] != "undecided"].copy()

# Normalize text exactly as in training
df["text"] = df["text"].apply(normalize_tweet)

perspective_cols = [
    'information_providing', 'information_seeking', 'epistemic',
    'experiential', 'opinion', 'aot'
]

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

# -----------------------------
# Load your model class
# -----------------------------
class BERTweetWithPerspectives(torch.nn.Module):
    def __init__(self, num_labels=4, num_perspective_features=6):
        super().__init__()
        self.bertweet = AutoModelForSequenceClassification.from_pretrained(
            "vinai/bertweet-base",
            num_labels=num_labels
        )
        hidden_size = self.bertweet.config.hidden_size
        self.fusion = torch.nn.Linear(hidden_size + num_perspective_features, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, perspectives):
        outputs = self.bertweet.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_embedding, perspectives], dim=1)
        fused = torch.relu(self.fusion(combined))
        return fused  # <-- return fused embedding only


# -----------------------------
# Extraction function
# -----------------------------
def extract_fold_embeddings(fold):
    print(f"\nExtracting embeddings for fold {fold}...")

    # Load trained model
    model_path = f"{MODEL_DIR}/cv_fold_{fold}_perspectives_context"
    model = BERTweetWithPerspectives()
    #model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    state = torch.load(f"{model_path}/pytorch_model.bin", map_location="cuda") 
    model.load_state_dict(state, strict=False)
    model.eval()
    model.cuda()

    # Load fold splits
    fold_df = df.copy()
    fold_df = fold_df.sort_values("tweet_id")  # MUST match KG ordering

    texts = fold_df["text"].tolist()
    persp = torch.FloatTensor(fold_df[perspective_cols].values).cuda()

    all_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32)):
            batch_texts = texts[i:i+32]
            batch_persp = persp[i:i+32]

            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to("cuda")

            fused = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                perspectives=batch_persp
            )

            all_embs.append(fused.cpu())

    all_embs = torch.cat(all_embs, dim=0).numpy()

    # Save
    np.save(f"{OUTPUT_DIR}/embeddings_fold_{fold}.npy", all_embs)
    np.save(f"{OUTPUT_DIR}/tweet_ids_fold_{fold}.npy", fold_df["tweet_id"].values)

    print(f"Saved embeddings for fold {fold}:", all_embs.shape)


# -----------------------------
# Run for all folds
# -----------------------------
for fold in range(10):
    extract_fold_embeddings(fold)
