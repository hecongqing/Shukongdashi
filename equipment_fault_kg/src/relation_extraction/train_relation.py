import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from loguru import logger
import yaml

# ---------------------------------------------------------
# Configurations
# ---------------------------------------------------------
RELATION_LABELS = ["部件故障", "性能故障", "检测工具", "组成"]
LABEL2ID = {lbl: idx for idx, lbl in enumerate(RELATION_LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SEPARATOR_TOKEN = "[SEP]"  # use BERT's separator

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class RelationDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], tokenizer: BertTokenizer, max_len: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ---------------------------------------------------------
# Data utils
# ---------------------------------------------------------

def mark_sentence(record: dict) -> List[Tuple[str, int]]:
    """Convert a single JSON record into (input, label) pairs. Each spo becomes a sample.

    We mark subject/object with special separators: subj [SEP] obj [SEP] text
    """
    text = record["text"]
    samples = []
    for spo in record.get("spo_list", []):
        subject = spo["h"]["name"]
        obj = spo["t"]["name"]
        rel = spo["relation"]
        if rel not in LABEL2ID:
            # skip unknown relation type
            continue
        input_text = f"{subject} {SEPARATOR_TOKEN} {obj} {SEPARATOR_TOKEN} {text}"
        samples.append((input_text, LABEL2ID[rel]))
    return samples


def load_relation_dataset(file_path: Path) -> List[Tuple[str, int]]:
    data = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            data.extend(mark_sentence(record))
    return data

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train(model, train_loader, val_loader, epochs: int, lr: float, device):
    optimizer = AdamW(model.parameters(), lr=lr)
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"[Epoch {epoch+1}] train loss: {total_loss / len(train_loader):.4f}")
        f1 = evaluate(model, val_loader, device)
        logger.info(f"[Epoch {epoch+1}] val F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(output_dir / "relation_classifier")
            logger.success(f"New best model saved with F1={best_f1:.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return f1_score(all_labels, all_preds, average="macro")

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train relation classifier for equipment-fault dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to training jsonl file")
    parser.add_argument("--config", type=str, default="../../config/config.yaml", help="Config YAML path")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    logger.info("Preparing data …")
    samples = load_relation_dataset(dataset_path)
    logger.info(f"Collected {len(samples)} relation samples")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["relation_extraction"]["model"]

    model_name = cfg.get("name", "bert-base-chinese")
    max_len = cfg.get("max_length", 256)
    batch_size = cfg.get("batch_size", 32)
    learning_rate = cfg.get("learning_rate", 3e-5)
    epochs = cfg.get("epochs", 3)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Split
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    train_ds = RelationDataset(train_samples, tokenizer, max_len)
    val_ds = RelationDataset(val_samples, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(RELATION_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    logger.info("Start training …")
    train(model, train_loader, val_loader, epochs, learning_rate, device)
    logger.success("Relation training finished 🌟")


if __name__ == "__main__":
    main()