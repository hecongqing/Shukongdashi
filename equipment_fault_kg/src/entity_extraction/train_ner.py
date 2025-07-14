import argparse
import json
from pathlib import Path
from typing import List, Tuple

import yaml
from loguru import logger

# Re-use the high-level NER wrapper that already exists in this package
from .ner_model import NERModel

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

RELATION_TO_ENTITY_TYPES = {
    # 关系: (头实体类型, 尾实体类型)
    "部件故障": ("Component", "Fault"),
    "性能故障": ("Symptom", "Fault"),  # 将“性能表征”统一视为 Symptom
    "检测工具": ("Tool", "Symptom"),
    "组成": ("Component", "Component"),
}


def build_char_level_labels(text: str, spo_list: List[dict]) -> List[str]:
    """Given a piece of text and its spo annotations build char-level BIO labels.

    The official dataset gives the character offset of each entity span, so we can
    initialise every character with label "O" and then overwrite the gold spans.
    """
    labels = ["O"] * len(text)

    def apply_span(start: int, end: int, ent_type: str):
        if start >= end or start >= len(labels):
            return
        labels[start] = f"B-{ent_type}"
        for idx in range(start + 1, min(end, len(labels))):
            labels[idx] = f"I-{ent_type}"

    for spo in spo_list:
        relation = spo["relation"]
        head_type, tail_type = RELATION_TO_ENTITY_TYPES.get(relation, (None, None))
        # Head entity
        h_start, h_end = spo["h"]["pos"]
        apply_span(h_start, h_end, head_type or "Component")
        # Tail entity
        t_start, t_end = spo["t"]["pos"]
        apply_span(t_start, t_end, tail_type or "Fault")

    return labels


def load_dataset(file_path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read the official jsonl file and return texts + char-level label lists."""
    texts, labels = [], []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["text"]
            spo_list = record.get("spo_list", [])
            texts.append(text)
            labels.append(build_char_level_labels(text, spo_list))
    return texts, labels


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NER model for equipment-fault dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to training jsonl file")
    parser.add_argument("--config", type=str, default="../../config/config.yaml", help="Config YAML path (defaults to project config)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    logger.info(f"Loading dataset from {data_path}")
    texts, char_labels = load_dataset(data_path)
    logger.info(f"Loaded {len(texts)} samples")

    # ------------------------------------------------------------------
    # Init model with project-level config
    # ------------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    ner_cfg = full_cfg["entity_extraction"]

    model_mgr = NERModel(ner_cfg)

    train_loader, val_loader = model_mgr.prepare_data(texts, char_labels)

    logger.info("Start training …")
    model_mgr.train(train_loader, val_loader)

    logger.success("NER training completed 🎉")


if __name__ == "__main__":
    main()