"""Dataset utilities for R-BERT relation classification."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertTokenizer
from typing import List, Dict, Optional

from .rbert_model import add_special_tokens, mark_entities, SPECIAL_TOKENS


class RbertRelationDataset(Dataset):
    """Prepare inputs for R-BERT training / evaluation."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 256,
    ) -> None:
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-chinese")

        # Register special tokens only once (idempotent)
        add_special_tokens(self.tokenizer)
        self.tokenizer.model_max_length = max_length

        # Mapping relation → id (shared with train_relation.py for compatibility)
        self.relation2id = {
            "部件故障": 0,
            "性能故障": 1,
            "检测工具": 2,
            "组成": 3,
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        # Cache special token ids
        self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

    # ------------------------------------------------------------------
    # Required by torch.utils.data.Dataset
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        head_pos = sample["head_pos"]
        tail_pos = sample["tail_pos"]
        relation_type = sample["relation_type"]

        # Insert entity markers
        marked_text = mark_entities(text, head_pos, tail_pos)

        # Encode
        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Locate marker positions (index of [E1] and [E2])
        e1_idx = (input_ids == self.e1_token_id).nonzero(as_tuple=False)
        e2_idx = (input_ids == self.e2_token_id).nonzero(as_tuple=False)
        # Fallback if marker truncated (should not happen with sensible max_length)
        e1_index = e1_idx[0, 0] if len(e1_idx) else 0
        e2_index = e2_idx[0, 0] if len(e2_idx) else 0

        relation_id = self.relation2id.get(relation_type, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_idx": torch.tensor(e1_index, dtype=torch.long),
            "e2_idx": torch.tensor(e2_index, dtype=torch.long),
            "relation_ids": torch.tensor(relation_id, dtype=torch.long),
        }