"""
R-BERT relation classification model implementation.

This module defines two components:
1. mark_entities            – helper that inserts entity markers into raw text.
2. RbertRelationModel       – PyTorch model that follows the R-BERT architecture
                               (CLS vector + entity1 vector + entity2 vector).

The implementation is intentionally lightweight so that it can be plugged into
existing training / inference pipelines in *train_relation.py* and
*deploy_relation.py*.

The default entity markers are:  [E1] [/E1]  [E2] [/E2].  They are
registered as *additional_special_tokens* of the HuggingFace tokenizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedTokenizer
from typing import Optional

# ---------------------------------------------------------------------------
# Helper – insert entity markers into raw sentence
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def add_special_tokens(tokenizer: PreTrainedTokenizer):
    """Register R-BERT entity special tokens to a tokenizer (idempotent)."""
    additional_tokens = {
        "additional_special_tokens": [tok for tok in SPECIAL_TOKENS if tok not in tokenizer.additional_special_tokens]
    }
    if additional_tokens["additional_special_tokens"]:
        tokenizer.add_special_tokens(additional_tokens)


def mark_entities(text: str, head_pos: list[int], tail_pos: list[int]) -> str:
    """Insert entity markers into *text* according to absolute character spans.

    Args:
        text:      original sentence.
        head_pos:  [start, end] span (character indices) of head entity.
        tail_pos:  [start, end] span (character indices) of tail entity.

    Returns:
        Marked sentence like:  "... [E1] head_entity [/E1] ... [E2] tail [/E2] ...".
    """
    # Ensure head before tail for deterministic insertion
    h_start, h_end = head_pos
    t_start, t_end = tail_pos
    if h_start < t_start:
        first_start, first_end, first_open, first_close = h_start, h_end, "[E1]", "[/E1]"
        second_start, second_end, second_open, second_close = t_start, t_end, "[E2]", "[/E2]"
    else:
        first_start, first_end, first_open, first_close = t_start, t_end, "[E2]", "[/E2]"
        second_start, second_end, second_open, second_close = h_start, h_end, "[E1]", "[/E1]"

    # Insert from the back so indices remain valid
    marked = (
        text[:second_start]
        + second_open
        + text[second_start:second_end]
        + second_close
        + text[second_end:]
    )
    marked = (
        marked[:first_start]
        + first_open
        + marked[first_start:first_end + len(second_open) + len(second_close)]
        + first_close
        + marked[first_end + len(second_open) + len(second_close):]
    )
    return marked


# ---------------------------------------------------------------------------
# R-BERT model
# ---------------------------------------------------------------------------


class RbertRelationModel(nn.Module):
    """R-BERT: CLS + entity-1 + entity-2 concatenation → softmax"""

    def __init__(
        self,
        bert_model_name: str,
        num_relations: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        hidden = self.bert.config.hidden_size

        # Independent FC layers for three vectors (paper implementation)
        self.cls_fc = nn.Linear(hidden, hidden)
        self.e1_fc = nn.Linear(hidden, hidden)
        self.e2_fc = nn.Linear(hidden, hidden)
        self.activation = nn.Tanh()

        self.classifier = nn.Linear(hidden * 3, num_relations)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_idx: torch.Tensor,
        e2_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Args:
            input_ids:     [B, L]
            attention_mask [B, L]
            e1_idx:        [B] – index of "[E1]" token in each sequence
            e2_idx:        [B] – index of "[E2]" token in each sequence
            labels:        [B] – gold relation ids (optional)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        # Gather vectors
        cls_vec = sequence_output[:, 0]  # CLS
        batch_indices = torch.arange(sequence_output.size(0), device=sequence_output.device)
        e1_vec = sequence_output[batch_indices, e1_idx]
        e2_vec = sequence_output[batch_indices, e2_idx]

        # FC + Tanh (as in paper)
        cls_vec = self.activation(self.cls_fc(self.dropout(cls_vec)))
        e1_vec = self.activation(self.e1_fc(self.dropout(e1_vec)))
        e2_vec = self.activation(self.e2_fc(self.dropout(e2_vec)))

        concat = torch.cat([cls_vec, e1_vec, e2_vec], dim=-1)
        logits = self.classifier(self.dropout(concat))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits