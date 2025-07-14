"""BiLSTM-CRF Named Entity Recognition baseline.

依赖：torch>=1.9, torchcrf>=1.1
本示例仅提供模型定义与推理示例，训练代码简化。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchcrf import CRF  # type: ignore


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, tokens: torch.Tensor, tags: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        embeds = self.embedding(tokens)
        feats, _ = self.lstm(embeds)
        emissions = self.fc(feats)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return pred

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

class SimpleTokenizer:
    """Char-level tokenizer for Chinese."""

    def __init__(self):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.idx2char = [self.pad, self.unk]
        self.char2idx = {self.pad: 0, self.unk: 1}

    def fit(self, texts: List[str]):
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.char2idx)
                    self.idx2char.append(ch)

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        ids = [self.char2idx.get(ch, 1) for ch in text[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def vocab_size(self):
        return len(self.idx2char)