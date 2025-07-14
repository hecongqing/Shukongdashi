"""Data cleaning & preprocessing utilities."""
from __future__ import annotations

import re
import unicodedata
from typing import List

import jieba

__all__ = [
    "normalize_text",
    "clean_text",
    "split_sentences",
]

# ----------------------------------------------------------------------
# Basic cleaners
# ----------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """统一 unicode 表现形式 + 全角转半角。"""
    text = unicodedata.normalize("NFKC", text)

    # 全角转半角
    def _dbc_to_sbc(char):
        code = ord(char)
        if code == 0x3000:
            code = 32
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        return chr(code)

    return "".join(_dbc_to_sbc(c) for c in text)


def clean_text(text: str) -> str:
    """去除多余空白、控制字符、HTML 标签等。"""
    text = normalize_text(text)
    text = re.sub(r"<[^>]+>", "", text)  # HTML 标签
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)  # 控制字符
    text = re.sub(r"\s+", " ", text)  # 多空白
    return text.strip()


# ----------------------------------------------------------------------
# Sentence splitting
# ----------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """基于标点和 jieba 分词进行粗略分句。"""
    # 预处理替换换行
    text = text.replace("\n", "。")
    sentences: List[str] = re.split(r"[。！？!?]\s*", text)
    return [s for s in (s.strip() for s in sentences) if s]