from __future__ import annotations

"""Knowledge-base service

This module simulates two responsibilities shown in the architecture diagram:
 1) 根据故障现象检索出 *标准现象*（相似度计算）
 2) 再根据标准现象，从结构化知识库(MySQL 或其他)查询出原因和解决方案

为了让文件即刻可运行，这里使用了 *内存假数据* 与 `difflib.SequenceMatcher` 做
相似度，而不是连接真实 MySQL / Faiss。只要替换对应方法即可升级到真正数据库。
"""

from typing import List, Dict, Tuple, Any
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Very small in-memory dataset – replace with real DB later
# ---------------------------------------------------------------------------

_KB = [
    {
        "id": 1,
        "symptom": "主轴无法启动",
        "causes": [
            {"text": "变频器故障", "reliability": 0.9},
            {"text": "主轴电机烧毁", "reliability": 0.7},
        ],
        "solutions": [
            {"text": "更换变频器", "reliability": 0.9},
            {"text": "检修或更换主轴电机", "reliability": 0.7},
        ],
    },
    {
        "id": 2,
        "symptom": "刀库报警 ATC100",
        "causes": [
            {"text": "刀库位置传感器损坏", "reliability": 0.85}
        ],
        "solutions": [
            {"text": "更换位置传感器", "reliability": 0.85}
        ],
    },
]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _similar(a: str, b: str) -> float:
    """Return a quick ratio in [0,1] using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# KBService – public API
# ---------------------------------------------------------------------------

class KBService:
    """Lightweight KB wrapper providing search + lookup APIs."""

    def __init__(self):
        logger.debug("KBService initialised with %d demo entries", len(_KB))

    # ---------------------------------------------------------------------
    # Search – retrieve top-k standard symptoms given a free-text query
    # ---------------------------------------------------------------------
    def search_standard_symptom(self, query: str, topk: int = 3) -> List[Tuple[str, float]]:
        scored = [
            (item["symptom"], _similar(query, item["symptom"]))
            for item in _KB
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    # ---------------------------------------------------------------------
    # Lookup – given standard symptom, fetch causes & solutions
    # ---------------------------------------------------------------------
    def fetch_causes_solutions(self, symptom_text: str) -> Dict[str, List[Dict[str, Any]]]:
        for item in _KB:
            if item["symptom"] == symptom_text:
                return {
                    "causes": item["causes"],
                    "solutions": item["solutions"],
                }
        return {"causes": [], "solutions": []}