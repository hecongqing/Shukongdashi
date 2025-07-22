"""
EASY KGQA Framework - 简化版知识图谱问答框架
专为教学设计，包含KGQA的核心功能
"""

from .core.kg_engine import KnowledgeGraphEngine
from .core.easy_analyzer import EasyAnalyzer
from .models.entities import KnowledgeGraphNode, FaultElement, FaultType
from .utils.text_processor import SimpleTextProcessor

__version__ = "1.0.0"
__all__ = [
    "KnowledgeGraphEngine",
    "EasyAnalyzer", 
    "KnowledgeGraphNode",
    "FaultElement",
    "FaultType",
    "SimpleTextProcessor"
]