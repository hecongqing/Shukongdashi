"""
关系抽取模块

该模块提供从文本中提取实体间关系的功能，支持：
1. 基于规则模式的关系抽取
2. 基于启发式方法的关系抽取
3. 基于已知实体的关系抽取
4. 关系置信度评估
"""

from .relation_extractor import RelationExtractor
from .relation_patterns import RelationPatterns
from .relation_validator import RelationValidator

__all__ = [
    'RelationExtractor',
    'RelationPatterns', 
    'RelationValidator'
]