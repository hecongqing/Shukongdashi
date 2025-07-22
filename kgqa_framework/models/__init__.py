"""
数据模型包
"""

from .entities import *

__all__ = [
    'FaultType', 'EquipmentInfo', 'FaultElement', 'KnowledgeGraphNode',
    'KnowledgeGraphRelation', 'SimilarCase', 'DiagnosisResult', 'UserQuery'
]