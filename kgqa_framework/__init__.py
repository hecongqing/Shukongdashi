"""
基于知识图谱的故障诊断问答框架
Knowledge Graph based Question Answering Framework for Fault Diagnosis
"""

__version__ = "1.0.0"
__author__ = "KGQA Team"

from .core.fault_analyzer import FaultAnalyzer
from .core.kg_engine import KnowledgeGraphEngine
from .core.similarity_matcher import SimilarityMatcher
from .core.solution_recommender import SolutionRecommender
from .models.entities import *
from .utils.text_processor import TextProcessor

__all__ = [
    'FaultAnalyzer',
    'KnowledgeGraphEngine', 
    'SimilarityMatcher',
    'SolutionRecommender',
    'TextProcessor'
]