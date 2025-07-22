"""
简洁版知识图谱问答框架 - 用于教学
Easy Knowledge Graph Question Answering Framework for Teaching
"""

__version__ = "1.0.0"
__author__ = "Easy KGQA"

from .core.kgqa_engine import EasyKGQA
from .utils.text_utils import TextUtils
from .utils.entity_recognizer import SimpleEntityRecognizer

__all__ = [
    'EasyKGQA',
    'TextUtils',
    'SimpleEntityRecognizer'
]