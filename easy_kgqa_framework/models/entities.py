"""
简化版数据模型
定义KGQA框架的核心数据结构
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class FaultType(Enum):
    """故障元素类型"""
    OPERATION = "操作"      # 用户操作
    PHENOMENON = "现象"     # 故障现象
    LOCATION = "部位"       # 故障部位
    ALARM = "报警"         # 报警信息
    CAUSE = "原因"         # 故障原因
    SOLUTION = "解决方案"   # 解决方法


@dataclass
class FaultElement:
    """故障元素"""
    content: str                    # 元素内容
    element_type: FaultType        # 元素类型
    confidence: float = 1.0        # 置信度


@dataclass
class KnowledgeGraphNode:
    """知识图谱节点"""
    id: str                        # 节点ID
    name: str                      # 节点名称
    labels: List[str]              # 节点标签
    properties: Dict[str, Any]     # 节点属性


@dataclass
class KnowledgeGraphRelation:
    """知识图谱关系"""
    start_node: str               # 起始节点ID
    end_node: str                 # 结束节点ID
    relation_type: str            # 关系类型
    properties: Dict[str, Any]    # 关系属性


@dataclass
class UserQuery:
    """用户查询"""
    question: str                 # 用户问题
    extracted_elements: List[FaultElement] = None  # 提取的故障元素
    
    def __post_init__(self):
        if self.extracted_elements is None:
            self.extracted_elements = []


@dataclass
class AnalysisResult:
    """分析结果"""
    question: str                 # 原始问题
    elements: List[FaultElement]  # 提取的元素
    kg_results: List[Dict]        # 知识图谱查询结果
    reasoning_path: List[str]     # 推理路径
    confidence: float             # 整体置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question": self.question,
            "elements": [
                {
                    "content": elem.content,
                    "type": elem.element_type.value,
                    "confidence": elem.confidence
                } for elem in self.elements
            ],
            "kg_results": self.kg_results,
            "reasoning_path": self.reasoning_path,
            "confidence": self.confidence
        }