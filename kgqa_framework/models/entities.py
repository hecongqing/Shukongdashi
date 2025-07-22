"""
实体模型定义
定义故障诊断系统中的核心数据实体
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class FaultType(Enum):
    """故障类型枚举"""
    OPERATION = "操作"  # 用户执行的操作
    PHENOMENON = "现象"  # 故障现象
    LOCATION = "部位"   # 故障部位
    ALARM = "报警"     # 报警信息
    CAUSE = "原因"     # 故障原因


@dataclass
class EquipmentInfo:
    """设备信息"""
    brand: Optional[str] = None      # 品牌
    model: Optional[str] = None      # 型号
    error_code: Optional[str] = None # 故障代码
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'brand': self.brand,
            'model': self.model,
            'error_code': self.error_code
        }


@dataclass
class FaultElement:
    """故障元素"""
    content: str                    # 内容
    element_type: FaultType        # 类型
    confidence: float = 0.0        # 置信度
    position: Optional[int] = None # 在原文中的位置
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'type': self.element_type.value,
            'confidence': self.confidence,
            'position': self.position
        }


@dataclass
class KnowledgeGraphNode:
    """知识图谱节点"""
    id: str                        # 节点ID
    label: str                     # 节点标签
    properties: Dict[str, Any]     # 节点属性
    node_type: str                 # 节点类型
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'label': self.label,
            'properties': self.properties,
            'type': self.node_type
        }


@dataclass
class KnowledgeGraphRelation:
    """知识图谱关系"""
    source_id: str                 # 源节点ID
    target_id: str                 # 目标节点ID
    relation_type: str             # 关系类型
    properties: Dict[str, Any]     # 关系属性
    confidence: float = 1.0        # 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source_id,
            'target': self.target_id,
            'type': self.relation_type,
            'properties': self.properties,
            'confidence': self.confidence
        }


@dataclass
class SimilarCase:
    """相似案例"""
    case_id: str                   # 案例ID
    description: str               # 案例描述
    similarity: float              # 相似度
    elements: List[FaultElement]   # 故障元素
    solution: str                  # 解决方案
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'description': self.description,
            'similarity': self.similarity,
            'elements': [elem.to_dict() for elem in self.elements],
            'solution': self.solution
        }


@dataclass
class DiagnosisResult:
    """诊断结果"""
    causes: List[str]              # 可能的故障原因
    solutions: List[str]           # 解决方案
    confidence: float              # 总体置信度
    reasoning_path: List[Dict]     # 推理路径
    similar_cases: List[SimilarCase] # 相似案例
    recommendations: List[str]     # 进一步检查建议
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'causes': self.causes,
            'solutions': self.solutions,
            'confidence': self.confidence,
            'reasoning_path': self.reasoning_path,
            'similar_cases': [case.to_dict() for case in self.similar_cases],
            'recommendations': self.recommendations
        }


@dataclass
class UserQuery:
    """用户查询"""
    equipment_info: EquipmentInfo  # 设备信息
    fault_description: str         # 故障描述
    related_phenomena: List[str]   # 相关现象
    user_feedback: Optional[str]   # 用户反馈
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'equipment_info': self.equipment_info.to_dict(),
            'fault_description': self.fault_description,
            'related_phenomena': self.related_phenomena,
            'user_feedback': self.user_feedback
        }