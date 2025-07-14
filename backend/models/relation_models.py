from pydantic import BaseModel
from enum import Enum
from typing import Optional

class RelationType(str, Enum):
    """关系类型枚举"""
    CAUSES = "CAUSES"           # A 导致 B
    REPAIRS = "REPAIRS"         # A 解决/维修 B
    CONTAINS = "CONTAINS"       # A 包含 B
    BELONGS_TO = "BELONGS_TO"   # A 属于 B
    RELATED_TO = "RELATED_TO"   # 一般关联
    CONCURRENT = "CONCURRENT"   # 并发发生
    PRECEDES = "PRECEDES"       # A 先导 B

class RelationResult(BaseModel):
    """关系抽取结果"""
    source_entity: str                    # 关系源实体文本
    target_entity: str                    # 关系目标实体文本
    relation_type: RelationType           # 关系类型
    confidence: float = 1.0               # 置信度
    extra_info: Optional[dict] = None     # 额外信息（可选）