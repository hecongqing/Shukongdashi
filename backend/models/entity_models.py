from __future__ import annotations

"""Entity related Pydantic models.

这些模型主要供外层 API / Service 统一返回实体抽取结果时使用，
与 NLPService 内部的 `Entity` NamedTuple 字段保持一致，后续可以将
NLPService 返回的实体转换为该 Pydantic 类型，以便 FastAPI 自动生成
OpenAPI schema。
"""

from enum import Enum
from pydantic import BaseModel


class EntityType(str, Enum):
    EQUIPMENT = "EQUIPMENT"
    FAULT_SYMPTOM = "FAULT_SYMPTOM"
    ALARM_CODE = "ALARM_CODE"
    PART = "PART"
    OPERATION = "OPERATION"


class EntityResult(BaseModel):
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float

    class Config:
        orm_mode = True