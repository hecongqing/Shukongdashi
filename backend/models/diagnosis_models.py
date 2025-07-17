from __future__ import annotations

"""Diagnosis related Pydantic models used by API endpoint."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from backend.models.entity_models import EntityResult


class DiagnosisRequest(BaseModel):
    """Input payload for /diagnose API."""

    equipment_brand: Optional[str] = Field(None, description="设备品牌")
    equipment_model: Optional[str] = Field(None, description="设备型号")
    fault_description: str = Field(..., description="故障现象原始描述")

    class Config:
        schema_extra = {
            "example": {
                "equipment_brand": "FANUC",
                "equipment_model": "0i",
                "fault_description": "加工中心出现ALM401报警，主轴无法启动"
            }
        }


class DiagnosisResponse(BaseModel):
    """Standard response schema returned by DiagnosisService."""

    diagnosis_id: str
    matched_symptom: Optional[str] = Field(None, description="匹配到的标准化故障现象")
    similarity_score: float = Field(..., ge=0, le=1, description="相似度得分")
    extracted_entities: List[EntityResult]
    fault_causes: List[Dict[str, Any]]
    repair_solutions: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "diagnosis_id": "3abf8d035c874a248571b92d2b149c32",
                "matched_symptom": "主轴无法启动",
                "similarity_score": 0.88,
                "extracted_entities": [
                    {
                        "text": "ALM401",
                        "entity_type": "ALARM_CODE",
                        "start_pos": 5,
                        "end_pos": 11,
                        "confidence": 0.9
                    }
                ],
                "fault_causes": [
                    {"text": "变频器故障", "reliability": 0.9}
                ],
                "repair_solutions": [
                    {"text": "更换变频器", "reliability": 0.9}
                ]
            }
        }