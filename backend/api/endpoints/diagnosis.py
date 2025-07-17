from __future__ import annotations

"""Diagnosis API endpoint (simplified version).

Exposes a single `/diagnose` POST endpoint that wires the Pydantic models
with the DiagnosisService implemented in `backend.services.diagnosis_service`.
"""

from fastapi import APIRouter, Depends, HTTPException
from backend.models.diagnosis_models import DiagnosisRequest, DiagnosisResponse
from backend.services.diagnosis_service import DiagnosisService

router = APIRouter()


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest, service: DiagnosisService = Depends()):
    """Run fault diagnosis based on the free-text description provided by user."""
    try:
        result_dict = await service.diagnose(request.dict())
        return DiagnosisResponse(**result_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))