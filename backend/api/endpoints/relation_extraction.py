from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List

from backend.services.relation_extraction_service import RelationExtractionService
from backend.models.relation_models import RelationResult

router = APIRouter()

class RelationExtractionInput(BaseModel):
    text: str

class BatchRelationExtractionInput(BaseModel):
    texts: List[str]

@router.post("/extract", response_model=List[RelationResult])
async def extract_relations(
    request: RelationExtractionInput,
    relation_service: RelationExtractionService = Depends(),
):
    try:
        results = await relation_service.extract_relations(request.text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关系抽取失败: {str(e)}")

@router.post("/batch-extract", response_model=List[List[RelationResult]])
async def batch_extract_relations(
    request: BatchRelationExtractionInput,
    relation_service: RelationExtractionService = Depends(),
):
    try:
        results = await relation_service.batch_extract_relations(request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量关系抽取失败: {str(e)}")