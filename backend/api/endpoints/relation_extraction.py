from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging

from backend.services.relation_extraction_service import RelationExtractionService, Relation

logger = logging.getLogger(__name__)

router = APIRouter()

# 请求模型
class RelationExtractionRequest(BaseModel):
    text: str
    entities: Optional[List[str]] = None

class RelationExtractionWithEntitiesRequest(BaseModel):
    text: str
    entities: List[str]

# 响应模型
class RelationResponse(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str

class RelationExtractionResponse(BaseModel):
    relations: List[RelationResponse]
    statistics: dict
    total_count: int

class RelationStatisticsResponse(BaseModel):
    total_relations: int
    predicate_counts: dict
    confidence_distribution: dict

# 依赖注入
def get_relation_service():
    return RelationExtractionService()

@router.post("/extract", response_model=RelationExtractionResponse)
async def extract_relations(
    request: RelationExtractionRequest,
    relation_service: RelationExtractionService = Depends(get_relation_service)
):
    """
    从文本中提取关系
    
    Args:
        request: 包含文本和可选实体的请求
        
    Returns:
        提取的关系列表和统计信息
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        # 提取关系
        if request.entities:
            relations = await relation_service.extract_relations_with_entities(
                request.text, request.entities
            )
        else:
            relations = await relation_service.extract_relations(request.text)
        
        # 转换为响应格式
        relation_responses = [
            RelationResponse(
                subject=rel.subject,
                predicate=rel.predicate,
                object=rel.object,
                confidence=rel.confidence,
                source_text=rel.source_text
            )
            for rel in relations
        ]
        
        # 获取统计信息
        statistics = relation_service.get_relation_statistics(relations)
        
        return RelationExtractionResponse(
            relations=relation_responses,
            statistics=statistics,
            total_count=len(relations)
        )
        
    except Exception as e:
        logger.error(f"关系抽取失败: {e}")
        raise HTTPException(status_code=500, detail=f"关系抽取失败: {str(e)}")

@router.post("/extract-with-entities", response_model=RelationExtractionResponse)
async def extract_relations_with_entities(
    request: RelationExtractionWithEntitiesRequest,
    relation_service: RelationExtractionService = Depends(get_relation_service)
):
    """
    基于已知实体提取关系
    
    Args:
        request: 包含文本和实体列表的请求
        
    Returns:
        提取的关系列表和统计信息
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        if not request.entities:
            raise HTTPException(status_code=400, detail="实体列表不能为空")
        
        # 提取关系
        relations = await relation_service.extract_relations_with_entities(
            request.text, request.entities
        )
        
        # 转换为响应格式
        relation_responses = [
            RelationResponse(
                subject=rel.subject,
                predicate=rel.predicate,
                object=rel.object,
                confidence=rel.confidence,
                source_text=rel.source_text
            )
            for rel in relations
        ]
        
        # 获取统计信息
        statistics = relation_service.get_relation_statistics(relations)
        
        return RelationExtractionResponse(
            relations=relation_responses,
            statistics=statistics,
            total_count=len(relations)
        )
        
    except Exception as e:
        logger.error(f"基于实体的关系抽取失败: {e}")
        raise HTTPException(status_code=500, detail=f"基于实体的关系抽取失败: {str(e)}")

@router.get("/statistics", response_model=RelationStatisticsResponse)
async def get_relation_statistics(
    relations: List[Relation],
    relation_service: RelationExtractionService = Depends(get_relation_service)
):
    """
    获取关系统计信息
    
    Args:
        relations: 关系列表
        
    Returns:
        统计信息
    """
    try:
        statistics = relation_service.get_relation_statistics(relations)
        return RelationStatisticsResponse(**statistics)
        
    except Exception as e:
        logger.error(f"获取关系统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取关系统计失败: {str(e)}")

@router.get("/patterns")
async def get_relation_patterns(
    relation_service: RelationExtractionService = Depends(get_relation_service)
):
    """
    获取支持的关系模式
    
    Returns:
        关系模式列表
    """
    try:
        patterns = relation_service.relation_patterns
        return {
            "patterns": patterns,
            "common_predicates": list(relation_service.common_predicates)
        }
        
    except Exception as e:
        logger.error(f"获取关系模式失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取关系模式失败: {str(e)}")

@router.post("/test")
async def test_relation_extraction(
    request: RelationExtractionRequest,
    relation_service: RelationExtractionService = Depends(get_relation_service)
):
    """
    测试关系抽取功能
    
    Args:
        request: 测试请求
        
    Returns:
        测试结果
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        # 执行关系抽取
        relations = await relation_service.extract_relations(request.text)
        
        # 返回测试结果
        return {
            "success": True,
            "message": "关系抽取测试成功",
            "extracted_relations_count": len(relations),
            "sample_relations": [
                {
                    "subject": rel.subject,
                    "predicate": rel.predicate,
                    "object": rel.object,
                    "confidence": rel.confidence
                }
                for rel in relations[:5]  # 只返回前5个作为示例
            ]
        }
        
    except Exception as e:
        logger.error(f"关系抽取测试失败: {e}")
        return {
            "success": False,
            "message": f"关系抽取测试失败: {str(e)}",
            "extracted_relations_count": 0,
            "sample_relations": []
        }