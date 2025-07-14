from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from backend.services.diagnosis_service import DiagnosisService
from backend.services.knowledge_graph_service import KnowledgeGraphService
from backend.models.diagnosis_models import (
    DiagnosisRequest,
    DiagnosisResponse,
    FaultSymptom,
    DiagnosisResult,
    FeedbackRequest
)

router = APIRouter()

class DiagnosisInput(BaseModel):
    """故障诊断输入"""
    equipment_brand: Optional[str] = None  # 设备品牌
    equipment_model: Optional[str] = None  # 设备型号
    fault_code: Optional[str] = None       # 故障代码
    fault_description: str                 # 故障描述
    symptoms: List[str] = []               # 故障现象
    operation_history: List[str] = []      # 操作历史
    user_id: Optional[str] = None          # 用户ID
    session_id: Optional[str] = None       # 会话ID

class DiagnosisOutput(BaseModel):
    """故障诊断输出"""
    diagnosis_id: str
    fault_causes: List[Dict[str, Any]]
    repair_suggestions: List[Dict[str, Any]]
    related_symptoms: List[str]
    confidence_score: float
    reasoning_path: List[Dict[str, Any]]
    additional_checks: List[str]
    estimated_repair_time: Optional[int] = None
    difficulty_level: str = "medium"
    created_at: datetime

class InteractiveDiagnosisRequest(BaseModel):
    """交互式诊断请求"""
    diagnosis_id: str
    confirmed_symptoms: List[str]
    rejected_symptoms: List[str]
    additional_info: Optional[str] = None

@router.post("/diagnose", response_model=DiagnosisOutput)
async def diagnose_fault(
    request: DiagnosisInput,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    故障诊断接口
    
    基于用户输入的故障信息，进行智能故障诊断
    """
    try:
        # 创建诊断请求
        diagnosis_request = DiagnosisRequest(
            equipment_brand=request.equipment_brand,
            equipment_model=request.equipment_model,
            fault_code=request.fault_code,
            fault_description=request.fault_description,
            symptoms=request.symptoms,
            operation_history=request.operation_history,
            user_id=request.user_id,
            session_id=request.session_id or str(uuid.uuid4())
        )
        
        # 执行诊断
        diagnosis_result = await diagnosis_service.diagnose(diagnosis_request)
        
        # 构建响应
        response = DiagnosisOutput(
            diagnosis_id=diagnosis_result.diagnosis_id,
            fault_causes=diagnosis_result.fault_causes,
            repair_suggestions=diagnosis_result.repair_suggestions,
            related_symptoms=diagnosis_result.related_symptoms,
            confidence_score=diagnosis_result.confidence_score,
            reasoning_path=diagnosis_result.reasoning_path,
            additional_checks=diagnosis_result.additional_checks,
            estimated_repair_time=diagnosis_result.estimated_repair_time,
            difficulty_level=diagnosis_result.difficulty_level,
            created_at=diagnosis_result.created_at
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"诊断失败: {str(e)}")

@router.post("/interactive-diagnose", response_model=DiagnosisOutput)
async def interactive_diagnose(
    request: InteractiveDiagnosisRequest,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    交互式诊断接口
    
    基于用户反馈的确认信息，进行二次诊断
    """
    try:
        # 执行交互式诊断
        diagnosis_result = await diagnosis_service.interactive_diagnose(
            diagnosis_id=request.diagnosis_id,
            confirmed_symptoms=request.confirmed_symptoms,
            rejected_symptoms=request.rejected_symptoms,
            additional_info=request.additional_info
        )
        
        # 构建响应
        response = DiagnosisOutput(
            diagnosis_id=diagnosis_result.diagnosis_id,
            fault_causes=diagnosis_result.fault_causes,
            repair_suggestions=diagnosis_result.repair_suggestions,
            related_symptoms=diagnosis_result.related_symptoms,
            confidence_score=diagnosis_result.confidence_score,
            reasoning_path=diagnosis_result.reasoning_path,
            additional_checks=diagnosis_result.additional_checks,
            estimated_repair_time=diagnosis_result.estimated_repair_time,
            difficulty_level=diagnosis_result.difficulty_level,
            created_at=diagnosis_result.created_at
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"交互式诊断失败: {str(e)}")

@router.get("/history/{user_id}")
async def get_diagnosis_history(
    user_id: str,
    page: int = 1,
    size: int = 20,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取用户诊断历史
    """
    try:
        history = await diagnosis_service.get_user_diagnosis_history(
            user_id=user_id,
            page=page,
            size=size
        )
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取诊断历史失败: {str(e)}")

@router.get("/detail/{diagnosis_id}")
async def get_diagnosis_detail(
    diagnosis_id: str,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取诊断详情
    """
    try:
        detail = await diagnosis_service.get_diagnosis_detail(diagnosis_id)
        if not detail:
            raise HTTPException(status_code=404, detail="诊断记录不存在")
        return detail
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取诊断详情失败: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    提交诊断反馈
    """
    try:
        success = await diagnosis_service.submit_feedback(request)
        return {"success": success, "message": "反馈提交成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

@router.get("/statistics")
async def get_diagnosis_statistics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取诊断统计信息
    """
    try:
        stats = await diagnosis_service.get_diagnosis_statistics(
            start_date=start_date,
            end_date=end_date
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.get("/fault-types")
async def get_fault_types(
    equipment_type: Optional[str] = None,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取故障类型列表
    """
    try:
        fault_types = await diagnosis_service.get_fault_types(equipment_type)
        return fault_types
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取故障类型失败: {str(e)}")

@router.get("/equipment-types")
async def get_equipment_types(
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取设备类型列表
    """
    try:
        equipment_types = await diagnosis_service.get_equipment_types()
        return equipment_types
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取设备类型失败: {str(e)}")

@router.post("/batch-diagnose")
async def batch_diagnose(
    requests: List[DiagnosisInput],
    diagnosis_service: DiagnosisService = Depends()
):
    """
    批量故障诊断
    """
    try:
        results = []
        for req in requests:
            diagnosis_request = DiagnosisRequest(
                equipment_brand=req.equipment_brand,
                equipment_model=req.equipment_model,
                fault_code=req.fault_code,
                fault_description=req.fault_description,
                symptoms=req.symptoms,
                operation_history=req.operation_history,
                user_id=req.user_id,
                session_id=req.session_id or str(uuid.uuid4())
            )
            
            result = await diagnosis_service.diagnose(diagnosis_request)
            results.append(result)
        
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量诊断失败: {str(e)}")

@router.post("/export-report/{diagnosis_id}")
async def export_diagnosis_report(
    diagnosis_id: str,
    format: str = "pdf",  # pdf, word, excel
    diagnosis_service: DiagnosisService = Depends()
):
    """
    导出诊断报告
    """
    try:
        report = await diagnosis_service.export_diagnosis_report(
            diagnosis_id=diagnosis_id,
            format=format
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出报告失败: {str(e)}")

@router.get("/trending-faults")
async def get_trending_faults(
    days: int = 30,
    limit: int = 10,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    获取趋势故障
    """
    try:
        trending = await diagnosis_service.get_trending_faults(days, limit)
        return trending
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取趋势故障失败: {str(e)}")

@router.post("/similar-cases/{diagnosis_id}")
async def find_similar_cases(
    diagnosis_id: str,
    limit: int = 5,
    diagnosis_service: DiagnosisService = Depends()
):
    """
    查找相似案例
    """
    try:
        similar_cases = await diagnosis_service.find_similar_cases(diagnosis_id, limit)
        return similar_cases
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查找相似案例失败: {str(e)}")