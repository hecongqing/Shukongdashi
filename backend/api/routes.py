from fastapi import APIRouter
from backend.api.endpoints import (
    diagnosis,
    knowledge_graph,
    data_collection,
    entity_extraction,
    relation_extraction,
    qa_system,
    llm_service,
    visualization
)

# 创建主路由器
api_router = APIRouter()

# 包含各个子路由
api_router.include_router(
    diagnosis.router,
    prefix="/diagnosis",
    tags=["故障诊断"]
)

api_router.include_router(
    knowledge_graph.router,
    prefix="/knowledge-graph",
    tags=["知识图谱"]
)

api_router.include_router(
    data_collection.router,
    prefix="/data-collection",
    tags=["数据采集"]
)

api_router.include_router(
    entity_extraction.router,
    prefix="/entity-extraction",
    tags=["实体抽取"]
)

api_router.include_router(
    relation_extraction.router,
    prefix="/relation-extraction",
    tags=["关系抽取"]
)

api_router.include_router(
    qa_system.router,
    prefix="/qa",
    tags=["问答系统"]
)

api_router.include_router(
    llm_service.router,
    prefix="/llm",
    tags=["大模型服务"]
)

api_router.include_router(
    visualization.router,
    prefix="/visualization",
    tags=["可视化"]
)