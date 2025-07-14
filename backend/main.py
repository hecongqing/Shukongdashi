from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config.settings import get_settings
from backend.config.database import init_databases, close_databases
from backend.api.routes import api_router

# 配置日志
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Equipment Fault Knowledge Graph System...")
    
    # 初始化数据库连接
    await init_databases()
    
    # 加载模型（如果需要）
    logger.info("Loading models...")
    
    yield
    
    # 关闭数据库连接
    await close_databases()
    logger.info("Application shutdown complete.")

def create_app() -> FastAPI:
    """创建FastAPI应用实例"""
    settings = get_settings()
    
    app = FastAPI(
        title="装备制造故障知识图谱系统",
        description="基于知识图谱的装备制造故障诊断专家系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 静态文件服务
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # 注册路由
    app.include_router(api_router, prefix="/api/v1")
    
    return app

# 创建应用实例
app = create_app()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "装备制造故障知识图谱系统",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )