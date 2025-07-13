"""
项目配置文件
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME: str = "知识图谱实战项目"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 数据库配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # 向量数据库配置
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    
    # 大模型配置
    LLM_MODEL_PATH: str = "THUDM/chatglm3-6b"
    LLM_DEVICE: str = "cuda"  # cuda, cpu, mps
    LLM_MAX_LENGTH: int = 2048
    LLM_TEMPERATURE: float = 0.7
    
    # 模型配置
    NER_MODEL_PATH: str = "models/ner_model"
    RE_MODEL_PATH: str = "models/re_model"
    
    # 数据配置
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    ANNOTATIONS_DIR: str = "data/annotations"
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # 缓存配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # 爬虫配置
    CRAWLER_DELAY: float = 1.0
    CRAWLER_TIMEOUT: int = 30
    CRAWLER_USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # 文件上传配置
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()


# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.ANNOTATIONS_DIR,
        settings.UPLOAD_DIR,
        "logs",
        "models",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# 初始化时创建目录
ensure_directories()