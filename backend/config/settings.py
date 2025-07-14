from pydantic import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """应用配置"""
    
    # 基本设置
    APP_NAME: str = "装备制造故障知识图谱系统"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 跨域配置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # Neo4j配置
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password123"
    
    # MySQL配置
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "app_user"
    MYSQL_PASSWORD: str = "app_password"
    MYSQL_DATABASE: str = "equipment_fault"
    
    # Redis配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "redis123"
    REDIS_DB: int = 0
    
    # Elasticsearch配置
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "equipment_fault"
    
    # 模型配置
    MODEL_DIR: str = "models"
    NER_MODEL_PATH: str = "models/ner/best_model"
    RELATION_MODEL_PATH: str = "models/relation/best_model"
    LLM_MODEL_PATH: str = "models/llm/chatglm3-6b"
    
    # 向量数据库配置
    VECTOR_DB_TYPE: str = "faiss"  # faiss, chromadb, qdrant
    VECTOR_DB_PATH: str = "data/vector_db"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 数据路径
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    KNOWLEDGE_DATA_DIR: str = "data/knowledge"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    
    # 缓存配置
    CACHE_TTL: int = 3600  # 1小时
    
    # 分页配置
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # 文件上传配置
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".xls"]
    
    # 爬虫配置
    CRAWLER_USER_AGENT: str = "Equipment Fault Crawler/1.0"
    CRAWLER_DELAY: int = 1
    CRAWLER_TIMEOUT: int = 30
    
    # 模型推理配置
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # 知识图谱配置
    KG_ENTITY_TYPES: List[str] = [
        "设备", "故障现象", "故障原因", "维修方法", "操作步骤", "故障部位", "报警代码"
    ]
    KG_RELATION_TYPES: List[str] = [
        "导致", "解决", "包含", "属于", "关联", "并发", "先导"
    ]
    
    # 推理配置
    INFERENCE_CONFIDENCE_THRESHOLD: float = 0.8
    MAX_INFERENCE_DEPTH: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """获取配置实例"""
    return Settings()