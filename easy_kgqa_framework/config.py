"""
简化版配置文件
只包含KGQA教学必要的配置
"""

import os


class EasyConfig:
    """简化配置类"""
    
    # Neo4j 数据库配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:50002')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # 实体识别服务配置
    ENTITY_SERVICE_URL = os.getenv('ENTITY_SERVICE_URL', 'http://127.0.0.1:50003/extract_entities')
    
    # 系统配置
    MAX_QUERY_RESULTS = 10
    MIN_CONFIDENCE_THRESHOLD = 0.5
    
    # API配置
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))


# 默认配置
config = EasyConfig