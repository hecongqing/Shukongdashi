"""
配置文件
定义KGQA框架的各种配置参数
"""

import os
from typing import Dict, Any


class Config:
    """基础配置类"""
    
    # Neo4j 数据库配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # 文件路径配置
    DATA_DIR = os.getenv('KGQA_DATA_DIR', './data')
    CASE_DATABASE_PATH = os.path.join(DATA_DIR, 'cases.pkl')
    VECTORIZER_PATH = os.path.join(DATA_DIR, 'vectorizer.pkl')
    STOPWORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')
    CUSTOM_DICT_PATH = os.path.join(DATA_DIR, 'custom_dict.txt')
    
    # 模型配置
    CNN_MODEL_PATH = os.path.join(DATA_DIR, 'cnn_model')
    
    # 系统配置
    ENABLE_WEB_SEARCH = True
    WEB_SEARCH_TIMEOUT = 10
    MAX_SIMILAR_CASES = 5
    MIN_SIMILARITY_THRESHOLD = 0.1
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(DATA_DIR, 'logs', 'kgqa.log')
    
    # API配置
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            os.path.dirname(cls.LOG_FILE),
            os.path.dirname(cls.CASE_DATABASE_PATH)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """转换为字典格式"""
        config_dict = {}
        for attr_name in dir(cls):
            if attr_name.isupper():
                config_dict[attr_name] = getattr(cls, attr_name)
        return config_dict


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    NEO4J_URI = 'bolt://localhost:7687'
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    ENABLE_WEB_SEARCH = False  # 生产环境可能需要禁用网络搜索


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    NEO4J_URI = 'bolt://localhost:7687'  # 测试数据库
    CASE_DATABASE_PATH = './test_data/test_cases.pkl'
    LOG_LEVEL = 'DEBUG'


# 根据环境变量选择配置
ENV = os.getenv('KGQA_ENV', 'development').lower()

if ENV == 'production':
    current_config = ProductionConfig
elif ENV == 'testing':
    current_config = TestingConfig
else:
    current_config = DevelopmentConfig