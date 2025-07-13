"""
知识图谱项目配置文件
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from loguru import logger

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 环境变量配置
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# 数据库配置
DATABASE_CONFIG = {
    # Neo4j 图数据库
    "neo4j": {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "knowledge123"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    },
    
    # Redis 缓存
    "redis": {
        "url": os.getenv("REDIS_URL", "redis://:redis123@localhost:6379"),
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5
    },
    
    # MongoDB 文档数据库
    "mongodb": {
        "url": os.getenv("MONGODB_URL", "mongodb://admin:mongo123@localhost:27017"),
        "database": os.getenv("MONGODB_DATABASE", "knowledge_graph")
    },
    
    # Elasticsearch
    "elasticsearch": {
        "hosts": [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")],
        "timeout": 30,
        "max_retries": 3
    }
}

# 模型配置
MODEL_CONFIG = {
    # 实体识别模型
    "ner": {
        "model_name": "bert-base-chinese",
        "model_path": str(BASE_DIR / "models" / "ner"),
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "labels": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    },
    
    # 关系抽取模型
    "relation_extraction": {
        "model_name": "bert-base-chinese",
        "model_path": str(BASE_DIR / "models" / "relation_extraction"),
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "relations": ["出生于", "毕业于", "工作于", "位于", "属于", "其他"]
    },
    
    # 大语言模型
    "llm": {
        "model_name": "chatglm3-6b",
        "model_path": str(BASE_DIR / "models" / "chatglm3-6b"),
        "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    },
    
    # 知识图谱嵌入
    "kg_embedding": {
        "model_name": "TransE",
        "embedding_dim": 128,
        "learning_rate": 0.001,
        "batch_size": 1024,
        "num_epochs": 100
    }
}

# 数据配置
DATA_CONFIG = {
    "raw_data_dir": str(BASE_DIR / "data" / "raw"),
    "processed_data_dir": str(BASE_DIR / "data" / "processed"),
    "annotated_data_dir": str(BASE_DIR / "data" / "annotated"),
    "corpus_dir": str(BASE_DIR / "data" / "corpus"),
    "output_dir": str(BASE_DIR / "data" / "output"),
    
    # 数据采集配置
    "crawler": {
        "max_pages": 1000,
        "delay": 1,
        "timeout": 30,
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    },
    
    # 数据预处理
    "preprocessing": {
        "min_text_length": 10,
        "max_text_length": 1000,
        "remove_duplicates": True,
        "clean_html": True,
        "normalize_whitespace": True
    }
}

# 训练配置
TRAINING_CONFIG = {
    "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
    "mixed_precision": True,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_steps": 500,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
    "early_stopping_patience": 3,
    "seed": 42
}

# API配置
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "workers": int(os.getenv("API_WORKERS", 1)),
    "reload": DEBUG,
    "access_log": True,
    "cors_origins": ["*"] if DEBUG else [],
    "rate_limit": {
        "requests_per_minute": 100,
        "burst": 10
    }
}

# 日志配置
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "100 MB",
    "retention": "30 days",
    "log_dir": str(BASE_DIR / "logs")
}

# 知识图谱Schema配置
GRAPH_SCHEMA = {
    "entities": {
        "Person": {
            "properties": ["name", "birth_date", "nationality", "description"],
            "constraints": ["name"]
        },
        "Organization": {
            "properties": ["name", "founded_date", "type", "description"],
            "constraints": ["name"]
        },
        "Location": {
            "properties": ["name", "type", "coordinates", "description"],
            "constraints": ["name"]
        },
        "Event": {
            "properties": ["name", "date", "location", "description"],
            "constraints": ["name", "date"]
        }
    },
    "relations": {
        "BORN_IN": {"from": "Person", "to": "Location"},
        "WORKS_FOR": {"from": "Person", "to": "Organization"},
        "LOCATED_IN": {"from": "Organization", "to": "Location"},
        "PARTICIPATED_IN": {"from": "Person", "to": "Event"},
        "HAPPENED_IN": {"from": "Event", "to": "Location"}
    }
}

# 问答系统配置
QA_CONFIG = {
    "similarity_threshold": 0.8,
    "max_results": 10,
    "answer_generation": {
        "max_length": 200,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "query_expansion": True,
    "use_cache": True,
    "cache_ttl": 3600  # 1小时
}

# 评估配置
EVALUATION_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "cross_validation_folds": 5,
    "metrics": ["precision", "recall", "f1", "accuracy"],
    "save_predictions": True
}

# 部署配置
DEPLOYMENT_CONFIG = {
    "model_serving": {
        "timeout": 30,
        "max_batch_size": 32,
        "cache_size": 1000
    },
    "monitoring": {
        "enable_metrics": True,
        "metrics_port": 9090,
        "health_check_interval": 30
    }
}

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                import json
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}

def get_config() -> Dict[str, Any]:
    """获取完整配置"""
    config = {
        "environment": ENV,
        "debug": DEBUG,
        "base_dir": str(BASE_DIR),
        "database": DATABASE_CONFIG,
        "model": MODEL_CONFIG,
        "data": DATA_CONFIG,
        "training": TRAINING_CONFIG,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "graph_schema": GRAPH_SCHEMA,
        "qa": QA_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "deployment": DEPLOYMENT_CONFIG
    }
    
    # 从环境特定配置文件加载额外配置
    env_config_path = BASE_DIR / "config" / f"{ENV}.yaml"
    if env_config_path.exists():
        env_config = load_config_from_file(str(env_config_path))
        # 递归更新配置
        def update_config(base_config, new_config):
            for key, value in new_config.items():
                if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                    update_config(base_config[key], value)
                else:
                    base_config[key] = value
        
        update_config(config, env_config)
    
    return config

# 初始化目录
def init_directories():
    """初始化必要的目录"""
    directories = [
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "processed", 
        BASE_DIR / "data" / "annotated",
        BASE_DIR / "data" / "corpus",
        BASE_DIR / "data" / "output",
        BASE_DIR / "models",
        BASE_DIR / "logs",
        BASE_DIR / "checkpoints",
        BASE_DIR / "cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 在模块导入时初始化目录
init_directories()

# 导出主要配置对象
CONFIG = get_config()