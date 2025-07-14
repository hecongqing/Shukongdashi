#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块
用于管理项目的各种配置参数
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DatabaseConfig:
    """数据库配置"""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "password"
    mysql_database: str = "equipment_fault"

@dataclass
class ModelConfig:
    """模型配置"""
    bert_model_name: str = "bert-base-chinese"
    llm_model_name: str = "THUDM/chatglm2-6b"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 5

@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    timeout: int = 30

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "../data"
    logs_dir: str = "../logs"
    models_dir: str = "../models"
    temp_dir: str = "../temp"
    max_file_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class Config:
    """主配置类"""
    # 环境配置
    environment: str = "development"
    debug: bool = True
    
    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # 自定义配置
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        self._create_directories()
        
        # 加载环境变量
        self._load_environment_variables()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data.data_dir,
            self.data.logs_dir,
            self.data.models_dir,
            self.data.temp_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _load_environment_variables(self):
        """加载环境变量"""
        # 数据库配置
        if os.getenv("NEO4J_URI"):
            self.database.neo4j_uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            self.database.neo4j_user = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            self.database.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # 模型配置
        if os.getenv("MODEL_NAME"):
            self.model.llm_model_name = os.getenv("MODEL_NAME")
        if os.getenv("DEVICE"):
            self.model.device = os.getenv("DEVICE")
        
        # API配置
        if os.getenv("HOST"):
            self.api.host = os.getenv("HOST")
        if os.getenv("PORT"):
            self.api.port = int(os.getenv("PORT"))
        
        # 环境配置
        if os.getenv("ENVIRONMENT"):
            self.environment = os.getenv("ENVIRONMENT")
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG").lower() == "true"
    
    def load_from_file(self, config_file: str):
        """从文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        if config_path.suffix.lower() == '.json':
            self._load_from_json(config_path)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            self._load_from_yaml(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def _load_from_json(self, config_path: Path):
        """从JSON文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self._update_config(config_data)
    
    def _load_from_yaml(self, config_path: Path):
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        self._update_config(config_data)
    
    def _update_config(self, config_data: Dict[str, Any]):
        """更新配置"""
        # 更新数据库配置
        if 'database' in config_data:
            db_config = config_data['database']
            for key, value in db_config.items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        # 更新模型配置
        if 'model' in config_data:
            model_config = config_data['model']
            for key, value in model_config.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # 更新API配置
        if 'api' in config_data:
            api_config = config_data['api']
            for key, value in api_config.items():
                if hasattr(self.api, key):
                    setattr(self.api, key, value)
        
        # 更新数据配置
        if 'data' in config_data:
            data_config = config_data['data']
            for key, value in data_config.items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        # 更新自定义配置
        if 'custom' in config_data:
            self.custom.update(config_data['custom'])
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        config_path = Path(config_file)
        
        config_data = {
            'environment': self.environment,
            'debug': self.debug,
            'database': {
                'neo4j_uri': self.database.neo4j_uri,
                'neo4j_user': self.database.neo4j_user,
                'neo4j_password': self.database.neo4j_password,
                'mysql_host': self.database.mysql_host,
                'mysql_port': self.database.mysql_port,
                'mysql_user': self.database.mysql_user,
                'mysql_password': self.database.mysql_password,
                'mysql_database': self.database.mysql_database
            },
            'model': {
                'bert_model_name': self.model.bert_model_name,
                'llm_model_name': self.model.llm_model_name,
                'device': self.model.device,
                'max_length': self.model.max_length,
                'batch_size': self.model.batch_size,
                'learning_rate': self.model.learning_rate,
                'epochs': self.model.epochs
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'workers': self.api.workers,
                'timeout': self.api.timeout
            },
            'data': {
                'data_dir': self.data.data_dir,
                'logs_dir': self.data.logs_dir,
                'models_dir': self.data.models_dir,
                'temp_dir': self.data.temp_dir,
                'max_file_size': self.data.max_file_size
            },
            'custom': self.custom
        }
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        target = self
        
        for k in keys[:-1]:
            if hasattr(target, k):
                target = getattr(target, k)
            elif isinstance(target, dict):
                if k not in target:
                    target[k] = {}
                target = target[k]
            else:
                raise ValueError(f"无法设置配置项: {key}")
        
        last_key = keys[-1]
        if hasattr(target, last_key):
            setattr(target, last_key, value)
        elif isinstance(target, dict):
            target[last_key] = value
        else:
            raise ValueError(f"无法设置配置项: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'database': {
                'neo4j_uri': self.database.neo4j_uri,
                'neo4j_user': self.database.neo4j_user,
                'neo4j_password': self.database.neo4j_password,
                'mysql_host': self.database.mysql_host,
                'mysql_port': self.database.mysql_port,
                'mysql_user': self.database.mysql_user,
                'mysql_password': self.database.mysql_password,
                'mysql_database': self.database.mysql_database
            },
            'model': {
                'bert_model_name': self.model.bert_model_name,
                'llm_model_name': self.model.llm_model_name,
                'device': self.model.device,
                'max_length': self.model.max_length,
                'batch_size': self.model.batch_size,
                'learning_rate': self.model.learning_rate,
                'epochs': self.model.epochs
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'workers': self.api.workers,
                'timeout': self.api.timeout
            },
            'data': {
                'data_dir': self.data.data_dir,
                'logs_dir': self.data.logs_dir,
                'models_dir': self.data.models_dir,
                'temp_dir': self.data.temp_dir,
                'max_file_size': self.data.max_file_size
            },
            'custom': self.custom
        }

def create_default_config(config_file: str = "config.json"):
    """创建默认配置文件"""
    config = Config()
    config.save_to_file(config_file)
    return config

def load_config(config_file: str = None) -> Config:
    """加载配置"""
    config = Config()
    
    if config_file and Path(config_file).exists():
        config.load_from_file(config_file)
    
    return config

# 全局配置实例
_config = None

def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def set_config(config: Config):
    """设置全局配置实例"""
    global _config
    _config = config