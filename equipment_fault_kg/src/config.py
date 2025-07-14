"""
设备故障知识图谱 - 配置文件
包含模型训练和部署的各种参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据相关配置
DATA_CONFIG = {
    # 实体类型
    'entity_types': {
        "部件单元": "COMPONENT",
        "性能表征": "PERFORMANCE", 
        "故障状态": "FAULT_STATE",
        "检测工具": "DETECTION_TOOL"
    },
    
    # 关系类型
    'relation_types': {
        "部件故障": "COMPONENT_FAULT",
        "性能故障": "PERFORMANCE_FAULT", 
        "检测工具": "DETECTION_TOOL_REL",
        "组成": "COMPOSITION"
    },
    
    # 数据目录
    'data_dir': PROJECT_ROOT / "data",
    'train_data_path': PROJECT_ROOT / "data" / "train_data.json",
    'test_data_path': PROJECT_ROOT / "data" / "test_data.json",
    
    # 处理后的数据目录
    'processed_data_dir': PROJECT_ROOT / "data" / "processed",
}

# 模型相关配置
MODEL_CONFIG = {
    # BERT模型配置
    'bert_model_name': 'bert-base-chinese',
    'max_length': 512,
    'dropout': 0.1,
    
    # 训练配置
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'test_size': 0.2,
    'random_seed': 42,
    
    # 模型保存目录
    'model_dir': PROJECT_ROOT / "models",
    'ner_model_dir': PROJECT_ROOT / "models" / "ner_models",
    'relation_model_dir': PROJECT_ROOT / "models" / "relation_models",
    
    # 模型文件名
    'ner_model_name': 'best_ner_model.pth',
    'relation_model_name': 'best_relation_model.pth',
}

# 部署相关配置
DEPLOY_CONFIG = {
    # API服务配置
    'host': '0.0.0.0',
    'port': 5000,
    'ner_port': 5001,
    'relation_port': 5002,
    'debug': False,
    
    # 模型路径
    'ner_model_path': MODEL_CONFIG['ner_model_dir'] / MODEL_CONFIG['ner_model_name'],
    'relation_model_path': MODEL_CONFIG['relation_model_dir'] / MODEL_CONFIG['relation_model_name'],
    
    # 日志配置
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# 评估相关配置
EVAL_CONFIG = {
    # 评估指标
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'average': 'weighted',
    
    # 阈值配置
    'confidence_threshold': 0.5,
    'relation_confidence_threshold': 0.6,
}

# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    directories = [
        DATA_CONFIG['data_dir'],
        DATA_CONFIG['processed_data_dir'],
        MODEL_CONFIG['model_dir'],
        MODEL_CONFIG['ner_model_dir'],
        MODEL_CONFIG['relation_model_dir'],
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# 获取配置
def get_config():
    """获取完整配置"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'deploy': DEPLOY_CONFIG,
        'eval': EVAL_CONFIG,
    }

# 验证配置
def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查BERT模型名称
    if not MODEL_CONFIG['bert_model_name']:
        errors.append("BERT模型名称不能为空")
    
    # 检查数值参数
    if MODEL_CONFIG['batch_size'] <= 0:
        errors.append("批次大小必须大于0")
    
    if MODEL_CONFIG['epochs'] <= 0:
        errors.append("训练轮数必须大于0")
    
    if MODEL_CONFIG['learning_rate'] <= 0:
        errors.append("学习率必须大于0")
    
    if DEPLOY_CONFIG['port'] <= 0 or DEPLOY_CONFIG['port'] > 65535:
        errors.append("端口号必须在1-65535之间")
    
    if errors:
        raise ValueError(f"配置验证失败: {'; '.join(errors)}")
    
    return True

# 初始化配置
def init_config():
    """初始化配置"""
    create_directories()
    validate_config()
    return get_config()

if __name__ == "__main__":
    # 测试配置
    try:
        config = init_config()
        print("配置初始化成功!")
        print(f"项目根目录: {PROJECT_ROOT}")
        print(f"模型目录: {MODEL_CONFIG['model_dir']}")
        print(f"数据目录: {DATA_CONFIG['data_dir']}")
    except Exception as e:
        print(f"配置初始化失败: {e}")