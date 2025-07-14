#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取模型训练脚本
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.entity_extraction.data_processor import DataProcessor
from src.entity_extraction.trainer import NERTrainer
from src.relation_extraction.trainer import RETrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """创建必要的目录"""
    directories = [
        "data",
        "models/ner_model",
        "models/re_model",
        "logs",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def train_ner_model(train_file: str, output_dir: str, config: dict):
    """训练NER模型"""
    logger.info("Starting NER model training...")
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        return False
    
    data = processor.load_data(train_file)
    logger.info(f"Loaded {len(data)} training samples")
    
    # 转换为NER格式
    ner_data = processor.convert_to_ner_format(data)
    logger.info(f"Converted to {len(ner_data)} NER samples")
    
    # 划分数据集
    train_data, val_data, test_data = processor.split_data(
        ner_data, 
        train_ratio=config.get('train_ratio', 0.8),
        val_ratio=config.get('val_ratio', 0.1)
    )
    
    # 保存处理后的数据
    processor.save_processed_data(train_data, "data/ner_train.json")
    processor.save_processed_data(val_data, "data/ner_val.json")
    processor.save_processed_data(test_data, "data/ner_test.json")
    
    # 初始化训练器
    trainer = NERTrainer(
        model_name=config.get('model_name', 'bert-base-chinese'),
        device=config.get('device', None)
    )
    
    # 训练模型
    try:
        best_f1 = trainer.train(
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
            batch_size=config.get('batch_size', 16),
            epochs=config.get('epochs', 10),
            learning_rate=config.get('learning_rate', 2e-5),
            warmup_steps=config.get('warmup_steps', 500)
        )
        
        logger.info(f"NER training completed. Best F1: {best_f1:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"NER training failed: {e}")
        return False

def train_re_model(train_file: str, output_dir: str, config: dict):
    """训练关系抽取模型"""
    logger.info("Starting RE model training...")
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        return False
    
    data = processor.load_data(train_file)
    logger.info(f"Loaded {len(data)} training samples")
    
    # 转换为关系抽取格式
    re_data = processor.convert_to_re_format(data)
    logger.info(f"Converted to {len(re_data)} RE samples")
    
    # 划分数据集
    train_data, val_data, test_data = processor.split_data(
        re_data,
        train_ratio=config.get('train_ratio', 0.8),
        val_ratio=config.get('val_ratio', 0.1)
    )
    
    # 保存处理后的数据
    processor.save_processed_data(train_data, "data/re_train.json")
    processor.save_processed_data(val_data, "data/re_val.json")
    processor.save_processed_data(test_data, "data/re_test.json")
    
    # 初始化训练器
    trainer = RETrainer(
        model_name=config.get('model_name', 'bert-base-chinese'),
        device=config.get('device', None)
    )
    
    # 训练模型
    try:
        best_f1 = trainer.train(
            train_data=train_data,
            val_data=val_data,
            output_dir=output_dir,
            batch_size=config.get('batch_size', 16),
            epochs=config.get('epochs', 10),
            learning_rate=config.get('learning_rate', 2e-5),
            warmup_steps=config.get('warmup_steps', 500)
        )
        
        logger.info(f"RE training completed. Best F1: {best_f1:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"RE training failed: {e}")
        return False

def create_sample_data():
    """创建示例训练数据"""
    sample_data = [
        {
            "ID": "AT0001",
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "spo_list": [
                {
                    "h": {"name": "发动机盖", "pos": [14, 18]},
                    "t": {"name": "抖动", "pos": [24, 26]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "发动机盖锁", "pos": [46, 51]},
                    "t": {"name": "松旷", "pos": [58, 60]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "发动机盖铰链", "pos": [52, 58]},
                    "t": {"name": "松旷", "pos": [58, 60]},
                    "relation": "部件故障"
                }
            ]
        },
        {
            "ID": "AT0002",
            "text": "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸，因此将影响发动机的正常运转，使得发动机出现加速不良的症状。",
            "spo_list": [
                {
                    "h": {"name": "燃油泵", "pos": [0, 3]},
                    "t": {"name": "损坏", "pos": [15, 17]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "发动机", "pos": [25, 28]},
                    "t": {"name": "加速不良", "pos": [45, 49]},
                    "relation": "部件故障"
                }
            ]
        }
    ]
    
    # 保存示例数据
    with open("data/sample_train.json", "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info("Created sample training data: data/sample_train.json")

def main():
    parser = argparse.ArgumentParser(description="Train entity and relation extraction models")
    parser.add_argument("--train_file", type=str, default="data/train.json",
                       help="Path to training data file")
    parser.add_argument("--ner_only", action="store_true",
                       help="Train only NER model")
    parser.add_argument("--re_only", action="store_true",
                       help="Train only RE model")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample training data")
    parser.add_argument("--config", type=str, default="config/training_config.json",
                       help="Path to training configuration file")
    
    args = parser.parse_args()
    
    # 创建目录
    setup_directories()
    
    # 创建示例数据
    if args.create_sample:
        create_sample_data()
        return
    
    # 加载配置
    config = {
        'model_name': 'bert-base-chinese',
        'device': None,
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'train_ratio': 0.8,
        'val_ratio': 0.1
    }
    
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
    
    logger.info(f"Training configuration: {config}")
    
    success = True
    
    # 训练NER模型
    if not args.re_only:
        ner_success = train_ner_model(
            train_file=args.train_file,
            output_dir="models/ner_model",
            config=config
        )
        success = success and ner_success
    
    # 训练RE模型
    if not args.ner_only:
        re_success = train_re_model(
            train_file=args.train_file,
            output_dir="models/re_model",
            config=config
        )
        success = success and re_success
    
    if success:
        logger.info("All training completed successfully!")
    else:
        logger.error("Some training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()