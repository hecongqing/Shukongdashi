#!/usr/bin/env python3
"""
设备故障知识图谱 - 模型训练主脚本
用于训练实体抽取和关系抽取模型
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.entity_extraction.train_ner import train_ner_model
from src.relation_extraction.train_relation import train_relation_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """创建示例训练数据"""
    sample_data = [
        {
            "ID": "AT0001",
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "spo_list": [
                {"h": {"name": "发动机盖", "pos": [14, 18]}, "t": {"name": "抖动", "pos": [24, 26]}, "relation": "部件故障"},
                {"h": {"name": "发动机盖锁", "pos": [46, 51]}, "t": {"name": "松旷", "pos": [58, 60]}, "relation": "部件故障"},
                {"h": {"name": "发动机盖铰链", "pos": [52, 58]}, "t": {"name": "松旷", "pos": [58, 60]}, "relation": "部件故障"}
            ]
        },
        {
            "ID": "AT0002",
            "text": "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸，因此将影响发动机的正常运转，使得发动机出现加速不良的症状。",
            "spo_list": [
                {"h": {"name": "燃油泵", "pos": [0, 3]}, "t": {"name": "损坏", "pos": [15, 17]}, "relation": "部件故障"},
                {"h": {"name": "发动机", "pos": [25, 28]}, "t": {"name": "加速不良", "pos": [45, 49]}, "relation": "部件故障"},
                {"h": {"name": "燃油", "pos": [18, 20]}, "t": {"name": "不能正常喷入", "pos": [21, 27]}, "relation": "性能故障"}
            ]
        },
        {
            "ID": "AT0003",
            "text": "使用漏电测试仪检测电流异常，发现保护器动作频繁，需要更换零序互感器。",
            "spo_list": [
                {"h": {"name": "漏电测试仪", "pos": [1, 6]}, "t": {"name": "电流", "pos": [8, 10]}, "relation": "检测工具"},
                {"h": {"name": "保护器", "pos": [15, 18]}, "t": {"name": "动作频繁", "pos": [19, 23]}, "relation": "部件故障"},
                {"h": {"name": "零序互感器", "pos": [28, 33]}, "t": {"name": "更换", "pos": [26, 28]}, "relation": "部件故障"}
            ]
        },
        {
            "ID": "AT0004",
            "text": "减振器活塞与缸体发卡，工作阻力过大诊断排除。",
            "spo_list": [
                {"h": {"name": "减振器活塞", "pos": [0, 5]}, "t": {"name": "发卡", "pos": [6, 8]}, "relation": "部件故障"},
                {"h": {"name": "缸体", "pos": [6, 8]}, "t": {"name": "发卡", "pos": [6, 8]}, "relation": "部件故障"},
                {"h": {"name": "工作阻力", "pos": [10, 13]}, "t": {"name": "过大", "pos": [13, 15]}, "relation": "性能故障"}
            ]
        }
    ]
    
    return sample_data

def train_models(data_path: str = None, output_dir: str = "./models"):
    """训练实体抽取和关系抽取模型"""
    
    # 如果没有提供数据路径，创建示例数据
    if data_path is None or not os.path.exists(data_path):
        logger.info("创建示例训练数据...")
        sample_data = create_sample_data()
        data_path = "sample_training_data.json"
        
        with open(data_path, 'w', encoding='utf-8') as f:
            for sample in sample_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"示例数据已保存到: {data_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练实体抽取模型
    logger.info("开始训练实体抽取模型...")
    ner_output_dir = os.path.join(output_dir, "ner_models")
    try:
        train_ner_model(data_path, ner_output_dir)
        logger.info("实体抽取模型训练完成!")
    except Exception as e:
        logger.error(f"实体抽取模型训练失败: {e}")
        return False
    
    # 训练关系抽取模型
    logger.info("开始训练关系抽取模型...")
    relation_output_dir = os.path.join(output_dir, "relation_models")
    try:
        train_relation_model(data_path, relation_output_dir)
        logger.info("关系抽取模型训练完成!")
    except Exception as e:
        logger.error(f"关系抽取模型训练失败: {e}")
        return False
    
    logger.info("所有模型训练完成!")
    return True

def main():
    parser = argparse.ArgumentParser(description="训练实体抽取和关系抽取模型")
    parser.add_argument("--data_path", type=str, help="训练数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./models", help="模型输出目录")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    
    args = parser.parse_args()
    
    if args.create_sample:
        # 只创建示例数据
        sample_data = create_sample_data()
        output_file = "sample_training_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in sample_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"示例数据已保存到: {output_file}")
        return
    
    # 训练模型
    success = train_models(args.data_path, args.output_dir)
    
    if success:
        logger.info("模型训练成功完成!")
        logger.info(f"模型保存在: {args.output_dir}")
    else:
        logger.error("模型训练失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()