#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取系统演示脚本
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.entity_extraction.data_processor import DataProcessor
from src.deployment.pipeline import InformationExtractionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_data():
    """创建演示数据"""
    demo_data = [
        {
            "ID": "DEMO001",
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
            "ID": "DEMO002",
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
        },
        {
            "ID": "DEMO003",
            "text": "减振器活塞与缸体发卡，工作阻力过大诊断排除。当液面变低时，需要检查燃油泵的工作状态。",
            "spo_list": [
                {
                    "h": {"name": "减振器活塞", "pos": [0, 5]},
                    "t": {"name": "发卡", "pos": [6, 8]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "液面", "pos": [20, 22]},
                    "t": {"name": "变低", "pos": [23, 25]},
                    "relation": "性能故障"
                },
                {
                    "h": {"name": "燃油泵", "pos": [32, 35]},
                    "t": {"name": "工作状态", "pos": [39, 43]},
                    "relation": "检测工具"
                }
            ]
        }
    ]
    
    # 保存演示数据
    with open("data/demo_train.json", "w", encoding="utf-8") as f:
        for item in demo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info("演示数据已创建: data/demo_train.json")
    return demo_data

def run_demo():
    """运行演示"""
    logger.info("=" * 60)
    logger.info("设备故障知识图谱信息抽取系统演示")
    logger.info("=" * 60)
    
    # 创建必要的目录
    directories = ["data", "models/ner_model", "models/re_model", "results"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 1. 创建演示数据
    logger.info("\n1. 创建演示数据...")
    demo_data = create_demo_data()
    
    # 2. 数据处理演示
    logger.info("\n2. 数据处理演示...")
    processor = DataProcessor()
    
    # 提取实体
    print("\n   实体提取结果:")
    for i, sample in enumerate(demo_data):
        entities = processor.extract_entities_from_spo(sample['spo_list'])
        print(f"   样本 {i+1}: {len(entities)} 个实体")
        for entity in entities:
            print(f"     - {entity.name} ({entity.type}) 位置: [{entity.start}, {entity.end}]")
    
    # 转换为训练格式
    ner_data = processor.convert_to_ner_format(demo_data)
    re_data = processor.convert_to_re_format(demo_data)
    
    print(f"\n   数据转换结果:")
    print(f"   - NER样本: {len(ner_data)} 个")
    print(f"   - RE样本: {len(re_data)} 个")
    
    # 3. 模型训练演示（如果模型不存在）
    logger.info("\n3. 模型训练演示...")
    
    ner_model_path = "models/ner_model"
    re_model_path = "models/re_model"
    
    if not os.path.exists(os.path.join(ner_model_path, "ner_model.pth")):
        logger.info("   NER模型不存在，开始训练...")
        try:
            from src.entity_extraction.trainer import NERTrainer
            
            # 划分数据
            train_data, val_data, test_data = processor.split_data(ner_data)
            
            # 训练NER模型
            trainer = NERTrainer()
            best_f1 = trainer.train(
                train_data=train_data,
                val_data=val_data,
                output_dir=ner_model_path,
                batch_size=8,  # 使用较小的batch size
                epochs=3       # 使用较少的epochs用于演示
            )
            logger.info(f"   NER模型训练完成，最佳F1: {best_f1:.4f}")
        except Exception as e:
            logger.warning(f"   NER模型训练失败: {e}")
    else:
        logger.info("   NER模型已存在")
    
    if not os.path.exists(os.path.join(re_model_path, "re_model.pth")):
        logger.info("   RE模型不存在，开始训练...")
        try:
            from src.relation_extraction.trainer import RETrainer
            
            # 划分数据
            train_data, val_data, test_data = processor.split_data(re_data)
            
            # 训练RE模型
            trainer = RETrainer()
            best_f1 = trainer.train(
                train_data=train_data,
                val_data=val_data,
                output_dir=re_model_path,
                batch_size=8,  # 使用较小的batch size
                epochs=3       # 使用较少的epochs用于演示
            )
            logger.info(f"   RE模型训练完成，最佳F1: {best_f1:.4f}")
        except Exception as e:
            logger.warning(f"   RE模型训练失败: {e}")
    else:
        logger.info("   RE模型已存在")
    
    # 4. 模型推理演示
    logger.info("\n4. 模型推理演示...")
    
    if (os.path.exists(os.path.join(ner_model_path, "ner_model.pth")) and 
        os.path.exists(os.path.join(re_model_path, "re_model.pth"))):
        
        try:
            # 初始化管道
            pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
            
            # 测试文本
            test_texts = [
                "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
                "燃油泵损坏导致发动机无法启动。",
                "减振器活塞与缸体发卡，工作阻力过大。",
                "当液面变低时，需要检查燃油泵的工作状态。",
                "使用漏电测试仪检测电流异常情况。"
            ]
            
            print("\n   推理结果:")
            for i, text in enumerate(test_texts):
                print(f"\n   文本 {i+1}: {text}")
                
                # 执行抽取
                result = pipeline.extract(text)
                
                # 显示实体
                if result['entities']:
                    print(f"   实体 ({len(result['entities'])}):")
                    for entity in result['entities']:
                        print(f"     - {entity['text']} ({entity['type']}) 位置: [{entity['start']}, {entity['end']}]")
                else:
                    print("   实体: 无")
                
                # 显示关系
                if result['relations']:
                    print(f"   关系 ({len(result['relations'])}):")
                    for relation in result['relations']:
                        print(f"     - {relation['head']['text']} --{relation['relation']}--> {relation['tail']['text']}")
                else:
                    print("   关系: 无")
        
        except Exception as e:
            logger.error(f"   模型推理失败: {e}")
    else:
        logger.warning("   模型文件不存在，跳过推理演示")
    
    # 5. 总结
    logger.info("\n" + "=" * 60)
    logger.info("演示完成！")
    logger.info("\n系统功能:")
    logger.info("✓ 数据处理: 支持JSON格式的训练数据处理")
    logger.info("✓ 实体抽取: 支持4种实体类型的识别")
    logger.info("✓ 关系抽取: 支持4种关系类型的识别")
    logger.info("✓ 模型训练: 基于BERT的深度学习模型")
    logger.info("✓ 端到端推理: 完整的文本到三元组抽取")
    logger.info("✓ API服务: RESTful API接口")
    
    logger.info("\n下一步:")
    logger.info("1. 准备更多训练数据以提高模型性能")
    logger.info("2. 调整训练参数以获得更好的效果")
    logger.info("3. 部署到生产环境")
    logger.info("4. 集成到知识图谱构建流程中")

if __name__ == "__main__":
    run_demo()