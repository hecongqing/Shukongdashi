"""
故障信息抽取系统使用示例

演示如何使用实体抽取和关系抽取功能
"""

import json
import sys
import os
from typing import List, Dict, Any

# 添加路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'entity_extraction'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'relation_extraction'))

# 示例数据
SAMPLE_DATA = [
    {
        "ID": "EXAMPLE_001",
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
        "ID": "EXAMPLE_002", 
        "text": "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸，因此将影响发动机的正常运转，使得发动机出现加速不良的症状。",
        "spo_list": [
            {
                "h": {"name": "燃油泵", "pos": [0, 3]},
                "t": {"name": "损坏", "pos": [23, 25]},
                "relation": "部件故障"
            },
            {
                "h": {"name": "发动机", "pos": [41, 44]},
                "t": {"name": "加速不良", "pos": [65, 69]},
                "relation": "部件故障"
            }
        ]
    },
    {
        "ID": "EXAMPLE_003",
        "text": "减振器活塞与缸体发卡，工作阻力过大诊断排除。",
        "spo_list": [
            {
                "h": {"name": "减振器活塞", "pos": [0, 5]},
                "t": {"name": "发卡", "pos": [8, 10]},
                "relation": "部件故障"
            },
            {
                "h": {"name": "工作阻力", "pos": [12, 16]},
                "t": {"name": "过大", "pos": [16, 18]},
                "relation": "性能故障"
            }
        ]
    }
]


def create_sample_data_file(filename: str = "sample_data.jsonl"):
    """创建示例数据文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"示例数据已保存到: {filename}")
    return filename


def demo_ner_training():
    """演示NER模型训练"""
    print("=== NER模型训练演示 ===")
    
    # 创建示例数据
    data_file = create_sample_data_file("train_sample.jsonl")
    
    try:
        from entity_extraction.fault_ner_model import FaultNERTrainer
        
        # 配置
        config = {
            "model_name": "bert-base-chinese",
            "max_length": 512,
            "batch_size": 2,  # 小批次用于演示
            "epochs": 1,       # 少轮训练用于演示
            "learning_rate": 2e-5,
            "dropout": 0.1,
            "output_dir": "./demo_ner_model"
        }
        
        print("初始化NER训练器...")
        trainer = FaultNERTrainer(config)
        
        print("开始训练...")
        trainer.train(data_file)
        
        print("训练完成！模型已保存到: ./demo_ner_model")
        
        # 测试预测
        print("\n测试预测:")
        test_text = "发动机盖出现了抖动故障"
        entities = trainer.predict(test_text)
        
        print(f"输入文本: {test_text}")
        print(f"提取的实体:")
        for entity in entities:
            print(f"  - {entity['text']} ({entity['type']}) [{entity['start']}:{entity['end']}]")
            
    except ImportError as e:
        print(f"无法导入NER模块: {e}")
        print("请确保相关依赖已安装")
    except Exception as e:
        print(f"NER训练演示失败: {e}")


def demo_relation_training():
    """演示关系抽取模型训练"""
    print("\n=== 关系抽取模型训练演示 ===")
    
    # 创建示例数据
    data_file = create_sample_data_file("train_sample.jsonl")
    
    try:
        from relation_extraction.fault_relation_model import FaultRelationTrainer
        
        # 配置
        config = {
            "model_name": "bert-base-chinese",
            "max_length": 512,
            "batch_size": 2,  # 小批次用于演示
            "epochs": 1,       # 少轮训练用于演示
            "learning_rate": 2e-5,
            "dropout": 0.1,
            "output_dir": "./demo_relation_model"
        }
        
        print("初始化关系抽取训练器...")
        trainer = FaultRelationTrainer(config)
        
        print("开始训练...")
        trainer.train(data_file)
        
        print("训练完成！模型已保存到: ./demo_relation_model")
        
        # 测试预测
        print("\n测试预测:")
        test_text = "发动机盖出现了抖动故障"
        test_entities = [
            {"text": "发动机盖", "type": "部件单元", "start": 0, "end": 4},
            {"text": "抖动", "type": "故障状态", "start": 7, "end": 9}
        ]
        
        relations = trainer.predict(test_text, test_entities)
        
        print(f"输入文本: {test_text}")
        print(f"输入实体: {test_entities}")
        print(f"提取的关系:")
        for relation in relations:
            print(f"  - {relation['h']['name']} -> {relation['t']['name']} ({relation['relation']}) [置信度: {relation['confidence']:.3f}]")
            
    except ImportError as e:
        print(f"无法导入关系抽取模块: {e}")
        print("请确保相关依赖已安装")
    except Exception as e:
        print(f"关系抽取训练演示失败: {e}")


def demo_pipeline():
    """演示完整流水线"""
    print("\n=== 完整流水线演示 ===")
    
    # 检查模型是否存在
    ner_model_dir = "./demo_ner_model"
    relation_model_dir = "./demo_relation_model"
    
    if not (os.path.exists(ner_model_dir) and os.path.exists(relation_model_dir)):
        print("请先运行NER和关系抽取训练，生成模型文件")
        return
    
    try:
        from pipeline import FaultKGPipeline
        
        print("初始化流水线...")
        pipeline = FaultKGPipeline(ner_model_dir, relation_model_dir)
        
        # 测试文本
        test_texts = [
            "燃油泵损坏导致发动机无法正常启动",
            "温度传感器检测到发动机温度过高",
            "制动器活塞出现卡滞现象"
        ]
        
        print("\n批量处理测试:")
        for i, text in enumerate(test_texts, 1):
            print(f"\n文本 {i}: {text}")
            result = pipeline.extract(text)
            
            print(f"实体数量: {result['entity_count']}")
            for entity in result['entities']:
                print(f"  - {entity['text']} ({entity['type']})")
            
            print(f"关系数量: {result['relation_count']}")
            for relation in result['relations']:
                print(f"  - {relation['h']['name']} -> {relation['t']['name']} ({relation['relation']})")
        
        # 生成SPO格式
        print("\n=== SPO格式输出 ===")
        test_text = test_texts[0]
        result = pipeline.extract(test_text)
        result['ID'] = 'DEMO_001'
        spo_result = pipeline.convert_to_spo_format(result)
        
        print("SPO格式结果:")
        print(json.dumps(spo_result, ensure_ascii=False, indent=2))
        
    except ImportError as e:
        print(f"无法导入流水线模块: {e}")
        print("请确保相关依赖已安装")
    except Exception as e:
        print(f"流水线演示失败: {e}")


def demo_data_analysis():
    """演示数据分析"""
    print("\n=== 数据分析演示 ===")
    
    # 分析示例数据
    entity_types = {}
    relation_types = {}
    
    for item in SAMPLE_DATA:
        text = item['text']
        spo_list = item['spo_list']
        
        print(f"\n文本: {text}")
        print(f"包含 {len(spo_list)} 个关系三元组")
        
        for spo in spo_list:
            relation = spo['relation']
            h_name = spo['h']['name']
            t_name = spo['t']['name']
            
            # 统计关系类型
            if relation not in relation_types:
                relation_types[relation] = 0
            relation_types[relation] += 1
            
            print(f"  {h_name} -> {t_name} ({relation})")
    
    print(f"\n=== 统计信息 ===")
    print(f"关系类型分布:")
    for rel_type, count in relation_types.items():
        print(f"  {rel_type}: {count} 次")


def main():
    """主函数"""
    print("故障信息抽取系统演示")
    print("=" * 50)
    
    # 创建示例数据
    create_sample_data_file()
    
    # 数据分析
    demo_data_analysis()
    
    # 询问用户是否要进行训练演示
    print("\n是否要进行模型训练演示？(需要较长时间)")
    choice = input("输入 'y' 开始训练演示，其他键跳过: ")
    
    if choice.lower() == 'y':
        # NER训练演示
        demo_ner_training()
        
        # 关系抽取训练演示  
        demo_relation_training()
        
        # 完整流水线演示
        demo_pipeline()
    else:
        print("跳过训练演示")
    
    print("\n演示完成！")
    print("\n快速开始指南:")
    print("1. 准备训练数据 (JSON Lines格式)")
    print("2. 训练NER模型: python entity_extraction/train_ner.py --train_data data.jsonl --output_dir ner_model")
    print("3. 训练关系模型: python relation_extraction/train_relation.py --train_data data.jsonl --output_dir relation_model")  
    print("4. 启动API服务: python pipeline.py --ner_model ner_model --relation_model relation_model")


if __name__ == "__main__":
    main()