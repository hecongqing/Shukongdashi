#!/usr/bin/env python3
"""
设备故障知识图谱 - 使用示例
展示如何使用实体抽取和关系抽取系统
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def example_entity_extraction():
    """实体抽取示例"""
    print("=" * 60)
    print("实体抽取示例")
    print("=" * 60)
    
    # 示例文本
    text = "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。"
    
    print(f"输入文本: {text}")
    print()
    
    # 这里需要先训练模型才能使用
    # 实际使用时，请先运行训练脚本
    print("注意: 需要先训练模型才能使用以下功能")
    print("运行命令: python train_models.py")
    print()
    
    # 模拟结果
    mock_entities = [
        {
            "name": "发动机盖",
            "type": "部件单元",
            "start_pos": 14,
            "end_pos": 18
        },
        {
            "name": "抖动",
            "type": "故障状态",
            "start_pos": 24,
            "end_pos": 26
        },
        {
            "name": "发动机盖锁",
            "type": "部件单元",
            "start_pos": 46,
            "end_pos": 51
        },
        {
            "name": "松旷",
            "type": "故障状态",
            "start_pos": 58,
            "end_pos": 60
        }
    ]
    
    print("抽取的实体:")
    for entity in mock_entities:
        print(f"  - {entity['name']} ({entity['type']}) 位置: {entity['start_pos']}-{entity['end_pos']}")
    print()

def example_relation_extraction():
    """关系抽取示例"""
    print("=" * 60)
    print("关系抽取示例")
    print("=" * 60)
    
    # 示例文本和实体
    text = "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。"
    entities = [
        {"name": "发动机盖", "type": "部件单元", "start_pos": 14, "end_pos": 18},
        {"name": "抖动", "type": "故障状态", "start_pos": 24, "end_pos": 26},
        {"name": "发动机盖锁", "type": "部件单元", "start_pos": 46, "end_pos": 51},
        {"name": "松旷", "type": "故障状态", "start_pos": 58, "end_pos": 60}
    ]
    
    print(f"输入文本: {text}")
    print(f"输入实体: {[e['name'] for e in entities]}")
    print()
    
    # 模拟结果
    mock_relations = [
        {
            "head_entity": "发动机盖",
            "tail_entity": "抖动",
            "relation_type": "部件故障",
            "confidence": 0.95
        },
        {
            "head_entity": "发动机盖锁",
            "tail_entity": "松旷",
            "relation_type": "部件故障",
            "confidence": 0.92
        }
    ]
    
    print("抽取的关系:")
    for relation in mock_relations:
        print(f"  - {relation['head_entity']} --[{relation['relation_type']}]--> {relation['tail_entity']} (置信度: {relation['confidence']:.2f})")
    print()

def example_joint_extraction():
    """联合抽取示例"""
    print("=" * 60)
    print("联合抽取示例 (SPO三元组)")
    print("=" * 60)
    
    # 示例文本
    text = "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。"
    
    print(f"输入文本: {text}")
    print()
    
    # 模拟结果
    mock_result = {
        "text": text,
        "entities": [
            {"name": "发动机盖", "type": "部件单元", "start_pos": 14, "end_pos": 18},
            {"name": "抖动", "type": "故障状态", "start_pos": 24, "end_pos": 26},
            {"name": "发动机盖锁", "type": "部件单元", "start_pos": 46, "end_pos": 51},
            {"name": "松旷", "type": "故障状态", "start_pos": 58, "end_pos": 60}
        ],
        "relations": [
            {
                "head_entity": "发动机盖",
                "tail_entity": "抖动",
                "relation_type": "部件故障",
                "confidence": 0.95
            },
            {
                "head_entity": "发动机盖锁",
                "tail_entity": "松旷",
                "relation_type": "部件故障",
                "confidence": 0.92
            }
        ],
        "spo_list": [
            {
                "h": {"name": "发动机盖"},
                "t": {"name": "抖动"},
                "relation": "部件故障"
            },
            {
                "h": {"name": "发动机盖锁"},
                "t": {"name": "松旷"},
                "relation": "部件故障"
            }
        ]
    }
    
    print("抽取的SPO三元组:")
    for spo in mock_result['spo_list']:
        print(f"  - ({spo['h']['name']}, {spo['relation']}, {spo['t']['name']})")
    print()

def example_api_usage():
    """API使用示例"""
    print("=" * 60)
    print("API使用示例")
    print("=" * 60)
    
    print("1. 启动API服务:")
    print("   python deploy_models.py --ner_model_path ./models/ner_models/best_ner_model.pth --relation_model_path ./models/relation_models/best_relation_model.pth")
    print()
    
    print("2. 实体抽取API调用:")
    print("   curl -X POST http://localhost:5000/extract_entities \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{\"text\": \"故障现象:车速到100迈以上发动机盖后部随着车速抖动。\"}'")
    print()
    
    print("3. 关系抽取API调用:")
    print("   curl -X POST http://localhost:5000/extract_relations \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{")
    print("       \"text\": \"故障现象:车速到100迈以上发动机盖后部随着车速抖动。\",")
    print("       \"entities\": [")
    print("         {\"name\": \"发动机盖\", \"type\": \"部件单元\", \"start_pos\": 14, \"end_pos\": 18},")
    print("         {\"name\": \"抖动\", \"type\": \"故障状态\", \"start_pos\": 24, \"end_pos\": 26}")
    print("       ]")
    print("     }'")
    print()
    
    print("4. SPO三元组抽取API调用:")
    print("   curl -X POST http://localhost:5000/extract_spo \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{\"text\": \"故障现象:车速到100迈以上发动机盖后部随着车速抖动。\"}'")
    print()
    
    print("5. Web演示界面:")
    print("   访问 http://localhost:5000/demo")
    print()

def example_training():
    """训练示例"""
    print("=" * 60)
    print("模型训练示例")
    print("=" * 60)
    
    print("1. 使用示例数据训练:")
    print("   python train_models.py")
    print()
    
    print("2. 使用自定义数据训练:")
    print("   python train_models.py --data_path /path/to/your/data.json --output_dir ./models")
    print()
    
    print("3. 只创建示例数据:")
    print("   python train_models.py --create_sample")
    print()
    
    print("4. 训练数据格式:")
    print("   {")
    print("     \"ID\": \"AT0001\",")
    print("     \"text\": \"故障现象:车速到100迈以上发动机盖后部随着车速抖动。\",")
    print("     \"spo_list\": [")
    print("       {")
    print("         \"h\": {\"name\": \"发动机盖\", \"pos\": [14, 18]},")
    print("         \"t\": {\"name\": \"抖动\", \"pos\": [24, 26]},")
    print("         \"relation\": \"部件故障\"")
    print("       }")
    print("     ]")
    print("   }")
    print()

def example_testing():
    """测试示例"""
    print("=" * 60)
    print("系统测试示例")
    print("=" * 60)
    
    print("1. 运行系统测试:")
    print("   python test_system.py")
    print()
    
    print("2. 使用自定义模型路径测试:")
    print("   python test_system.py --ner_model_path ./models/ner_models/best_ner_model.pth --relation_model_path ./models/relation_models/best_relation_model.pth")
    print()
    
    print("3. 测试内容包括:")
    print("   - 实体抽取功能测试")
    print("   - 关系抽取功能测试")
    print("   - 联合抽取功能测试")
    print("   - 批量处理功能测试")
    print()

def main():
    """主函数"""
    print("设备故障知识图谱 - 实体抽取与关系抽取系统")
    print("使用示例和说明")
    print()
    
    # 显示各种示例
    example_entity_extraction()
    example_relation_extraction()
    example_joint_extraction()
    example_api_usage()
    example_training()
    example_testing()
    
    print("=" * 60)
    print("快速开始指南")
    print("=" * 60)
    print("1. 安装依赖: pip install torch transformers scikit-learn flask tqdm numpy")
    print("2. 训练模型: python train_models.py")
    print("3. 启动服务: python deploy_models.py --ner_model_path ./models/ner_models/best_ner_model.pth --relation_model_path ./models/relation_models/best_relation_model.pth")
    print("4. 测试系统: python test_system.py")
    print("5. 访问演示: http://localhost:5000/demo")
    print()
    print("详细文档请参考: README.md")

if __name__ == "__main__":
    main()