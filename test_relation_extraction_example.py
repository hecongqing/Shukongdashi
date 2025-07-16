#!/usr/bin/env python3
"""
关系抽取器测试示例

演示如何使用RelationExtractor和JointExtractor进行关系抽取
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'equipment_fault_kg', 'src'))

def test_relation_extractor():
    """测试关系抽取器 - 需要预训练模型"""
    print("=== 关系抽取器测试 ===")
    
    try:
        # 注意：这里需要实际的模型路径
        # model_path = "equipment_fault_kg/models/relation_model.pth"
        # from relation_extraction.deploy_relation import RelationExtractor
        # relation_extractor = RelationExtractor(model_path)
        
        # 模拟输入数据
        text = "数控机床主轴故障导致加工精度下降，需要更换轴承解决。"
        entities = [
            {'name': '数控机床', 'type': 'COMPONENT'},
            {'name': '主轴', 'type': 'COMPONENT'},
            {'name': '加工精度', 'type': 'PERFORMANCE'},
            {'name': '轴承', 'type': 'COMPONENT'}
        ]
        
        print(f"输入文本: {text}")
        print(f"输入实体: {[e['name'] for e in entities]}")
        
        # 由于没有实际模型，这里展示期望的输出格式
        print("\n期望的输出格式:")
        expected_relations = [
            {
                'head_entity': '主轴',
                'tail_entity': '加工精度',
                'relation_type': '部件故障',
                'confidence': 0.85
            },
            {
                'head_entity': '主轴',
                'tail_entity': '轴承',
                'relation_type': '组成',
                'confidence': 0.78
            }
        ]
        
        for i, relation in enumerate(expected_relations, 1):
            print(f"{i}. {relation['head_entity']} --{relation['relation_type']}--> {relation['tail_entity']}")
            print(f"   置信度: {relation['confidence']:.2f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("注意：需要先训练或提供预训练的关系抽取模型")

def test_joint_extractor():
    """测试联合抽取器 - 需要预训练模型"""
    print("\n=== 联合抽取器测试 ===")
    
    try:
        # 注意：这里需要实际的模型路径
        # ner_model_path = "equipment_fault_kg/models/ner_model.pth"
        # relation_model_path = "equipment_fault_kg/models/relation_model.pth"
        # from relation_extraction.deploy_relation import JointExtractor
        # joint_extractor = JointExtractor(ner_model_path, relation_model_path)
        
        text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
        print(f"输入文本: {text}")
        
        # 期望的输出格式
        print("\n期望的输出格式:")
        expected_result = {
            'text': text,
            'entities': [
                {'name': '伺服电机', 'type': 'COMPONENT'},
                {'name': '维修人员', 'type': 'PERSON'},
                {'name': '万用表', 'type': 'DETECTION_TOOL'},
                {'name': '电路', 'type': 'COMPONENT'}
            ],
            'relations': [
                {
                    'head_entity': '维修人员',
                    'tail_entity': '万用表',
                    'relation_type': '检测工具',
                    'confidence': 0.92
                },
                {
                    'head_entity': '万用表',
                    'tail_entity': '电路',
                    'relation_type': '检测工具',
                    'confidence': 0.88
                }
            ],
            'spo_list': [
                {
                    'h': {'name': '维修人员'},
                    'relation': '检测工具',
                    't': {'name': '万用表'}
                },
                {
                    'h': {'name': '万用表'},
                    'relation': '检测工具',
                    't': {'name': '电路'}
                }
            ]
        }
        
        print("抽取的实体:")
        for entity in expected_result['entities']:
            print(f"  - {entity['name']} [{entity['type']}]")
        
        print("\n抽取的关系:")
        for relation in expected_result['relations']:
            print(f"  - {relation['head_entity']} --{relation['relation_type']}--> {relation['tail_entity']}")
            print(f"    置信度: {relation['confidence']:.2f}")
        
        print("\nSPO三元组:")
        for spo in expected_result['spo_list']:
            print(f"  - ({spo['h']['name']}, {spo['relation']}, {spo['t']['name']})")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("注意：需要先训练或提供预训练的NER和关系抽取模型")

def demo_usage_pattern():
    """演示使用模式"""
    print("\n=== 使用模式演示 ===")
    
    print("1. 单独使用关系抽取器（需要预先提供实体）:")
    print("""
    from relation_extraction.deploy_relation import RelationExtractor
    
    relation_extractor = RelationExtractor("path/to/model.pth")
    relations = relation_extractor.extract_relations(text, entities)
    """)
    
    print("2. 使用联合抽取器（自动抽取实体和关系）:")
    print("""
    from relation_extraction.deploy_relation import JointExtractor
    
    joint_extractor = JointExtractor("ner_model.pth", "relation_model.pth")
    result = joint_extractor.extract_spo(text)
    """)
    
    print("3. 批量处理:")
    print("""
    # 关系抽取器批量处理
    results = relation_extractor.extract_relations_batch(texts, entities_list)
    
    # 联合抽取器批量处理
    results = joint_extractor.extract_spo_batch(texts)
    """)

def show_model_requirements():
    """显示模型要求"""
    print("\n=== 模型要求 ===")
    print("要使用这些抽取器，你需要:")
    print("1. 训练好的NER模型（.pth文件）")
    print("2. 训练好的关系抽取模型（.pth文件）")
    print()
    print("模型训练:")
    print("- NER模型训练: equipment_fault_kg/src/entity_extraction/train_ner.py")
    print("- 关系模型训练: equipment_fault_kg/src/relation_extraction/train_relation.py")
    print()
    print("数据准备:")
    print("- 需要标注好的实体和关系数据")
    print("- 数据格式参考: equipment_fault_kg/data/ 目录")

def main():
    """主函数"""
    print("关系抽取使用示例")
    print("=" * 60)
    
    # 运行测试
    test_relation_extractor()
    test_joint_extractor()
    demo_usage_pattern()
    show_model_requirements()
    
    print("\n" + "=" * 60)
    print("注意事项:")
    print("1. 确保模型文件路径正确")
    print("2. 实体格式必须包含 'name' 和 'type' 字段")
    print("3. 支持的关系类型: 部件故障、性能故障、检测工具、组成")
    print("4. 可以调整置信度阈值来控制结果质量")
    print("5. 如有GPU，会自动使用GPU加速")

if __name__ == "__main__":
    main()