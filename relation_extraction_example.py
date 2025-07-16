#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关系抽取和联合抽取使用示例
演示如何使用 RelationExtractor 和 JointExtractor 类
"""

import asyncio
from typing import List, Dict
import json

# 假设的模型路径（实际使用时需要替换为真实的模型路径）
NER_MODEL_PATH = "models/ner_model"
RELATION_MODEL_PATH = "models/relation_model"

class RelationPredictor:
    """关系预测器 - 模拟实现"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"加载关系预测模型: {model_path}")
    
    def predict_all_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """预测所有关系"""
        # 这里是模拟实现，实际应该调用真实的模型
        relations = []
        
        # 模拟关系抽取逻辑
        entity_names = [entity.get('name', entity.get('text', '')) for entity in entities]
        
        # 简单的规则匹配示例
        if '故障' in text and '症状' in text:
            relations.append({
                'head_entity': '主轴故障',
                'tail_entity': '异常振动',
                'relation_type': '导致',
                'relation_confidence': 0.85
            })
        
        if '设备' in text and '维修' in text:
            relations.append({
                'head_entity': '数控机床',
                'tail_entity': '定期保养',
                'relation_type': '需要',
                'relation_confidence': 0.78
            })
        
        return relations

class RelationExtractor:
    """关系抽取器类 - 高级接口"""
    
    def __init__(self, model_path: str):
        self.predictor = RelationPredictor(model_path)
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """抽取文本中的关系"""
        relations = self.predictor.predict_all_relations(text, entities)
        
        # 转换格式
        result = []
        for relation in relations:
            result.append({
                'head_entity': relation['head_entity'],
                'tail_entity': relation['tail_entity'],
                'relation_type': relation['relation_type'],
                'confidence': relation['relation_confidence']
            })
        
        return result
    
    def extract_relations_batch(self, texts: List[str], entities_list: List[List[Dict]]) -> List[List[Dict]]:
        """批量抽取关系"""
        results = []
        for text, entities in zip(texts, entities_list):
            relations = self.extract_relations(text, entities)
            results.append(relations)
        return results
    
    def get_relations_by_type(self, text: str, entities: List[Dict], relation_type: str) -> List[Dict]:
        """根据关系类型获取关系"""
        all_relations = self.extract_relations(text, entities)
        return [rel for rel in all_relations if rel['relation_type'] == relation_type]

class EntityExtractor:
    """实体抽取器 - 模拟实现"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"加载实体抽取模型: {model_path}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取实体"""
        # 模拟实体抽取结果
        entities = []
        
        # 简单的规则匹配
        if '主轴' in text:
            entities.append({
                'name': '主轴',
                'type': 'EQUIPMENT',
                'start': text.find('主轴'),
                'end': text.find('主轴') + 2,
                'confidence': 0.9
            })
        
        if '故障' in text:
            entities.append({
                'name': '故障',
                'type': 'FAULT',
                'start': text.find('故障'),
                'end': text.find('故障') + 2,
                'confidence': 0.85
            })
        
        if '振动' in text:
            entities.append({
                'name': '振动',
                'type': 'SYMPTOM',
                'start': text.find('振动'),
                'end': text.find('振动') + 2,
                'confidence': 0.8
            })
        
        return entities

class JointExtractor:
    """联合抽取器类 - 结合实体抽取和关系抽取"""
    
    def __init__(self, ner_model_path: str, relation_model_path: str):
        self.entity_extractor = EntityExtractor(ner_model_path)
        self.relation_extractor = RelationExtractor(relation_model_path)
    
    def extract_spo(self, text: str) -> Dict:
        """抽取SPO三元组"""
        # 抽取实体
        entities = self.entity_extractor.extract_entities(text)
        
        # 抽取关系
        relations = self.relation_extractor.extract_relations(text, entities)
        
        # 构建SPO列表
        spo_list = []
        for relation in relations:
            spo_list.append({
                'h': {'name': relation['head_entity']},
                't': {'name': relation['tail_entity']},
                'relation': relation['relation_type']
            })
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations,
            'spo_list': spo_list
        }
    
    def extract_spo_batch(self, texts: List[str]) -> List[Dict]:
        """批量抽取SPO三元组"""
        results = []
        for text in texts:
            result = self.extract_spo(text)
            results.append(result)
        return results

def demo_relation_extraction():
    """演示关系抽取功能"""
    print("=" * 50)
    print("关系抽取演示")
    print("=" * 50)
    
    # 创建关系抽取器
    relation_extractor = RelationExtractor(RELATION_MODEL_PATH)
    
    # 示例文本
    text = "主轴故障导致异常振动，需要进行维修。"
    
    # 示例实体
    entities = [
        {'name': '主轴', 'type': 'EQUIPMENT'},
        {'name': '故障', 'type': 'FAULT'},
        {'name': '振动', 'type': 'SYMPTOM'},
        {'name': '维修', 'type': 'REPAIR'}
    ]
    
    print(f"输入文本: {text}")
    print(f"已知实体: {[e['name'] for e in entities]}")
    
    # 抽取所有关系
    relations = relation_extractor.extract_relations(text, entities)
    print(f"\n抽取的关系:")
    for i, rel in enumerate(relations, 1):
        print(f"  {i}. {rel['head_entity']} --[{rel['relation_type']}]--> {rel['tail_entity']} (置信度: {rel['confidence']:.2f})")
    
    # 按关系类型筛选
    cause_relations = relation_extractor.get_relations_by_type(text, entities, "导致")
    print(f"\n'导致'类型的关系:")
    for rel in cause_relations:
        print(f"  {rel['head_entity']} --[{rel['relation_type']}]--> {rel['tail_entity']}")

def demo_joint_extraction():
    """演示联合抽取功能"""
    print("\n" + "=" * 50)
    print("联合抽取演示")
    print("=" * 50)
    
    # 创建联合抽取器
    joint_extractor = JointExtractor(NER_MODEL_PATH, RELATION_MODEL_PATH)
    
    # 示例文本
    texts = [
        "主轴故障导致异常振动，需要进行维修。",
        "数控机床需要定期保养，避免设备损坏。",
        "刀具磨损会引起加工精度下降。"
    ]
    
    print("批量处理文本:")
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. {text}")
        
        # 抽取SPO三元组
        result = joint_extractor.extract_spo(text)
        
        print(f"   实体: {[e['name'] for e in result['entities']]}")
        print(f"   关系: {len(result['relations'])} 个")
        print(f"   SPO三元组:")
        for spo in result['spo_list']:
            print(f"     ({spo['h']['name']}, {spo['relation']}, {spo['t']['name']})")

def demo_batch_processing():
    """演示批量处理功能"""
    print("\n" + "=" * 50)
    print("批量处理演示")
    print("=" * 50)
    
    # 创建抽取器
    relation_extractor = RelationExtractor(RELATION_MODEL_PATH)
    joint_extractor = JointExtractor(NER_MODEL_PATH, RELATION_MODEL_PATH)
    
    # 批量文本
    texts = [
        "主轴故障导致异常振动。",
        "设备需要定期保养。",
        "刀具磨损影响加工精度。"
    ]
    
    # 批量实体（模拟）
    entities_list = [
        [{'name': '主轴', 'type': 'EQUIPMENT'}, {'name': '故障', 'type': 'FAULT'}, {'name': '振动', 'type': 'SYMPTOM'}],
        [{'name': '设备', 'type': 'EQUIPMENT'}, {'name': '保养', 'type': 'MAINTENANCE'}],
        [{'name': '刀具', 'type': 'TOOL'}, {'name': '磨损', 'type': 'FAULT'}, {'name': '精度', 'type': 'QUALITY'}]
    ]
    
    print("批量关系抽取:")
    batch_relations = relation_extractor.extract_relations_batch(texts, entities_list)
    for i, (text, relations) in enumerate(zip(texts, batch_relations), 1):
        print(f"\n{i}. {text}")
        for rel in relations:
            print(f"   {rel['head_entity']} --[{rel['relation_type']}]--> {rel['tail_entity']}")
    
    print("\n批量SPO抽取:")
    batch_spo = joint_extractor.extract_spo_batch(texts)
    for i, result in enumerate(batch_spo, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   SPO数量: {len(result['spo_list'])}")

def save_results_to_json(results: List[Dict], filename: str):
    """保存结果到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {filename}")

def main():
    """主函数"""
    print("关系抽取和联合抽取系统演示")
    print("=" * 60)
    
    try:
        # 演示各种功能
        demo_relation_extraction()
        demo_joint_extraction()
        demo_batch_processing()
        
        # 保存示例结果
        joint_extractor = JointExtractor(NER_MODEL_PATH, RELATION_MODEL_PATH)
        sample_text = "主轴故障导致异常振动，需要进行维修。"
        result = joint_extractor.extract_spo(sample_text)
        save_results_to_json([result], "extraction_result.json")
        
        print("\n" + "=" * 60)
        print("演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")

if __name__ == "__main__":
    main()