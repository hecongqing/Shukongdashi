#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关系抽取和联合抽取示例
展示如何使用RelationExtractor和JointExtractor类
"""

import asyncio
from typing import List, Dict
import json

# 假设的RelationPredictor类（实际项目中需要实现）
class RelationPredictor:
    """关系预测器 - 实际项目中需要实现具体的模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"加载关系预测模型: {model_path}")
    
    def predict_all_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """预测所有关系"""
        # 这里是示例实现，实际项目中需要调用真实的模型
        relations = []
        
        # 获取实体文本
        entity_texts = [entity.get('text', entity.get('name', '')) for entity in entities]
        
        # 关系关键词映射
        relation_keywords = {
            '导致': ['导致', '造成', '引起', '产生'],
            '引起': ['引起', '导致', '造成', '产生'],
            '解决方法': ['维修', '修理', '修复', '更换', '检查'],
            '属于': ['属于', '是', '为'],
            '包含': ['包含', '包括', '有']
        }
        
        # 遍历所有实体对，寻找关系
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i != j:
                    entity1_text = entity1.get('text', '')
                    entity2_text = entity2.get('text', '')
                    
                    # 检查两个实体之间是否有关系
                    for relation_type, keywords in relation_keywords.items():
                        for keyword in keywords:
                            if keyword in text:
                                # 检查实体在文本中的位置关系
                                entity1_pos = text.find(entity1_text)
                                entity2_pos = text.find(entity2_text)
                                keyword_pos = text.find(keyword)
                                
                                # 如果关键词在两个实体之间，可能存在关系
                                if (entity1_pos != -1 and entity2_pos != -1 and keyword_pos != -1):
                                    # 简单的距离判断
                                    if abs(entity1_pos - keyword_pos) < 20 and abs(entity2_pos - keyword_pos) < 20:
                                        # 确定主语和宾语
                                        if entity1_pos < keyword_pos < entity2_pos:
                                            relations.append({
                                                'head_entity': entity1_text,
                                                'tail_entity': entity2_text,
                                                'relation_type': relation_type,
                                                'relation_confidence': 0.85
                                            })
                                        elif entity2_pos < keyword_pos < entity1_pos:
                                            relations.append({
                                                'head_entity': entity2_text,
                                                'tail_entity': entity1_text,
                                                'relation_type': relation_type,
                                                'relation_confidence': 0.85
                                            })
        
        # 特殊规则：故障-症状关系
        fault_entities = [e for e in entities if '故障' in e.get('text', '') or 'FAULT' in e.get('entity_type', '')]
        symptom_entities = [e for e in entities if '异常' in e.get('text', '') or '报警' in e.get('text', '') or 'FAULT_SYMPTOM' in e.get('entity_type', '')]
        
        for fault in fault_entities:
            for symptom in symptom_entities:
                if fault.get('text') != symptom.get('text'):
                    relations.append({
                        'head_entity': fault.get('text', ''),
                        'tail_entity': symptom.get('text', ''),
                        'relation_type': '导致',
                        'relation_confidence': 0.9
                    })
        
        # 特殊规则：故障-维修方法关系
        repair_entities = [e for e in entities if '维修' in e.get('text', '') or '更换' in e.get('text', '') or 'REPAIR' in e.get('entity_type', '')]
        
        for fault in fault_entities:
            for repair in repair_entities:
                if fault.get('text') != repair.get('text'):
                    relations.append({
                        'head_entity': fault.get('text', ''),
                        'tail_entity': repair.get('text', ''),
                        'relation_type': '解决方法',
                        'relation_confidence': 0.8
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

# 假设的EntityExtractor类
class EntityExtractor:
    """实体抽取器 - 示例实现"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"加载实体抽取模型: {model_path}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取实体"""
        # 示例实现
        entities = []
        
        # 简单的关键词匹配
        keywords = {
            'EQUIPMENT': ['数控机床', '车床', '铣床', '加工中心'],
            'FAULT_SYMPTOM': ['故障', '异常', '报警', '停机'],
            'FAULT_CAUSE': ['原因', '导致', '引起'],
            'REPAIR_METHOD': ['维修', '修理', '修复', '更换']
        }
        
        for entity_type, words in keywords.items():
            for word in words:
                if word in text:
                    entities.append({
                        'text': word,
                        'entity_type': entity_type,
                        'start_pos': text.find(word),
                        'end_pos': text.find(word) + len(word),
                        'confidence': 0.9
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

def print_results(title: str, data: any):
    """打印结果"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(json.dumps(data, ensure_ascii=False, indent=2))

async def main():
    """主函数 - 演示各种抽取功能"""
    
    # 示例文本
    sample_texts = [
        "数控机床主轴故障导致加工精度异常，需要更换主轴轴承进行维修。",
        "FANUC系统报警代码ALM401表示伺服电机过载，通常由机械卡死引起。",
        "加工中心刀库故障造成换刀失败，维修方法是检查刀库电机和传感器。"
    ]
    
    print("关系抽取和联合抽取示例")
    print("="*60)
    
    # 1. 关系抽取器示例
    print("\n1. 关系抽取器示例")
    relation_extractor = RelationExtractor("models/relation_model")
    
    # 示例实体
    sample_entities = [
        {'text': '数控机床', 'entity_type': 'EQUIPMENT'},
        {'text': '主轴故障', 'entity_type': 'FAULT_SYMPTOM'},
        {'text': '加工精度异常', 'entity_type': 'FAULT_SYMPTOM'},
        {'text': '更换主轴轴承', 'entity_type': 'REPAIR_METHOD'}
    ]
    
    # 抽取关系
    relations = relation_extractor.extract_relations(sample_texts[0], sample_entities)
    print_results("关系抽取结果", relations)
    
    # 按关系类型过滤
    cause_relations = relation_extractor.get_relations_by_type(
        sample_texts[0], sample_entities, "导致"
    )
    print_results("'导致'关系", cause_relations)
    
    # 2. 联合抽取器示例
    print("\n2. 联合抽取器示例")
    joint_extractor = JointExtractor("models/ner_model", "models/relation_model")
    
    # 抽取单个文本的SPO
    spo_result = joint_extractor.extract_spo(sample_texts[0])
    print_results("SPO抽取结果", spo_result)
    
    # 批量抽取
    batch_results = joint_extractor.extract_spo_batch(sample_texts)
    print_results("批量SPO抽取结果", batch_results)
    
    # 3. 实际应用场景示例
    print("\n3. 实际应用场景示例")
    
    # 故障诊断文本
    fault_text = "FANUC-0i系统出现ALM401报警，机床主轴无法启动，经检查发现主轴电机过载，需要更换电机轴承。"
    
    # 使用联合抽取器
    fault_result = joint_extractor.extract_spo(fault_text)
    
    print("故障诊断文本:", fault_text)
    print_results("抽取的实体和关系", {
        'entities': [f"{e['text']}({e['entity_type']})" for e in fault_result['entities']],
        'relations': [f"{r['head_entity']} --{r['relation_type']}--> {r['tail_entity']}" 
                     for r in fault_result['relations']],
        'spo_triples': [f"({s['h']['name']}, {s['relation']}, {s['t']['name']})" 
                       for s in fault_result['spo_list']]
    })
    
    # 4. 关系类型统计
    print("\n4. 关系类型统计")
    all_relations = []
    for text in sample_texts:
        entities = joint_extractor.entity_extractor.extract_entities(text)
        relations = joint_extractor.relation_extractor.extract_relations(text, entities)
        all_relations.extend(relations)
    
    relation_types = {}
    for rel in all_relations:
        rel_type = rel['relation_type']
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
    
    print_results("关系类型统计", relation_types)

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())