#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实用的关系抽取示例
展示如何与现有的实体抽取和关系抽取服务集成
"""

import asyncio
import sys
import os
from typing import List, Dict, Optional
import json

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from services.entity_extraction_service import EntityExtractionService
    from services.relation_extraction_service import RelationExtractionService, Relation
except ImportError:
    print("无法导入服务模块，使用模拟实现")
    # 模拟实现
    class Relation:
        def __init__(self, subject: str, predicate: str, object: str, confidence: float, source_text: str):
            self.subject = subject
            self.predicate = predicate
            self.object = object
            self.confidence = confidence
            self.source_text = source_text
    
    class EntityExtractionService:
        async def extract_entities(self, text: str):
            return []
    
    class RelationExtractionService:
        async def extract_relations(self, text: str):
            return []
        
        async def extract_relations_with_entities(self, text: str, entities: List[str]):
            return []

class PracticalRelationExtractor:
    """实用的关系抽取器"""
    
    def __init__(self):
        self.entity_service = EntityExtractionService()
        self.relation_service = RelationExtractionService()
    
    async def extract_entities_and_relations(self, text: str) -> Dict:
        """抽取实体和关系"""
        try:
            # 1. 抽取实体
            entities = await self.entity_service.extract_entities(text)
            print(f"抽取到 {len(entities)} 个实体")
            
            # 2. 抽取关系
            relations = await self.relation_service.extract_relations(text)
            print(f"抽取到 {len(relations)} 个关系")
            
            # 3. 基于实体的关系抽取
            entity_names = [entity.text for entity in entities] if hasattr(entities[0], 'text') else []
            if entity_names:
                entity_relations = await self.relation_service.extract_relations_with_entities(text, entity_names)
                print(f"基于实体抽取到 {len(entity_relations)} 个关系")
                relations.extend(entity_relations)
            
            return {
                'text': text,
                'entities': entities,
                'relations': relations,
                'entity_count': len(entities),
                'relation_count': len(relations)
            }
            
        except Exception as e:
            print(f"抽取过程中出现错误: {e}")
            return {
                'text': text,
                'entities': [],
                'relations': [],
                'entity_count': 0,
                'relation_count': 0,
                'error': str(e)
            }
    
    async def batch_extract(self, texts: List[str]) -> List[Dict]:
        """批量抽取"""
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n处理第 {i}/{len(texts)} 个文本...")
            result = await self.extract_entities_and_relations(text)
            results.append(result)
        return results
    
    def format_relations(self, relations: List[Relation]) -> List[Dict]:
        """格式化关系输出"""
        formatted = []
        for rel in relations:
            formatted.append({
                'subject': rel.subject,
                'predicate': rel.predicate,
                'object': rel.object,
                'confidence': rel.confidence,
                'source_text': rel.source_text
            })
        return formatted
    
    def format_entities(self, entities) -> List[Dict]:
        """格式化实体输出"""
        formatted = []
        for entity in entities:
            if hasattr(entity, 'text'):
                formatted.append({
                    'text': entity.text,
                    'type': getattr(entity, 'entity_type', 'UNKNOWN'),
                    'confidence': getattr(entity, 'confidence', 0.0),
                    'start': getattr(entity, 'start_pos', 0),
                    'end': getattr(entity, 'end_pos', 0)
                })
            else:
                formatted.append(entity)
        return formatted

class EnhancedJointExtractor:
    """增强的联合抽取器"""
    
    def __init__(self):
        self.practical_extractor = PracticalRelationExtractor()
    
    async def extract_spo_triples(self, text: str) -> Dict:
        """抽取SPO三元组"""
        result = await self.practical_extractor.extract_entities_and_relations(text)
        
        # 构建SPO三元组
        spo_list = []
        relations = result.get('relations', [])
        
        for relation in relations:
            spo_list.append({
                'subject': relation.subject,
                'predicate': relation.predicate,
                'object': relation.object,
                'confidence': relation.confidence
            })
        
        result['spo_list'] = spo_list
        result['spo_count'] = len(spo_list)
        
        return result
    
    async def extract_spo_batch(self, texts: List[str]) -> List[Dict]:
        """批量抽取SPO三元组"""
        results = []
        for text in texts:
            result = await self.extract_spo_triples(text)
            results.append(result)
        return results
    
    def get_relations_by_type(self, result: Dict, relation_type: str) -> List[Dict]:
        """根据关系类型筛选"""
        relations = result.get('relations', [])
        filtered = []
        for rel in relations:
            if rel.predicate == relation_type:
                filtered.append({
                    'subject': rel.subject,
                    'predicate': rel.predicate,
                    'object': rel.object,
                    'confidence': rel.confidence
                })
        return filtered

async def demo_practical_extraction():
    """演示实用的抽取功能"""
    print("=" * 60)
    print("实用关系抽取演示")
    print("=" * 60)
    
    # 创建抽取器
    extractor = PracticalRelationExtractor()
    joint_extractor = EnhancedJointExtractor()
    
    # 示例文本（数控机床故障诊断相关）
    sample_texts = [
        "主轴轴承磨损导致主轴振动异常，需要更换轴承。",
        "数控系统报警代码ALM401表示伺服电机过载，检查电机负载。",
        "刀具磨损严重会影响加工精度，需要及时更换刀具。",
        "机床导轨润滑不足会导致导轨磨损，定期添加润滑油。",
        "主轴电机温度过高会引起主轴停止，检查冷却系统。"
    ]
    
    print("处理单个文本示例:")
    text = sample_texts[0]
    print(f"输入文本: {text}")
    
    # 抽取实体和关系
    result = await extractor.extract_entities_and_relations(text)
    
    print(f"\n抽取结果:")
    print(f"  实体数量: {result['entity_count']}")
    print(f"  关系数量: {result['relation_count']}")
    
    # 格式化输出
    entities = extractor.format_entities(result['entities'])
    relations = extractor.format_relations(result['relations'])
    
    if entities:
        print(f"\n  实体列表:")
        for entity in entities:
            print(f"    - {entity.get('text', '')} ({entity.get('type', '')})")
    
    if relations:
        print(f"\n  关系列表:")
        for rel in relations:
            print(f"    - {rel['subject']} --[{rel['predicate']}]--> {rel['object']} (置信度: {rel['confidence']:.2f})")
    
    # SPO三元组抽取
    print(f"\nSPO三元组抽取:")
    spo_result = await joint_extractor.extract_spo_triples(text)
    spo_list = spo_result.get('spo_list', [])
    
    for i, spo in enumerate(spo_list, 1):
        print(f"  {i}. ({spo['subject']}, {spo['predicate']}, {spo['object']})")
    
    return result

async def demo_batch_processing():
    """演示批量处理"""
    print("\n" + "=" * 60)
    print("批量处理演示")
    print("=" * 60)
    
    joint_extractor = EnhancedJointExtractor()
    
    # 批量文本
    texts = [
        "主轴故障导致振动异常。",
        "设备需要定期保养维护。",
        "刀具磨损影响加工质量。"
    ]
    
    print("批量处理文本:")
    results = await joint_extractor.extract_spo_batch(texts)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   实体: {result['entity_count']} 个")
        print(f"   关系: {result['relation_count']} 个")
        print(f"   SPO: {result['spo_count']} 个")
        
        # 显示SPO三元组
        spo_list = result.get('spo_list', [])
        for spo in spo_list:
            print(f"     ({spo['subject']}, {spo['predicate']}, {spo['object']})")

def save_results(results: List[Dict], filename: str):
    """保存结果到JSON文件"""
    # 转换结果格式以便JSON序列化
    serializable_results = []
    for result in results:
        serializable_result = {
            'text': result['text'],
            'entity_count': result['entity_count'],
            'relation_count': result['relation_count'],
            'spo_count': result.get('spo_count', 0),
            'entities': result.get('entities', []),
            'relations': result.get('relations', []),
            'spo_list': result.get('spo_list', [])
        }
        serializable_results.append(serializable_result)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {filename}")

async def main():
    """主函数"""
    print("实用关系抽取系统演示")
    print("=" * 80)
    
    try:
        # 演示实用抽取
        result = await demo_practical_extraction()
        
        # 演示批量处理
        await demo_batch_processing()
        
        # 保存结果
        save_results([result], "practical_extraction_result.json")
        
        print("\n" + "=" * 80)
        print("演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())