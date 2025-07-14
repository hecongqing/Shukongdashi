"""
关系抽取器

实现从文本中提取实体间关系的核心功能
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .relation_patterns import RelationPatterns

logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """关系数据类"""
    subject: str
    predicate: str
    object: str
    relation_type: str
    confidence: float
    source_text: str
    start_pos: int
    end_pos: int

class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self):
        self.patterns = RelationPatterns()
        self.logger = logging.getLogger(__name__)
    
    def extract_relations(self, text: str, entities: Optional[List[str]] = None) -> List[Relation]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 可选的实体列表，用于基于已知实体提取关系
            
        Returns:
            关系列表
        """
        relations = []
        
        try:
            # 使用模式匹配提取关系
            pattern_relations = self._extract_by_patterns(text)
            relations.extend(pattern_relations)
            
            # 使用启发式方法提取关系
            heuristic_relations = self._extract_by_heuristics(text)
            relations.extend(heuristic_relations)
            
            # 如果提供了实体列表，基于实体提取关系
            if entities:
                entity_relations = self._extract_by_entities(text, entities)
                relations.extend(entity_relations)
            
            # 去重和排序
            relations = self._deduplicate_relations(relations)
            
            self.logger.info(f"从文本中提取到 {len(relations)} 个关系")
            return relations
            
        except Exception as e:
            self.logger.error(f"关系抽取失败: {e}")
            return []
    
    def _extract_by_patterns(self, text: str) -> List[Relation]:
        """使用预定义模式提取关系"""
        relations = []
        
        all_patterns = self.patterns.get_all_patterns()
        
        for relation_type, patterns in all_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 3:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                        object_entity = match.group(3).strip()
                        
                        # 过滤掉太短或太长的实体
                        if self._is_valid_entity(subject) and self._is_valid_entity(object_entity):
                            relation = Relation(
                                subject=subject,
                                predicate=predicate,
                                object=object_entity,
                                relation_type=relation_type,
                                confidence=0.8,  # 模式匹配的置信度
                                source_text=match.group(0),
                                start_pos=match.start(),
                                end_pos=match.end()
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_by_heuristics(self, text: str) -> List[Relation]:
        """使用启发式方法提取关系"""
        relations = []
        
        # 分句处理
        sentences = re.split(r'[。！？；]', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
                
            # 寻找包含关系谓词的句子
            for predicate in self.patterns.common_predicates:
                if predicate in sentence:
                    # 尝试提取主语和宾语
                    parts = sentence.split(predicate)
                    if len(parts) >= 2:
                        subject_part = parts[0].strip()
                        object_part = parts[1].strip()
                        
                        # 提取主语和宾语
                        subject = self._extract_entity(subject_part)
                        object_entity = self._extract_entity(object_part)
                        
                        if subject and object_entity:
                            # 推断关系类型
                            relation_type = self._infer_relation_type(subject, predicate, object_entity)
                            
                            relation = Relation(
                                subject=subject,
                                predicate=predicate,
                                object=object_entity,
                                relation_type=relation_type,
                                confidence=0.6,  # 启发式方法的置信度
                                source_text=sentence,
                                start_pos=text.find(sentence),
                                end_pos=text.find(sentence) + len(sentence)
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_by_entities(self, text: str, entities: List[str]) -> List[Relation]:
        """基于已知实体提取关系"""
        relations = []
        
        try:
            # 寻找实体间的关系
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j and entity1 in text and entity2 in text:
                        # 检查两个实体是否在同一个句子中
                        sentences = re.split(r'[。！？；]', text)
                        for sentence in sentences:
                            if entity1 in sentence and entity2 in sentence:
                                # 寻找连接词
                                for predicate in self.patterns.common_predicates:
                                    if predicate in sentence:
                                        # 确定主语和宾语
                                        entity1_pos = sentence.find(entity1)
                                        entity2_pos = sentence.find(entity2)
                                        predicate_pos = sentence.find(predicate)
                                        
                                        if entity1_pos < predicate_pos < entity2_pos:
                                            relation_type = self._infer_relation_type(entity1, predicate, entity2)
                                            relation = Relation(
                                                subject=entity1,
                                                predicate=predicate,
                                                object=entity2,
                                                relation_type=relation_type,
                                                confidence=0.7,
                                                source_text=sentence,
                                                start_pos=text.find(sentence),
                                                end_pos=text.find(sentence) + len(sentence)
                                            )
                                            relations.append(relation)
                                        elif entity2_pos < predicate_pos < entity1_pos:
                                            relation_type = self._infer_relation_type(entity2, predicate, entity1)
                                            relation = Relation(
                                                subject=entity2,
                                                predicate=predicate,
                                                object=entity1,
                                                relation_type=relation_type,
                                                confidence=0.7,
                                                source_text=sentence,
                                                start_pos=text.find(sentence),
                                                end_pos=text.find(sentence) + len(sentence)
                                            )
                                            relations.append(relation)
            
            self.logger.info(f"基于实体提取到 {len(relations)} 个关系")
            return relations
            
        except Exception as e:
            self.logger.error(f"基于实体的关系抽取失败: {e}")
            return []
    
    def _extract_entity(self, text: str) -> Optional[str]:
        """从文本片段中提取实体"""
        if not text:
            return None
            
        # 简单的实体提取：取最后一个有意义的短语
        words = text.split()
        if len(words) <= 2:
            return text.strip()
        
        # 尝试提取最后一个名词短语
        for i in range(len(words) - 1, 0, -1):
            phrase = ' '.join(words[i:]).strip()
            if len(phrase) > 2 and len(phrase) < 30:
                return phrase
        
        return text.strip()
    
    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        return len(entity) > 2 and len(entity) < 50
    
    def _infer_relation_type(self, subject: str, predicate: str, object_entity: str) -> str:
        """推断关系类型"""
        # 基于主语、谓词和宾语的语义推断关系类型
        subject_lower = subject.lower()
        object_lower = object_entity.lower()
        predicate_lower = predicate.lower()
        
        # 故障相关关系
        if '故障' in subject_lower or '故障' in object_lower:
            if '症状' in subject_lower or '症状' in object_lower:
                return 'fault_symptom'
            elif '原因' in subject_lower or '原因' in object_lower:
                return 'fault_cause'
            elif '解决' in predicate_lower or '修复' in predicate_lower:
                return 'fault_solution'
            elif '影响' in predicate_lower:
                return 'fault_impact'
        
        # 设备相关关系
        if '设备' in subject_lower or '设备' in object_lower:
            if '故障' in subject_lower or '故障' in object_lower:
                return 'equipment_fault'
        
        # 部件相关关系
        if '部件' in subject_lower or '部件' in object_lower:
            if '故障' in subject_lower or '故障' in object_lower:
                return 'component_fault'
        
        # 维修相关关系
        if '维修' in subject_lower or '维修' in object_lower:
            if '工具' in subject_lower or '工具' in object_lower:
                return 'maintenance_tool'
            elif '人员' in subject_lower or '人员' in object_lower:
                return 'maintenance_personnel'
            elif '时间' in subject_lower or '时间' in object_lower:
                return 'maintenance_time'
        
        # 检测相关关系
        if '检测' in subject_lower or '检测' in object_lower:
            if '方法' in subject_lower or '方法' in object_lower:
                return 'detection_method'
            elif '设备' in subject_lower or '设备' in object_lower:
                return 'detection_equipment'
        
        # 默认关系类型
        return 'unknown'
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 创建唯一标识
            key = (relation.subject, relation.predicate, relation.object, relation.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        # 按置信度排序
        unique_relations.sort(key=lambda x: x.confidence, reverse=True)
        return unique_relations
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict:
        """获取关系统计信息"""
        stats = {
            'total_relations': len(relations),
            'relation_type_counts': {},
            'predicate_counts': {},
            'confidence_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # 0.6-0.8
                'low': 0      # < 0.6
            }
        }
        
        for relation in relations:
            # 统计关系类型
            relation_type = relation.relation_type
            stats['relation_type_counts'][relation_type] = stats['relation_type_counts'].get(relation_type, 0) + 1
            
            # 统计谓词
            predicate = relation.predicate
            stats['predicate_counts'][predicate] = stats['predicate_counts'].get(predicate, 0) + 1
            
            # 统计置信度分布
            if relation.confidence >= 0.8:
                stats['confidence_distribution']['high'] += 1
            elif relation.confidence >= 0.6:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        return stats
    
    def filter_relations_by_type(self, relations: List[Relation], relation_type: str) -> List[Relation]:
        """根据关系类型过滤关系"""
        return [rel for rel in relations if rel.relation_type == relation_type]
    
    def filter_relations_by_confidence(self, relations: List[Relation], min_confidence: float) -> List[Relation]:
        """根据置信度过滤关系"""
        return [rel for rel in relations if rel.confidence >= min_confidence]