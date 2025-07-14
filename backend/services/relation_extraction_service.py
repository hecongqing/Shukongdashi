import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """关系数据类"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str

class RelationExtractionService:
    """关系抽取服务"""
    
    def __init__(self):
        self.relation_patterns = {
            # 故障-症状关系
            '故障导致症状': [
                r'([^，。；]*故障[^，。；]*)(导致|引起|造成|产生)([^，。；]*症状[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(出现|显示|表现)([^，。；]*症状[^，。；]*)',
            ],
            # 故障-原因关系
            '故障原因': [
                r'([^，。；]*故障[^，。；]*)(由|由于|因为|源于)([^，。；]*原因[^，。；]*)',
                r'([^，。；]*原因[^，。；]*)(导致|引起|造成)([^，。；]*故障[^，。；]*)',
            ],
            # 故障-解决方法关系
            '故障解决方法': [
                r'([^，。；]*故障[^，。；]*)(解决方法|解决方案|处理办法)([^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(修复|修理|维修)([^，。；]*)',
            ],
            # 设备-故障关系
            '设备故障': [
                r'([^，。；]*设备[^，。；]*)(出现|发生|产生)([^，。；]*故障[^，。；]*)',
                r'([^，。；]*设备[^，。；]*)(故障|问题)([^，。；]*)',
            ],
            # 部件-故障关系
            '部件故障': [
                r'([^，。；]*部件[^，。；]*)(损坏|故障|问题)([^，。；]*)',
                r'([^，。；]*部件[^，。；]*)(出现|发生)([^，。；]*故障[^，。；]*)',
            ]
        }
        
        # 常见的关系谓词
        self.common_predicates = {
            '导致', '引起', '造成', '产生', '出现', '显示', '表现',
            '由', '由于', '因为', '源于', '解决方法', '解决方案', '处理办法',
            '修复', '修理', '维修', '损坏', '故障', '问题'
        }
    
    async def extract_relations(self, text: str) -> List[Relation]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            
        Returns:
            关系列表
        """
        relations = []
        
        try:
            # 使用规则模式提取关系
            pattern_relations = self._extract_by_patterns(text)
            relations.extend(pattern_relations)
            
            # 使用启发式方法提取关系
            heuristic_relations = self._extract_by_heuristics(text)
            relations.extend(heuristic_relations)
            
            # 去重和排序
            relations = self._deduplicate_relations(relations)
            
            logger.info(f"从文本中提取到 {len(relations)} 个关系")
            return relations
            
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []
    
    def _extract_by_patterns(self, text: str) -> List[Relation]:
        """使用预定义模式提取关系"""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 3:
                        subject = match.group(1).strip()
                        predicate = match.group(2).strip()
                        object_entity = match.group(3).strip()
                        
                        # 过滤掉太短或太长的实体
                        if (len(subject) > 2 and len(subject) < 50 and 
                            len(object_entity) > 2 and len(object_entity) < 50):
                            
                            relation = Relation(
                                subject=subject,
                                predicate=predicate,
                                object=object_entity,
                                confidence=0.8,  # 模式匹配的置信度
                                source_text=match.group(0)
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
            for predicate in self.common_predicates:
                if predicate in sentence:
                    # 尝试提取主语和宾语
                    parts = sentence.split(predicate)
                    if len(parts) >= 2:
                        subject_part = parts[0].strip()
                        object_part = parts[1].strip()
                        
                        # 提取主语（最后一个名词短语）
                        subject = self._extract_entity(subject_part)
                        object_entity = self._extract_entity(object_part)
                        
                        if subject and object_entity:
                            relation = Relation(
                                subject=subject,
                                predicate=predicate,
                                object=object_entity,
                                confidence=0.6,  # 启发式方法的置信度
                                source_text=sentence
                            )
                            relations.append(relation)
        
        return relations
    
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
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 创建唯一标识
            key = (relation.subject, relation.predicate, relation.object)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        # 按置信度排序
        unique_relations.sort(key=lambda x: x.confidence, reverse=True)
        return unique_relations
    
    async def extract_relations_with_entities(self, text: str, entities: List[str]) -> List[Relation]:
        """
        基于已知实体提取关系
        
        Args:
            text: 输入文本
            entities: 已知实体列表
            
        Returns:
            关系列表
        """
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
                                for predicate in self.common_predicates:
                                    if predicate in sentence:
                                        # 确定主语和宾语
                                        entity1_pos = sentence.find(entity1)
                                        entity2_pos = sentence.find(entity2)
                                        predicate_pos = sentence.find(predicate)
                                        
                                        if entity1_pos < predicate_pos < entity2_pos:
                                            relation = Relation(
                                                subject=entity1,
                                                predicate=predicate,
                                                object=entity2,
                                                confidence=0.7,
                                                source_text=sentence
                                            )
                                            relations.append(relation)
                                        elif entity2_pos < predicate_pos < entity1_pos:
                                            relation = Relation(
                                                subject=entity2,
                                                predicate=predicate,
                                                object=entity1,
                                                confidence=0.7,
                                                source_text=sentence
                                            )
                                            relations.append(relation)
            
            # 去重
            relations = self._deduplicate_relations(relations)
            
            logger.info(f"基于实体提取到 {len(relations)} 个关系")
            return relations
            
        except Exception as e:
            logger.error(f"基于实体的关系抽取失败: {e}")
            return []
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict:
        """获取关系统计信息"""
        stats = {
            'total_relations': len(relations),
            'predicate_counts': {},
            'confidence_distribution': {
                'high': 0,    # >= 0.8
                'medium': 0,  # 0.6-0.8
                'low': 0      # < 0.6
            }
        }
        
        for relation in relations:
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