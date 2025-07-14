"""
关系验证器

用于验证和过滤从文本中提取的关系
"""

import re
import logging
from typing import List, Dict, Set
from dataclasses import dataclass

from .relation_extractor import Relation

logger = logging.getLogger(__name__)

@dataclass
class ValidationRule:
    """验证规则"""
    name: str
    description: str
    enabled: bool = True

class RelationValidator:
    """关系验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 定义验证规则
        self.validation_rules = {
            'entity_length': ValidationRule(
                name='entity_length',
                description='实体长度验证：实体长度应在合理范围内',
                enabled=True
            ),
            'entity_quality': ValidationRule(
                name='entity_quality',
                description='实体质量验证：过滤低质量实体',
                enabled=True
            ),
            'relation_semantics': ValidationRule(
                name='relation_semantics',
                description='关系语义验证：验证关系的语义合理性',
                enabled=True
            ),
            'duplicate_check': ValidationRule(
                name='duplicate_check',
                description='重复检查：检查重复关系',
                enabled=True
            ),
            'confidence_threshold': ValidationRule(
                name='confidence_threshold',
                description='置信度阈值：过滤低置信度关系',
                enabled=True
            )
        }
        
        # 低质量实体模式
        self.low_quality_patterns = [
            r'^[0-9]+$',  # 纯数字
            r'^[a-zA-Z]+$',  # 纯英文
            r'^[^\u4e00-\u9fff]+$',  # 不包含中文
            r'^.{1,2}$',  # 太短的实体
            r'^.{50,}$',  # 太长的实体
        ]
        
        # 无意义词汇
        self.meaningless_words = {
            '的', '了', '在', '是', '有', '和', '与', '或', '但', '而',
            '这', '那', '什么', '怎么', '为什么', '如何', '可以', '应该',
            '需要', '必须', '可能', '也许', '大概', '大约', '左右'
        }
        
        # 关系语义验证规则
        self.semantic_rules = {
            'fault_symptom': {
                'required_keywords': ['故障', '症状'],
                'forbidden_combinations': []
            },
            'fault_cause': {
                'required_keywords': ['故障', '原因'],
                'forbidden_combinations': []
            },
            'fault_solution': {
                'required_keywords': ['故障', '解决'],
                'forbidden_combinations': []
            },
            'equipment_fault': {
                'required_keywords': ['设备', '故障'],
                'forbidden_combinations': []
            },
            'component_fault': {
                'required_keywords': ['部件', '故障'],
                'forbidden_combinations': []
            }
        }
    
    def validate_relations(self, relations: List[Relation], 
                          min_confidence: float = 0.5,
                          enable_rules: Set[str] = None) -> List[Relation]:
        """
        验证关系列表
        
        Args:
            relations: 关系列表
            min_confidence: 最小置信度阈值
            enable_rules: 启用的验证规则集合
            
        Returns:
            验证通过的关系列表
        """
        if enable_rules is None:
            enable_rules = set(self.validation_rules.keys())
        
        validated_relations = []
        
        for relation in relations:
            is_valid = True
            
            # 应用各种验证规则
            if 'confidence_threshold' in enable_rules:
                if not self._validate_confidence(relation, min_confidence):
                    is_valid = False
                    self.logger.debug(f"关系置信度验证失败: {relation}")
            
            if 'entity_length' in enable_rules:
                if not self._validate_entity_length(relation):
                    is_valid = False
                    self.logger.debug(f"实体长度验证失败: {relation}")
            
            if 'entity_quality' in enable_rules:
                if not self._validate_entity_quality(relation):
                    is_valid = False
                    self.logger.debug(f"实体质量验证失败: {relation}")
            
            if 'relation_semantics' in enable_rules:
                if not self._validate_relation_semantics(relation):
                    is_valid = False
                    self.logger.debug(f"关系语义验证失败: {relation}")
            
            if is_valid:
                validated_relations.append(relation)
        
        # 重复检查
        if 'duplicate_check' in enable_rules:
            validated_relations = self._remove_duplicates(validated_relations)
        
        self.logger.info(f"关系验证完成: {len(relations)} -> {len(validated_relations)}")
        return validated_relations
    
    def _validate_confidence(self, relation: Relation, min_confidence: float) -> bool:
        """验证关系置信度"""
        return relation.confidence >= min_confidence
    
    def _validate_entity_length(self, relation: Relation) -> bool:
        """验证实体长度"""
        # 检查主语长度
        if len(relation.subject) < 2 or len(relation.subject) > 50:
            return False
        
        # 检查宾语长度
        if len(relation.object) < 2 or len(relation.object) > 50:
            return False
        
        return True
    
    def _validate_entity_quality(self, relation: Relation) -> bool:
        """验证实体质量"""
        # 检查主语质量
        if not self._is_high_quality_entity(relation.subject):
            return False
        
        # 检查宾语质量
        if not self._is_high_quality_entity(relation.object):
            return False
        
        return True
    
    def _is_high_quality_entity(self, entity: str) -> bool:
        """检查实体是否为高质量实体"""
        # 检查低质量模式
        for pattern in self.low_quality_patterns:
            if re.match(pattern, entity):
                return False
        
        # 检查是否包含无意义词汇
        words = entity.split()
        meaningless_count = sum(1 for word in words if word in self.meaningless_words)
        if meaningless_count > len(words) * 0.5:  # 超过50%是无意义词汇
            return False
        
        # 检查是否包含中文（对于中文文本）
        if not re.search(r'[\u4e00-\u9fff]', entity):
            return False
        
        return True
    
    def _validate_relation_semantics(self, relation: Relation) -> bool:
        """验证关系语义"""
        relation_type = relation.relation_type
        
        if relation_type not in self.semantic_rules:
            return True  # 未知类型的关系默认通过
        
        rule = self.semantic_rules[relation_type]
        
        # 检查必需关键词
        if 'required_keywords' in rule:
            text = f"{relation.subject} {relation.predicate} {relation.object}".lower()
            for keyword in rule['required_keywords']:
                if keyword not in text:
                    return False
        
        # 检查禁止组合
        if 'forbidden_combinations' in rule:
            for forbidden in rule['forbidden_combinations']:
                if (forbidden[0] in relation.subject and 
                    forbidden[1] in relation.object):
                    return False
        
        return True
    
    def _remove_duplicates(self, relations: List[Relation]) -> List[Relation]:
        """移除重复关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 创建唯一标识（考虑关系类型）
            key = (relation.subject, relation.predicate, relation.object, relation.relation_type)
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def get_validation_statistics(self, original_relations: List[Relation], 
                                 validated_relations: List[Relation]) -> Dict:
        """获取验证统计信息"""
        stats = {
            'original_count': len(original_relations),
            'validated_count': len(validated_relations),
            'filtered_count': len(original_relations) - len(validated_relations),
            'filter_rate': 0.0,
            'validation_details': {}
        }
        
        if len(original_relations) > 0:
            stats['filter_rate'] = stats['filtered_count'] / len(original_relations)
        
        return stats
    
    def add_custom_rule(self, rule_name: str, rule: ValidationRule):
        """添加自定义验证规则"""
        self.validation_rules[rule_name] = rule
    
    def enable_rule(self, rule_name: str):
        """启用验证规则"""
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = True
    
    def disable_rule(self, rule_name: str):
        """禁用验证规则"""
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = False
    
    def get_enabled_rules(self) -> Set[str]:
        """获取启用的验证规则"""
        return {name for name, rule in self.validation_rules.items() if rule.enabled}
    
    def add_low_quality_pattern(self, pattern: str):
        """添加低质量实体模式"""
        self.low_quality_patterns.append(pattern)
    
    def add_meaningless_word(self, word: str):
        """添加无意义词汇"""
        self.meaningless_words.add(word)
    
    def add_semantic_rule(self, relation_type: str, rule: Dict):
        """添加语义验证规则"""
        self.semantic_rules[relation_type] = rule