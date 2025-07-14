#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题分析模块
用于理解用户输入的装备制造故障相关问题
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class QuestionAnalyzer:
    """问题分析器"""
    
    def __init__(self):
        self.question_types = {
            'fault_diagnosis': {
                'keywords': ['故障', '问题', '异常', '报警', '错误', '不工作', '失灵'],
                'patterns': [
                    r'(.+?)出现(.+?)故障',
                    r'(.+?)发生(.+?)问题',
                    r'(.+?)报警(.+?)',
                    r'(.+?)异常(.+?)'
                ]
            },
            'solution_query': {
                'keywords': ['怎么', '如何', '怎么办', '解决方法', '维修', '修理', '修复'],
                'patterns': [
                    r'(.+?)怎么(.+?)',
                    r'如何(.+?)',
                    r'(.+?)的解决方法',
                    r'(.+?)怎么维修'
                ]
            },
            'component_query': {
                'keywords': ['部件', '组件', '零件', '元件', '器件'],
                'patterns': [
                    r'(.+?)的(.+?)',
                    r'(.+?)部件(.+?)',
                    r'(.+?)组件(.+?)'
                ]
            },
            'cause_analysis': {
                'keywords': ['原因', '为什么', '导致', '引起', '造成'],
                'patterns': [
                    r'(.+?)的原因',
                    r'为什么(.+?)',
                    r'(.+?)导致(.+?)',
                    r'(.+?)引起(.+?)'
                ]
            },
            'prevention_query': {
                'keywords': ['预防', '避免', '防止', '保养', '维护'],
                'patterns': [
                    r'如何预防(.+?)',
                    r'(.+?)的预防措施',
                    r'(.+?)的保养方法',
                    r'如何避免(.+?)'
                ]
            }
        }
        
        self.entity_types = {
            'equipment': ['机床', '设备', '机器', '系统', '装置'],
            'component': ['主轴', '轴承', '电机', '驱动器', '传感器', '控制器'],
            'fault': ['振动', '噪音', '过热', '漏油', '卡死', '不动作'],
            'brand': ['FANUC', '西门子', '发那科', '三菱', '海德汉'],
            'error_code': [r'[A-Z]{2,}\d{3,}', r'ALM\d+', r'E\d+']
        }
        
        # 加载专业词典
        self.load_domain_dictionary()
    
    def load_domain_dictionary(self):
        """加载领域专业词典"""
        # 添加装备制造领域的专业词汇
        domain_words = [
            '数控机床', '加工中心', '车床', '铣床', '钻床', '磨床',
            '主轴', '进给轴', '工作台', '刀库', '换刀装置', '冷却系统',
            '润滑系统', '液压系统', '气压系统', '电气系统', '控制系统',
            '伺服电机', '步进电机', '编码器', '光栅尺', '限位开关',
            '变频器', '驱动器', 'PLC', '触摸屏', '操作面板',
            '报警', '故障', '异常', '振动', '噪音', '过热', '漏油',
            '精度', '定位', '重复定位', '加工精度', '表面粗糙度'
        ]
        
        for word in domain_words:
            jieba.add_word(word)
    
    def analyze_question_type(self, question: str) -> Dict[str, Any]:
        """分析问题类型"""
        question = question.strip()
        
        # 计算每种问题类型的匹配度
        type_scores = {}
        
        for q_type, config in self.question_types.items():
            score = 0
            
            # 关键词匹配
            for keyword in config['keywords']:
                if keyword in question:
                    score += 1
            
            # 模式匹配
            for pattern in config['patterns']:
                matches = re.findall(pattern, question)
                if matches:
                    score += len(matches) * 2
            
            type_scores[q_type] = score
        
        # 确定主要问题类型
        if type_scores:
            primary_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[primary_type] / max(type_scores.values()) if max(type_scores.values()) > 0 else 0
        else:
            primary_type = 'general'
            confidence = 0
        
        return {
            'question_type': primary_type,
            'confidence': confidence,
            'type_scores': type_scores
        }
    
    def extract_entities(self, question: str) -> List[Dict[str, Any]]:
        """提取问题中的实体"""
        entities = []
        
        # 分词和词性标注
        words = pseg.cut(question)
        
        for word, flag in words:
            # 根据词性标注识别实体
            if flag.startswith('n'):  # 名词
                entity_type = self.classify_entity(word)
                if entity_type:
                    entities.append({
                        'text': word,
                        'type': entity_type,
                        'position': question.find(word),
                        'confidence': 0.8
                    })
        
        # 使用正则表达式识别错误代码
        for pattern in self.entity_types['error_code']:
            matches = re.finditer(pattern, question)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'type': 'error_code',
                    'position': match.start(),
                    'confidence': 0.9
                })
        
        # 使用词典匹配识别专业术语
        for entity_type, terms in self.entity_types.items():
            if entity_type == 'error_code':
                continue
            
            for term in terms:
                if term in question:
                    entities.append({
                        'text': term,
                        'type': entity_type,
                        'position': question.find(term),
                        'confidence': 0.7
                    })
        
        return entities
    
    def classify_entity(self, word: str) -> Optional[str]:
        """分类实体类型"""
        word_lower = word.lower()
        
        # 设备类型
        if any(equipment in word for equipment in ['机床', '设备', '机器', '系统']):
            return 'equipment'
        
        # 组件类型
        if any(component in word for component in ['主轴', '轴承', '电机', '驱动器', '传感器']):
            return 'component'
        
        # 故障类型
        if any(fault in word for fault in ['振动', '噪音', '过热', '漏油', '卡死']):
            return 'fault'
        
        # 品牌类型
        if any(brand in word for brand in ['FANUC', '西门子', '发那科', '三菱']):
            return 'brand'
        
        return None
    
    def extract_intent(self, question: str) -> Dict[str, Any]:
        """提取用户意图"""
        intent = {
            'action': 'query',
            'target': None,
            'constraints': []
        }
        
        # 分析动作意图
        action_keywords = {
            'diagnose': ['诊断', '分析', '检查', '排查'],
            'solve': ['解决', '修复', '修理', '维修'],
            'prevent': ['预防', '避免', '防止'],
            'explain': ['解释', '说明', '介绍', '什么是']
        }
        
        for action, keywords in action_keywords.items():
            if any(keyword in question for keyword in keywords):
                intent['action'] = action
                break
        
        # 分析目标对象
        entities = self.extract_entities(question)
        if entities:
            intent['target'] = entities[0]  # 取第一个实体作为主要目标
        
        # 分析约束条件
        constraint_patterns = [
            r'在(.+?)情况下',
            r'当(.+?)时',
            r'如果(.+?)',
            r'(.+?)的时候'
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, question)
            intent['constraints'].extend(matches)
        
        return intent
    
    def generate_cypher_query(self, question_analysis: Dict[str, Any]) -> str:
        """根据问题分析生成Cypher查询"""
        question_type = question_analysis['question_type']
        entities = question_analysis['entities']
        intent = question_analysis['intent']
        
        if question_type == 'fault_diagnosis':
            return self._generate_diagnosis_query(entities, intent)
        elif question_type == 'solution_query':
            return self._generate_solution_query(entities, intent)
        elif question_type == 'cause_analysis':
            return self._generate_cause_query(entities, intent)
        elif question_type == 'component_query':
            return self._generate_component_query(entities, intent)
        else:
            return self._generate_general_query(entities, intent)
    
    def _generate_diagnosis_query(self, entities: List[Dict], intent: Dict) -> str:
        """生成故障诊断查询"""
        if not entities:
            return "MATCH (f:Fault) RETURN f.name as fault LIMIT 10"
        
        # 查找与故障相关的解决方案
        entity_texts = [e['text'] for e in entities if e['type'] in ['fault', 'error_code']]
        
        if entity_texts:
            conditions = []
            for text in entity_texts:
                conditions.append(f"f.name CONTAINS '{text}' OR ec.code CONTAINS '{text}'")
            
            condition_str = " OR ".join(conditions)
            
            return f"""
            MATCH (f:Fault)-[:cause]->(sol:Solution)
            OPTIONAL MATCH (ec:ErrorCode)-[:indicate]->(f)
            WHERE {condition_str}
            RETURN f.name as fault, sol.name as solution, ec.code as error_code
            LIMIT 10
            """
        
        return "MATCH (f:Fault)-[:cause]->(sol:Solution) RETURN f.name as fault, sol.name as solution LIMIT 10"
    
    def _generate_solution_query(self, entities: List[Dict], intent: Dict) -> str:
        """生成解决方案查询"""
        if not entities:
            return "MATCH (sol:Solution) RETURN sol.name as solution LIMIT 10"
        
        entity_texts = [e['text'] for e in entities]
        conditions = []
        
        for text in entity_texts:
            conditions.append(f"f.name CONTAINS '{text}' OR c.name CONTAINS '{text}' OR e.name CONTAINS '{text}'")
        
        if conditions:
            condition_str = " OR ".join(conditions)
            
            return f"""
            MATCH (f:Fault)-[:cause]->(sol:Solution)
            OPTIONAL MATCH (c:Component)-[:belong_to]->(e:Equipment)
            WHERE {condition_str}
            RETURN sol.name as solution, f.name as fault, c.name as component, e.name as equipment
            LIMIT 10
            """
        
        return "MATCH (sol:Solution) RETURN sol.name as solution LIMIT 10"
    
    def _generate_cause_query(self, entities: List[Dict], intent: Dict) -> str:
        """生成原因分析查询"""
        if not entities:
            return "MATCH (f:Fault) RETURN f.name as fault LIMIT 10"
        
        entity_texts = [e['text'] for e in entities if e['type'] == 'fault']
        
        if entity_texts:
            conditions = []
            for text in entity_texts:
                conditions.append(f"f.name CONTAINS '{text}'")
            
            condition_str = " OR ".join(conditions)
            
            return f"""
            MATCH (f:Fault)-[:cause]->(sol:Solution)
            WHERE {condition_str}
            RETURN f.name as fault, sol.name as cause, sol.description as explanation
            LIMIT 10
            """
        
        return "MATCH (f:Fault) RETURN f.name as fault LIMIT 10"
    
    def _generate_component_query(self, entities: List[Dict], intent: Dict) -> str:
        """生成组件查询"""
        if not entities:
            return "MATCH (c:Component) RETURN c.name as component LIMIT 10"
        
        entity_texts = [e['text'] for e in entities if e['type'] in ['component', 'equipment']]
        
        if entity_texts:
            conditions = []
            for text in entity_texts:
                conditions.append(f"c.name CONTAINS '{text}' OR e.name CONTAINS '{text}'")
            
            condition_str = " OR ".join(conditions)
            
            return f"""
            MATCH (c:Component)-[:belong_to]->(e:Equipment)
            WHERE {condition_str}
            RETURN c.name as component, e.name as equipment, c.type as component_type
            LIMIT 10
            """
        
        return "MATCH (c:Component)-[:belong_to]->(e:Equipment) RETURN c.name as component, e.name as equipment LIMIT 10"
    
    def _generate_general_query(self, entities: List[Dict], intent: Dict) -> str:
        """生成通用查询"""
        if not entities:
            return "MATCH (n) RETURN labels(n) as type, count(n) as count"
        
        # 根据实体类型生成查询
        entity_types = [e['type'] for e in entities]
        
        if 'fault' in entity_types:
            return "MATCH (f:Fault) RETURN f.name as fault LIMIT 10"
        elif 'component' in entity_types:
            return "MATCH (c:Component) RETURN c.name as component LIMIT 10"
        elif 'equipment' in entity_types:
            return "MATCH (e:Equipment) RETURN e.name as equipment LIMIT 10"
        else:
            return "MATCH (n) RETURN labels(n) as type, count(n) as count"
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """完整的问题分析"""
        logger.info(f"分析问题: {question}")
        
        # 分析问题类型
        type_analysis = self.analyze_question_type(question)
        
        # 提取实体
        entities = self.extract_entities(question)
        
        # 提取意图
        intent = self.extract_intent(question)
        
        # 生成Cypher查询
        cypher_query = self.generate_cypher_query({
            'question_type': type_analysis['question_type'],
            'entities': entities,
            'intent': intent
        })
        
        analysis_result = {
            'original_question': question,
            'question_type': type_analysis['question_type'],
            'confidence': type_analysis['confidence'],
            'entities': entities,
            'intent': intent,
            'cypher_query': cypher_query,
            'analysis_time': datetime.now().isoformat()
        }
        
        logger.info(f"问题分析完成: {type_analysis['question_type']} (置信度: {type_analysis['confidence']:.2f})")
        
        return analysis_result

def main():
    """主函数"""
    analyzer = QuestionAnalyzer()
    
    # 测试问题
    test_questions = [
        "数控机床主轴出现异常振动怎么办？",
        "FANUC系统报警ALM401的解决方法是什么？",
        "伺服电机过热的原因是什么？",
        "如何预防主轴轴承磨损？",
        "机床的润滑系统有哪些组件？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        analysis = analyzer.analyze_question(question)
        
        print(f"问题类型: {analysis['question_type']}")
        print(f"置信度: {analysis['confidence']:.2f}")
        print("识别实体:")
        for entity in analysis['entities']:
            print(f"  {entity['text']} ({entity['type']})")
        print(f"用户意图: {analysis['intent']['action']}")
        print(f"Cypher查询: {analysis['cypher_query']}")

if __name__ == "__main__":
    main()