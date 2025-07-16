#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的NER实体抽取器
包含更好的后处理规则和实体验证
"""

import re
from typing import List, Dict, Any, Set
from .deploy_ner import EntityExtractor

class ImprovedEntityExtractor(EntityExtractor):
    """改进的实体抽取器"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
        # 定义过滤规则
        self.punctuation_chars = set('。，；：！？、（）【】""''')
        self.single_char_verbs = set(['使', '用', '检', '测', '修', '维', '换', '装', '拆', '调', '校'])
        self.single_char_entities = set(['泵', '阀', '管', '箱', '盖', '锁', '表', '器', '机'])
        
        # 定义实体验证规则
        self.component_keywords = set([
            '电机', '泵', '阀', '管', '箱', '盖', '锁', '铰链', '轴承', '齿轮', '皮带', '链条',
            '传感器', '控制器', '显示器', '开关', '按钮', '指示灯', '报警器', '保护器',
            '伺服', '变频器', '驱动器', '编码器', '限位开关', '接近开关', '压力开关'
        ])
        
        self.fault_keywords = set([
            '故障', '异常', '损坏', '磨损', '断裂', '变形', '松动', '卡死', '堵塞', '泄漏',
            '过热', '过载', '短路', '断路', '接地', '绝缘', '振动', '噪声', '抖动', '不稳定'
        ])
        
        self.tool_keywords = set([
            '万用表', '示波器', '频谱仪', '振动仪', '温度计', '压力表', '流量计', '转速表',
            '测试仪', '检测器', '分析仪', '诊断仪', '校准器', '校验仪'
        ])
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体（改进版本）"""
        # 调用父类方法获取原始实体
        raw_entities = super().extract_entities(text)
        
        # 应用后处理规则
        processed_entities = self._post_process_entities(raw_entities, text)
        
        # 验证实体合理性
        validated_entities = self._validate_entities(processed_entities)
        
        return validated_entities
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """后处理实体"""
        processed = []
        
        for entity in entities:
            # 1. 过滤标点符号
            if entity['name'] in self.punctuation_chars:
                continue
            
            # 2. 过滤单字符动词（除非是已知的实体）
            if len(entity['name']) == 1 and entity['name'] in self.single_char_verbs:
                if entity['name'] not in self.single_char_entities:
                    continue
            
            # 3. 清理实体文本（移除末尾标点）
            clean_name = entity['name'].rstrip('。，；：！？、')
            if clean_name != entity['name']:
                entity['name'] = clean_name
                # 更新结束位置
                entity['end_pos'] = entity['start_pos'] + len(clean_name)
            
            # 4. 过滤空实体
            if not entity['name'].strip():
                continue
            
            # 5. 合并被错误分割的实体
            processed.append(entity)
        
        # 6. 尝试合并相邻的同类实体
        merged_entities = self._merge_adjacent_entities(processed, text)
        
        return merged_entities
    
    def _merge_adjacent_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """合并相邻的同类实体"""
        if not entities:
            return entities
        
        merged = []
        current = entities[0].copy()
        
        for next_entity in entities[1:]:
            # 检查是否应该合并
            if (current['type'] == next_entity['type'] and 
                next_entity['start_pos'] == current['end_pos']):
                # 合并实体
                current['name'] += next_entity['name']
                current['end_pos'] = next_entity['end_pos']
            else:
                merged.append(current)
                current = next_entity.copy()
        
        merged.append(current)
        return merged
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """验证实体合理性"""
        validated = []
        
        for entity in entities:
            entity_type = entity['type']
            entity_name = entity['name']
            
            # 根据实体类型验证
            if entity_type == '部件单元':
                if self._is_valid_component(entity_name):
                    validated.append(entity)
            elif entity_type == '故障状态':
                if self._is_valid_fault(entity_name):
                    validated.append(entity)
            elif entity_type == '检测工具':
                if self._is_valid_tool(entity_name):
                    validated.append(entity)
            elif entity_type == '性能表征':
                if self._is_valid_performance(entity_name):
                    validated.append(entity)
            else:
                # 未知类型，保留但标记
                validated.append(entity)
        
        return validated
    
    def _is_valid_component(self, name: str) -> bool:
        """验证是否为有效的部件"""
        # 检查是否包含部件关键词
        for keyword in self.component_keywords:
            if keyword in name:
                return True
        
        # 检查是否以常见部件后缀结尾
        if any(name.endswith(suffix) for suffix in ['机', '器', '泵', '阀', '管', '箱', '盖', '锁']):
            return True
        
        # 检查长度（部件名称通常至少2个字符）
        return len(name) >= 2
    
    def _is_valid_fault(self, name: str) -> bool:
        """验证是否为有效的故障状态"""
        # 检查是否包含故障关键词
        for keyword in self.fault_keywords:
            if keyword in name:
                return True
        
        # 检查是否包含故障相关词汇
        fault_patterns = ['异常', '故障', '问题', '损坏', '失效', '不良']
        if any(pattern in name for pattern in fault_patterns):
            return True
        
        return len(name) >= 2
    
    def _is_valid_tool(self, name: str) -> bool:
        """验证是否为有效的检测工具"""
        # 检查是否包含工具关键词
        for keyword in self.tool_keywords:
            if keyword in name:
                return True
        
        # 检查是否以工具后缀结尾
        if any(name.endswith(suffix) for suffix in ['表', '仪', '器', '计']):
            return True
        
        return len(name) >= 2
    
    def _is_valid_performance(self, name: str) -> bool:
        """验证是否为有效的性能表征"""
        # 性能相关词汇
        performance_keywords = ['压力', '温度', '转速', '液面', '电流', '电压', '功率', '效率', '精度']
        
        for keyword in performance_keywords:
            if keyword in name:
                return True
        
        return len(name) >= 2
    
    def extract_entities_with_confidence(self, text: str) -> List[Dict]:
        """抽取实体并返回置信度"""
        entities = self.extract_entities(text)
        
        for entity in entities:
            # 计算置信度（基于验证规则）
            confidence = self._calculate_confidence(entity)
            entity['confidence'] = confidence
        
        # 按置信度排序
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entities
    
    def _calculate_confidence(self, entity: Dict) -> float:
        """计算实体置信度"""
        name = entity['name']
        entity_type = entity['type']
        
        confidence = 0.5  # 基础置信度
        
        # 根据实体类型和名称特征调整置信度
        if entity_type == '部件单元':
            if any(keyword in name for keyword in self.component_keywords):
                confidence += 0.3
            if len(name) >= 3:
                confidence += 0.1
        elif entity_type == '故障状态':
            if any(keyword in name for keyword in self.fault_keywords):
                confidence += 0.3
            if len(name) >= 2:
                confidence += 0.1
        elif entity_type == '检测工具':
            if any(keyword in name for keyword in self.tool_keywords):
                confidence += 0.3
            if len(name) >= 2:
                confidence += 0.1
        elif entity_type == '性能表征':
            if any(keyword in name for keyword in ['压力', '温度', '转速', '电流', '电压']):
                confidence += 0.3
            if len(name) >= 2:
                confidence += 0.1
        
        # 长度惩罚
        if len(name) == 1:
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)