"""
简洁的实体识别器
Simple Entity Recognizer
"""

import re
from typing import List, Set


class SimpleEntityRecognizer:
    """简洁版实体识别器"""
    
    def __init__(self):
        # 预定义的实体词典 - 机床设备相关
        self.equipment_entities = {
            '数控机床', '机床', 'CNC', '车床', '铣床', '磨床', '钻床', '加工中心',
            '主轴', '刀具', '电机', '伺服电机', '步进电机', '丝杠', '导轨',
            '轴承', '联轴器', '变频器', '编码器', '传感器', '冷却系统',
            '润滑系统', '控制系统', '操作面板', '夹具', '工件', '刀架'
        }
        
        self.symptom_entities = {
            '不转', '异响', '振动', '发热', '漏油', '精度差', '卡死', '报警',
            '噪音', '异常', '故障', '停机', '断刀', '撞刀', '过载', '超程'
        }
        
        self.action_entities = {
            '检查', '更换', '维修', '调整', '清洗', '润滑', '校准', '重启'
        }
        
        # 合并所有实体
        self.all_entities = self.equipment_entities | self.symptom_entities | self.action_entities
        
        # 编译正则表达式以提高性能
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译实体匹配模式"""
        # 按长度排序，优先匹配长实体
        sorted_entities = sorted(self.all_entities, key=len, reverse=True)
        # 创建匹配模式
        pattern = '|'.join(re.escape(entity) for entity in sorted_entities)
        self.entity_pattern = re.compile(f'({pattern})')
    
    def recognize(self, text: str) -> List[str]:
        """
        识别文本中的实体
        
        Args:
            text: 输入文本
            
        Returns:
            识别到的实体列表
        """
        if not text:
            return []
        
        # 使用正则表达式匹配实体
        matches = self.entity_pattern.findall(text)
        
        # 去重并保持顺序
        entities = []
        seen = set()
        
        for match in matches:
            if match not in seen:
                entities.append(match)
                seen.add(match)
        
        return entities
    
    def recognize_by_type(self, text: str) -> dict:
        """
        按类型识别实体
        
        Args:
            text: 输入文本
            
        Returns:
            按类型分组的实体字典
        """
        entities = self.recognize(text)
        
        result = {
            'equipment': [],
            'symptom': [],
            'action': []
        }
        
        for entity in entities:
            if entity in self.equipment_entities:
                result['equipment'].append(entity)
            elif entity in self.symptom_entities:
                result['symptom'].append(entity)
            elif entity in self.action_entities:
                result['action'].append(entity)
        
        return result
    
    def add_entity(self, entity: str, entity_type: str = 'equipment'):
        """
        动态添加新实体
        
        Args:
            entity: 实体名称
            entity_type: 实体类型 ('equipment', 'symptom', 'action')
        """
        if entity_type == 'equipment':
            self.equipment_entities.add(entity)
        elif entity_type == 'symptom':
            self.symptom_entities.add(entity)
        elif entity_type == 'action':
            self.action_entities.add(entity)
        
        # 更新总实体集合
        self.all_entities = self.equipment_entities | self.symptom_entities | self.action_entities
        
        # 重新编译模式
        self._compile_patterns()
    
    def get_entity_count(self) -> dict:
        """获取各类型实体数量"""
        return {
            'equipment': len(self.equipment_entities),
            'symptom': len(self.symptom_entities),
            'action': len(self.action_entities),
            'total': len(self.all_entities)
        }