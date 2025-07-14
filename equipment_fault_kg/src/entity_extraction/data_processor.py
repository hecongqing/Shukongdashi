import json
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    name: str
    type: str
    start: int
    end: int

@dataclass
class Relation:
    head: Entity
    tail: Entity
    relation_type: str

class DataProcessor:
    """数据预处理类，用于处理实体抽取和关系抽取的训练数据"""
    
    def __init__(self):
        self.entity_types = {
            "部件单元": "COMPONENT",
            "性能表征": "PERFORMANCE", 
            "故障状态": "FAULT_STATE",
            "检测工具": "DETECTION_TOOL"
        }
        
        self.relation_types = {
            "部件故障": "COMPONENT_FAULT",
            "性能故障": "PERFORMANCE_FAULT", 
            "检测工具": "DETECTION_TOOL",
            "组成": "COMPOSITION"
        }
    
    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSON格式的训练数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def extract_entities_from_spo(self, spo_list: List[Dict]) -> List[Entity]:
        """从SPO列表中提取实体"""
        entities = []
        entity_set = set()  # 用于去重
        
        for spo in spo_list:
            # 提取头实体
            head = spo.get('h', {})
            if head:
                head_name = head.get('name', '')
                head_pos = head.get('pos', [0, 0])
                head_key = f"{head_name}_{head_pos[0]}_{head_pos[1]}"
                
                if head_key not in entity_set:
                    entities.append(Entity(
                        name=head_name,
                        type=self._infer_entity_type(head_name, spo.get('relation', '')),
                        start=head_pos[0],
                        end=head_pos[1]
                    ))
                    entity_set.add(head_key)
            
            # 提取尾实体
            tail = spo.get('t', {})
            if tail:
                tail_name = tail.get('name', '')
                tail_pos = tail.get('pos', [0, 0])
                tail_key = f"{tail_name}_{tail_pos[0]}_{tail_pos[1]}"
                
                if tail_key not in entity_set:
                    entities.append(Entity(
                        name=tail_name,
                        type=self._infer_entity_type(tail_name, spo.get('relation', '')),
                        start=tail_pos[0],
                        end=tail_pos[1]
                    ))
                    entity_set.add(tail_key)
        
        return entities
    
    def _infer_entity_type(self, entity_name: str, relation: str) -> str:
        """根据实体名称和关系推断实体类型"""
        # 基于关系类型推断
        if relation == "部件故障":
            if "故障" in entity_name or "损坏" in entity_name or "异常" in entity_name:
                return "FAULT_STATE"
            else:
                return "COMPONENT"
        elif relation == "性能故障":
            if "故障" in entity_name or "异常" in entity_name or "不良" in entity_name:
                return "FAULT_STATE"
            else:
                return "PERFORMANCE"
        elif relation == "检测工具":
            return "DETECTION_TOOL"
        elif relation == "组成":
            return "COMPONENT"
        
        # 基于实体名称关键词推断
        component_keywords = ["泵", "器", "机", "阀", "管", "盖", "锁", "铰链", "变压器", "分离器"]
        performance_keywords = ["压力", "转速", "温度", "液面", "电流", "电压", "功率"]
        fault_keywords = ["故障", "损坏", "异常", "不良", "抖动", "松旷", "漏油", "断裂", "变形", "卡滞"]
        tool_keywords = ["测试仪", "保护器", "互感器", "检测器", "传感器"]
        
        for keyword in component_keywords:
            if keyword in entity_name:
                return "COMPONENT"
        
        for keyword in performance_keywords:
            if keyword in entity_name:
                return "PERFORMANCE"
        
        for keyword in fault_keywords:
            if keyword in entity_name:
                return "FAULT_STATE"
        
        for keyword in tool_keywords:
            if keyword in entity_name:
                return "DETECTION_TOOL"
        
        # 默认返回部件单元
        return "COMPONENT"
    
    def convert_to_ner_format(self, data: List[Dict]) -> List[Dict]:
        """转换为NER训练格式"""
        ner_data = []
        
        for sample in data:
            text = sample.get('text', '')
            spo_list = sample.get('spo_list', [])
            
            # 提取所有实体
            entities = self.extract_entities_from_spo(spo_list)
            
            # 创建标签序列
            labels = ['O'] * len(text)
            
            # 标记实体位置
            for entity in entities:
                if entity.start < len(text) and entity.end <= len(text):
                    # 标记B-标签
                    labels[entity.start] = f"B-{entity.type}"
                    # 标记I-标签
                    for i in range(entity.start + 1, entity.end):
                        if i < len(labels):
                            labels[i] = f"I-{entity.type}"
            
            ner_data.append({
                'id': sample.get('ID', ''),
                'text': text,
                'labels': labels,
                'entities': entities
            })
        
        return ner_data
    
    def convert_to_re_format(self, data: List[Dict]) -> List[Dict]:
        """转换为关系抽取训练格式"""
        re_data = []
        
        for sample in data:
            text = sample.get('text', '')
            spo_list = sample.get('spo_list', [])
            
            # 提取所有实体
            entities = self.extract_entities_from_spo(spo_list)
            
            # 提取关系
            relations = []
            for spo in spo_list:
                head = spo.get('h', {})
                tail = spo.get('t', {})
                relation = spo.get('relation', '')
                
                if head and tail and relation:
                    # 找到对应的实体
                    head_entity = None
                    tail_entity = None
                    
                    for entity in entities:
                        if (entity.name == head.get('name', '') and 
                            entity.start == head.get('pos', [0, 0])[0] and
                            entity.end == head.get('pos', [0, 0])[1]):
                            head_entity = entity
                        
                        if (entity.name == tail.get('name', '') and 
                            entity.start == tail.get('pos', [0, 0])[0] and
                            entity.end == tail.get('pos', [0, 0])[1]):
                            tail_entity = entity
                    
                    if head_entity and tail_entity:
                        relations.append({
                            'head': head_entity,
                            'tail': tail_entity,
                            'relation_type': relation
                        })
            
            re_data.append({
                'id': sample.get('ID', ''),
                'text': text,
                'entities': entities,
                'relations': relations
            })
        
        return re_data
    
    def save_processed_data(self, data: List[Dict], output_path: str):
        """保存处理后的数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(data)} samples to {output_path}")
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """划分训练集、验证集和测试集"""
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data