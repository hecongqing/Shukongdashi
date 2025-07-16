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
    start_pos: int
    end_pos: int

@dataclass
class Relation:
    head_entity: Entity
    tail_entity: Entity
    relation_type: str

class EntityDataProcessor:
    """实体抽取数据预处理类"""
    
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
            "检测工具": "DETECTION_TOOL_REL",
            "组成": "COMPOSITION"
        }
    
    def load_data(self, file_path: str) -> List[Dict]:
        """加载训练数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def extract_entities_from_spo(self, spo_list: List[Dict], text: str) -> List[Entity]:
        """从SPO列表中提取实体"""
        entities = []
        entity_set = set()  # 用于去重
        
        for spo in spo_list:
            # 提取头实体
            head = spo.get('h', {})
            if head:
                head_name = head.get('name', '')
                head_pos = head.get('pos', [])
                if head_name and len(head_pos) == 2:
                    entity_key = (head_name, head_pos[0], head_pos[1])
                    if entity_key not in entity_set:
                        # 根据关系类型推断实体类型
                        relation = spo.get('relation', '')
                        head_type = self._infer_entity_type(head_name, relation, is_head=True)
                        entities.append(Entity(
                            name=head_name,
                            type=head_type,
                            start_pos=head_pos[0],
                            end_pos=head_pos[1]
                        ))
                        entity_set.add(entity_key)
            
            # 提取尾实体
            tail = spo.get('t', {})
            if tail:
                tail_name = tail.get('name', '')
                tail_pos = tail.get('pos', [])
                if tail_name and len(tail_pos) == 2:
                    entity_key = (tail_name, tail_pos[0], tail_pos[1])
                    if entity_key not in entity_set:
                        # 根据关系类型推断实体类型
                        relation = spo.get('relation', '')
                        tail_type = self._infer_entity_type(tail_name, relation, is_head=False)
                        entities.append(Entity(
                            name=tail_name,
                            type=tail_type,
                            start_pos=tail_pos[0],
                            end_pos=tail_pos[1]
                        ))
                        entity_set.add(entity_key)
        
        return entities
    
    def _infer_entity_type(self, entity_name: str, relation: str, is_head: bool) -> str:
        """根据实体名称和关系类型推断实体类型"""
        # 根据关系类型推断
        if relation == "部件故障":
            if is_head:
                return "COMPONENT"
            else:
                return "FAULT_STATE"
        elif relation == "性能故障":
            if is_head:
                return "PERFORMANCE"
            else:
                return "FAULT_STATE"
        elif relation == "检测工具":
            if is_head:
                return "DETECTION_TOOL"
            else:
                return "PERFORMANCE"
        elif relation == "组成":
            return "COMPONENT"
        
        # 根据实体名称特征推断
        if any(keyword in entity_name for keyword in ["泵", "器", "机", "阀", "管", "箱", "盖", "锁", "铰链"]):
            return "COMPONENT"
        elif any(keyword in entity_name for keyword in ["压力", "温度", "转速", "液面", "电流", "电压"]):
            return "PERFORMANCE"
        elif any(keyword in entity_name for keyword in ["漏", "断", "变形", "卡", "松", "抖动", "不良"]):
            return "FAULT_STATE"
        elif any(keyword in entity_name for keyword in ["测试仪", "互感器", "保护器", "检测器"]):
            return "DETECTION_TOOL"
        
        # 默认类型
        return "COMPONENT"
    
    def convert_to_ner_format(self, data: List[Dict]) -> List[Dict]:
        """转换为NER训练格式"""
        ner_data = []
        
        for sample in data:
            text = sample.get('text', '')
            spo_list = sample.get('spo_list', [])
            
            # 提取实体
            entities = self.extract_entities_from_spo(spo_list, text)
            
            # 创建标签序列
            labels = ['O'] * len(text)
            
            # 标记实体位置
            for entity in entities:
                if entity.start_pos < len(text) and entity.end_pos <= len(text):
                    # 标记B-标签
                    labels[entity.start_pos] = f'B-{entity.type}'
                    # 标记I-标签
                    for i in range(entity.start_pos + 1, entity.end_pos):
                        if i < len(labels):
                            labels[i] = f'I-{entity.type}'
            
            ner_data.append({
                'text': text,
                'labels': labels,
                'entities': entities,
                'spo_list': spo_list
            })
        
        logger.info(f"Converted {len(ner_data)} samples to NER format")
        return ner_data

    # ------------------------------------------------------------------
    # 额外工具：清洗和验证 NER 标注
    # ------------------------------------------------------------------

    def clean_ner_data(self, data: List[Dict]) -> List[Dict]:
        """验证并清洗 BIO 序列，确保长度一致且 I- 标签前面有相同类型的 B-/I-。

        返回清洗后的数据列表，同时记录纠正次数以便调试。
        """
        fixed_len_mismatch = 0
        fixed_bio_inconsistency = 0

        cleaned = []

        for sample in data:
            text = sample.get('text', '')
            labels: List[str] = sample.get('labels', [])

            # 1. 长度对齐
            if len(labels) != len(text):
                fixed_len_mismatch += 1
                # 截断或填充到文本长度，其余用 'O'
                if len(labels) > len(text):
                    labels = labels[:len(text)]
                else:
                    labels.extend(['O'] * (len(text) - len(labels)))

            # 2. BIO 合规：I-不能出现在序列开始，且前一个必须同类型实体
            prev_label = 'O'
            for idx, lab in enumerate(labels):
                if lab.startswith('I-'):
                    if prev_label == 'O' or (prev_label[2:] != lab[2:]):
                        # 修正为 B-
                        labels[idx] = 'B-' + lab[2:]
                        fixed_bio_inconsistency += 1
                prev_label = labels[idx]

            # 更新 sample
            sample['labels'] = labels
            cleaned.append(sample)

        logger.info(
            f"NER data cleaned: fixed length mismatches={fixed_len_mismatch}, BIO inconsistencies={fixed_bio_inconsistency}"
        )

        return cleaned

    def save_ner_data(self, data: List[Dict], output_path: str):
        """保存清洗后的 NER 数据到指定路径"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Saved NER data to {output_path}")
    
    def get_entity_statistics(self, data: List[Dict]) -> Dict:
        """获取实体统计信息"""
        entity_counts = {}
        relation_counts = {}
        
        for sample in data:
            spo_list = sample.get('spo_list', [])
            for spo in spo_list:
                relation = spo.get('relation', '')
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                
                # 统计头实体
                head = spo.get('h', {})
                if head:
                    head_name = head.get('name', '')
                    if head_name:
                        entity_counts[head_name] = entity_counts.get(head_name, 0) + 1
                
                # 统计尾实体
                tail = spo.get('t', {})
                if tail:
                    tail_name = tail.get('name', '')
                    if tail_name:
                        entity_counts[tail_name] = entity_counts.get(tail_name, 0) + 1
        
        return {
            'entity_counts': dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'relation_counts': relation_counts
        }

if __name__ == "__main__":
    # 测试数据处理器
    processor = EntityDataProcessor()
    
    # 示例数据
    sample_data = [
        {
            "ID": "AT0001",
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "spo_list": [
                {"h": {"name": "发动机盖", "pos": [14, 18]}, "t": {"name": "抖动", "pos": [24, 26]}, "relation": "部件故障"},
                {"h": {"name": "发动机盖锁", "pos": [46, 51]}, "t": {"name": "松旷", "pos": [58, 60]}, "relation": "部件故障"},
                {"h": {"name": "发动机盖铰链", "pos": [52, 58]}, "t": {"name": "松旷", "pos": [58, 60]}, "relation": "部件故障"}
            ]
        }
    ]
    
    # 转换为NER格式
    ner_data = processor.convert_to_ner_format(sample_data)
    print("NER格式数据示例:")
    print(json.dumps(ner_data[0], ensure_ascii=False, indent=2))
    
    # 统计信息
    stats = processor.get_entity_statistics(sample_data)
    print("\n统计信息:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))