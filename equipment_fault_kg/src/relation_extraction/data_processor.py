import json
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RelationSample:
    text: str
    head_entity: str
    tail_entity: str
    relation_type: str
    head_pos: List[int]
    tail_pos: List[int]

class RelationDataProcessor:
    """关系抽取数据预处理类"""
    
    def __init__(self):
        self.relation_types = {
            "部件故障": "COMPONENT_FAULT",
            "性能故障": "PERFORMANCE_FAULT", 
            "检测工具": "DETECTION_TOOL_REL",
            "组成": "COMPOSITION"
        }
        
        self.entity_types = {
            "部件单元": "COMPONENT",
            "性能表征": "PERFORMANCE", 
            "故障状态": "FAULT_STATE",
            "检测工具": "DETECTION_TOOL"
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
    
    def extract_relation_samples(self, data: List[Dict]) -> List[RelationSample]:
        """从原始数据中提取关系样本"""
        samples = []
        
        for sample in data:
            text = sample.get('text', '')
            spo_list = sample.get('spo_list', [])
            
            for spo in spo_list:
                head = spo.get('h', {})
                tail = spo.get('t', {})
                relation = spo.get('relation', '')
                
                if head and tail and relation:
                    head_name = head.get('name', '')
                    tail_name = tail.get('name', '')
                    head_pos = head.get('pos', [])
                    tail_pos = tail.get('pos', [])
                    
                    if head_name and tail_name and len(head_pos) == 2 and len(tail_pos) == 2:
                        samples.append(RelationSample(
                            text=text,
                            head_entity=head_name,
                            tail_entity=tail_name,
                            relation_type=relation,
                            head_pos=head_pos,
                            tail_pos=tail_pos
                        ))
        
        logger.info(f"Extracted {len(samples)} relation samples")
        return samples
    
    def convert_to_classification_format(self, samples: List[RelationSample]) -> List[Dict]:
        """转换为分类任务格式"""
        classification_data = []
        
        for sample in samples:
            # 创建正样本
            classification_data.append({
                'text': sample.text,
                'head_entity': sample.head_entity,
                'tail_entity': sample.tail_entity,
                'relation_type': sample.relation_type,
                'head_pos': sample.head_pos,
                'tail_pos': sample.tail_pos,
                'label': 1  # 正样本
            })
        
        logger.info(f"Converted {len(classification_data)} samples to classification format")
        return classification_data
    
    def create_negative_samples(self, samples: List[RelationSample], 
                               negative_ratio: float = 1.0) -> List[Dict]:
        """创建负样本"""
        negative_samples = []
        
        # 收集所有实体对
        entity_pairs = set()
        for sample in samples:
            entity_pairs.add((sample.head_entity, sample.tail_entity, sample.relation_type))
        
        # 创建负样本
        for i, sample in enumerate(samples):
            # 为每个正样本创建负样本
            for j in range(int(negative_ratio)):
                # 随机选择不同的关系类型
                import random
                other_relations = [rel for rel in self.relation_types.keys() 
                                 if rel != sample.relation_type]
                
                if other_relations:
                    random_relation = random.choice(other_relations)
                    negative_samples.append({
                        'text': sample.text,
                        'head_entity': sample.head_entity,
                        'tail_entity': sample.tail_entity,
                        'relation_type': random_relation,
                        'head_pos': sample.head_pos,
                        'tail_pos': sample.tail_pos,
                        'label': 0  # 负样本
                    })
        
        logger.info(f"Created {len(negative_samples)} negative samples")
        return negative_samples
    
    def create_span_extraction_format(self, samples: List[RelationSample]) -> List[Dict]:
        """转换为span抽取格式"""
        span_data = []
        
        for sample in samples:
            # 为每种关系类型创建样本
            span_data.append({
                'text': sample.text,
                'head_entity': sample.head_entity,
                'tail_entity': sample.tail_entity,
                'relation_type': sample.relation_type,
                'head_pos': sample.head_pos,
                'tail_pos': sample.tail_pos,
                'head_start': sample.head_pos[0],
                'head_end': sample.head_pos[1],
                'tail_start': sample.tail_pos[0],
                'tail_end': sample.tail_pos[1]
            })
        
        logger.info(f"Converted {len(span_data)} samples to span extraction format")
        return span_data
    
    def create_sequence_labeling_format(self, samples: List[RelationSample]) -> List[Dict]:
        """转换为序列标注格式"""
        seq_data = []
        
        for sample in samples:
            text = sample.text
            labels = ['O'] * len(text)
            
            # 标记头实体
            head_start, head_end = sample.head_pos
            if head_start < len(text) and head_end <= len(text):
                labels[head_start] = 'B-H'
                for i in range(head_start + 1, head_end):
                    if i < len(labels):
                        labels[i] = 'I-H'
            
            # 标记尾实体
            tail_start, tail_end = sample.tail_pos
            if tail_start < len(text) and tail_end <= len(text):
                labels[tail_start] = 'B-T'
                for i in range(tail_start + 1, tail_end):
                    if i < len(labels):
                        labels[i] = 'I-T'
            
            seq_data.append({
                'text': text,
                'labels': labels,
                'relation_type': sample.relation_type,
                'head_entity': sample.head_entity,
                'tail_entity': sample.tail_entity
            })
        
        logger.info(f"Converted {len(seq_data)} samples to sequence labeling format")
        return seq_data
    
    def save_data(self, data: List[Dict], output_path: str):
        """保存数据"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"Saved data to {output_path}")
    
    def get_relation_statistics(self, samples: List[RelationSample]) -> Dict:
        """获取关系统计信息"""
        relation_counts = {}
        entity_pair_counts = {}
        
        for sample in samples:
            relation_counts[sample.relation_type] = relation_counts.get(sample.relation_type, 0) + 1
            
            entity_pair = (sample.head_entity, sample.tail_entity)
            entity_pair_counts[entity_pair] = entity_pair_counts.get(entity_pair, 0) + 1
        
        return {
            'relation_counts': relation_counts,
            'entity_pair_counts': dict(sorted(entity_pair_counts.items(), 
                                            key=lambda x: x[1], reverse=True)[:20])
        }
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分割数据集"""
        total = len(data)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data

if __name__ == "__main__":
    # 测试数据处理器
    processor = RelationDataProcessor()
    
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
    
    # 提取关系样本
    relation_samples = processor.extract_relation_samples(sample_data)
    print(f"提取了 {len(relation_samples)} 个关系样本")
    
    # 转换为分类格式
    classification_data = processor.convert_to_classification_format(relation_samples)
    print(f"转换为分类格式: {len(classification_data)} 个样本")
    
    # 创建负样本
    negative_samples = processor.create_negative_samples(relation_samples, negative_ratio=0.5)
    print(f"创建负样本: {len(negative_samples)} 个样本")
    
    # 统计信息
    stats = processor.get_relation_statistics(relation_samples)
    print("\n统计信息:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))