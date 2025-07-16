import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import logging
from typing import List, Dict, Tuple, Set
import re
import os
import jieba
import string

from .train_ner import NERModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedNERPredictor:
    """改进的NER预测器类"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        
        # 初始化jieba分词
        jieba.initialize()
        
        # 实体合理性检查规则
        self.entity_patterns = {
            'COMPONENT': [
                r'.*[泵器机阀管箱盖锁轮].*',
                r'.*[电动液压气动].*',
                r'.*[传感器控制器].*',
                r'.*铰链.*'
            ],
            'FAULT_STATE': [
                r'.*[漏断裂变形卡死松动抖动].*',
                r'.*[异常故障失效].*',
                r'.*不良.*'
            ],
            'DETECTION_TOOL': [
                r'.*[测试检测].*[仪器表].*',
                r'.*[万用表示波器].*',
                r'.*[互感器保护器].*'
            ],
            'PERFORMANCE': [
                r'.*[压力温度转速液面电流电压].*',
                r'.*[性能参数指标].*'
            ]
        }
        
        # 无效实体过滤规则
        self.invalid_patterns = [
            r'^[，。！？；：""''（）\[\]{}()<>\/\\|+=\-*&^%$#@~`\s]+$',  # 纯标点和空白
            r'^[0-9]+$',  # 纯数字
            r'^[a-zA-Z]+$',  # 纯英文字母
            r'^[的了在是和与或]$',  # 常见虚词
        ]
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 加载标签映射 - 保持与训练时一致
            self.label2id = {
                'O': 0,
                'B-COMPONENT': 1, 'I-COMPONENT': 2,
                'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
                'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
                'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
            
            # 获取标签数量
            num_labels = len(self.label2id)
            
            # 加载模型
            self.model = NERModel('bert-base-chinese', num_labels)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Improved model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # 预处理：使用jieba分词获得词边界信息
        words = list(jieba.cut(text))
        word_positions = []
        current_pos = 0
        
        # 计算每个词的位置
        for word in words:
            start_pos = text.find(word, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(word)
                word_positions.append((word, start_pos, end_pos))
                current_pos = end_pos
            else:
                # 处理特殊情况
                word_positions.append((word, current_pos, current_pos + len(word)))
                current_pos += len(word)
        
        # 字符级标注（用于模型推理）
        tokens = []
        char_to_token = []
        
        # 添加[CLS]标记
        tokens.append('[CLS]')
        
        # 处理文本 - 按字符处理但保留词边界信息
        for i, char in enumerate(text):
            char_start_token = len(tokens)
            
            # 分词
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            
            tokens.extend(sub_tokens)
            char_to_token.append(char_start_token)
        
        # 添加[SEP]标记
        tokens.append('[SEP]')
        
        # 截断处理
        max_length = 512
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        pred_labels = predictions[0].cpu().numpy()
        
        # 提取实体（使用词边界信息）
        entities = self._extract_entities_with_word_boundary(text, pred_labels, char_to_token, word_positions)
        
        # 后处理：过滤和优化实体
        entities = self._post_process_entities(entities, text)
        
        return entities
    
    def _extract_entities_with_word_boundary(self, text: str, pred_labels: List[int], 
                                           char_to_token: List[int], word_positions: List[Tuple]) -> List[Dict]:
        """结合词边界信息提取实体"""
        entities = []
        current_entity = None
        
        for i, char in enumerate(text):
            if i >= len(char_to_token):
                continue
            
            token_idx = char_to_token[i] + 1  # +1是因为第一个token是[CLS]
            
            if token_idx >= len(pred_labels):
                continue
            
            label_id = pred_labels[token_idx]
            label = self.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'name': char,
                    'type': entity_type,
                    'start_pos': i,
                    'end_pos': i + 1
                }
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity['type']:
                    current_entity['name'] += char
                    current_entity['end_pos'] = i + 1
                else:
                    entities.append(current_entity)
                    current_entity = None
            
            elif label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        # 根据词边界调整实体边界
        entities = self._adjust_entity_boundaries(entities, word_positions)
        
        return entities
    
    def _adjust_entity_boundaries(self, entities: List[Dict], word_positions: List[Tuple]) -> List[Dict]:
        """根据词边界调整实体边界"""
        adjusted_entities = []
        
        for entity in entities:
            entity_start = entity['start_pos']
            entity_end = entity['end_pos']
            
            # 找到包含实体的词
            covering_words = []
            for word, word_start, word_end in word_positions:
                if (word_start <= entity_start < word_end) or (word_start < entity_end <= word_end) or \
                   (entity_start <= word_start and entity_end >= word_end):
                    covering_words.append((word, word_start, word_end))
            
            if covering_words:
                # 扩展实体边界到完整词边界
                new_start = min(word_start for _, word_start, _ in covering_words)
                new_end = max(word_end for _, _, word_end in covering_words)
                new_name = "".join(word for word, _, _ in covering_words)
                
                adjusted_entities.append({
                    'name': new_name,
                    'type': entity['type'],
                    'start_pos': new_start,
                    'end_pos': new_end
                })
            else:
                adjusted_entities.append(entity)
        
        return adjusted_entities
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """后处理：过滤和优化实体"""
        filtered_entities = []
        
        for entity in entities:
            # 跳过无效实体
            if self._is_invalid_entity(entity):
                continue
            
            # 验证实体类型合理性
            if not self._is_valid_entity_type(entity):
                # 尝试重新分类
                new_type = self._reclassify_entity(entity)
                if new_type:
                    entity['type'] = new_type
                else:
                    continue
            
            # 去重（避免重复实体）
            if not self._is_duplicate_entity(entity, filtered_entities):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _is_invalid_entity(self, entity: Dict) -> bool:
        """检查实体是否无效"""
        name = entity['name']
        
        # 长度检查
        if len(name) < 1 or len(name) > 20:
            return True
        
        # 使用正则表达式检查无效模式
        for pattern in self.invalid_patterns:
            if re.match(pattern, name):
                return True
        
        return False
    
    def _is_valid_entity_type(self, entity: Dict) -> bool:
        """检查实体类型是否合理"""
        name = entity['name']
        entity_type = entity['type']
        
        if entity_type not in self.entity_patterns:
            return True  # 如果没有定义模式，则认为有效
        
        patterns = self.entity_patterns[entity_type]
        for pattern in patterns:
            if re.match(pattern, name):
                return True
        
        return False
    
    def _reclassify_entity(self, entity: Dict) -> str:
        """重新分类实体"""
        name = entity['name']
        
        # 根据实体内容重新分类
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.match(pattern, name):
                    return entity_type
        
        return None
    
    def _is_duplicate_entity(self, entity: Dict, existing_entities: List[Dict]) -> bool:
        """检查是否为重复实体"""
        for existing in existing_entities:
            if (entity['name'] == existing['name'] and 
                entity['start_pos'] == existing['start_pos'] and
                entity['end_pos'] == existing['end_pos']):
                return True
        return False

class ImprovedEntityExtractor:
    """改进的实体抽取器类 - 高级接口"""
    
    def __init__(self, model_path: str):
        self.predictor = ImprovedNERPredictor(model_path)
        
        # 实体类型映射
        self.entity_type_mapping = {
            'COMPONENT': '部件单元',
            'PERFORMANCE': '性能表征',
            'FAULT_STATE': '故障状态',
            'DETECTION_TOOL': '检测工具'
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体"""
        entities = self.predictor.predict(text)
        
        # 转换格式
        result = []
        for entity in entities:
            result.append({
                'name': entity['name'],
                'type': self.entity_type_mapping.get(entity['type'], entity['type']),
                'start_pos': entity['start_pos'],
                'end_pos': entity['end_pos']
            })
        
        return result
    
    def extract_entities_with_confidence(self, text: str) -> List[Dict]:
        """带置信度的实体抽取"""
        entities = self.extract_entities(text)
        
        # 添加置信度评分
        for entity in entities:
            confidence = self._calculate_confidence(entity, text)
            entity['confidence'] = confidence
        
        # 按置信度排序
        entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return entities
    
    def _calculate_confidence(self, entity: Dict, text: str) -> float:
        """计算实体的置信度"""
        confidence = 0.5  # 基础置信度
        
        # 长度因子
        if 2 <= len(entity['name']) <= 6:
            confidence += 0.2
        elif len(entity['name']) == 1:
            confidence -= 0.3
        
        # 类型匹配度
        name = entity['name']
        entity_type = entity['type']
        
        # 反向映射到英文类型
        eng_type = None
        for eng, chn in self.entity_type_mapping.items():
            if chn == entity_type:
                eng_type = eng
                break
        
        if eng_type and eng_type in self.predictor.entity_patterns:
            patterns = self.predictor.entity_patterns[eng_type]
            for pattern in patterns:
                if re.match(pattern, name):
                    confidence += 0.3
                    break
        
        # 上下文相关性
        context_words = text[max(0, entity['start_pos']-10):entity['end_pos']+10]
        if any(keyword in context_words for keyword in ['故障', '检测', '维修', '异常']):
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))

# 测试函数
def test_improved_extractor():
    """测试改进的实体抽取器"""
    # 这里需要模型路径，实际使用时需要提供
    # extractor = ImprovedEntityExtractor("path/to/model.pth")
    
    test_text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    print(f"输入文本: {test_text}")
    
    # 模拟抽取结果（实际需要训练好的模型）
    mock_entities = [
        {'name': '伺服电机', 'type': '部件单元', 'start_pos': 0, 'end_pos': 4, 'confidence': 0.9},
        {'name': '运行异常', 'type': '故障状态', 'start_pos': 4, 'end_pos': 8, 'confidence': 0.8},
        {'name': '维修人员', 'type': '人员', 'start_pos': 9, 'end_pos': 13, 'confidence': 0.7},
        {'name': '万用表', 'type': '检测工具', 'start_pos': 15, 'end_pos': 18, 'confidence': 0.9},
        {'name': '电路故障', 'type': '故障状态', 'start_pos': 20, 'end_pos': 24, 'confidence': 0.8}
    ]
    
    print("改进后的抽取结果:")
    for entity in mock_entities:
        print(f"  - {entity['name']} [{entity['type']}] (置信度: {entity['confidence']:.2f})")

if __name__ == "__main__":
    test_improved_extractor()