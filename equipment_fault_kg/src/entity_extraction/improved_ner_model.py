"""
改进的装备制造领域命名实体识别模型

修复了标签对齐、分词处理和实体提取的问题
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from loguru import logger
import re


class ImprovedNERDataset(Dataset):
    """改进的装备制造领域NER数据集"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 实体标签映射
        self.label2id = {
            'O': 0,
            'B-COMPONENT': 1, 'I-COMPONENT': 2,
            'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
            'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
            'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_seq = self.labels[idx]
        
        # 确保标签序列长度与文本长度一致
        if len(label_seq) != len(text):
            # 如果长度不匹配，用'O'填充
            if len(label_seq) < len(text):
                label_seq = label_seq + ['O'] * (len(text) - len(label_seq))
            else:
                label_seq = label_seq[:len(text)]
        
        # 使用改进的标签对齐方法
        tokens, label_ids = self.align_tokens_and_labels(text, label_seq)
        
        # 截断到最大长度
        if len(tokens) > self.max_length - 2:  # 保留[CLS]和[SEP]
            tokens = tokens[:self.max_length - 2]
            label_ids = label_ids[:self.max_length - 2]
        
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [self.label2id['O']] + label_ids + [self.label2id['O']]
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)
            label_ids.extend([self.label2id['O']] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def align_tokens_and_labels(self, text: str, labels: List[str]) -> Tuple[List[str], List[int]]:
        """改进的标签对齐方法"""
        tokens = []
        label_ids = []
        
        # 使用BERT tokenizer进行分词
        bert_tokens = self.tokenizer.tokenize(text)
        
        # 创建字符到token的映射
        char_to_token = []
        current_token_idx = 0
        
        for char in text:
            # 找到当前字符对应的token
            char_tokens = self.tokenizer.tokenize(char)
            if char_tokens:
                char_to_token.append(current_token_idx)
                current_token_idx += len(char_tokens)
            else:
                # 如果字符无法分词，使用[UNK]
                char_to_token.append(current_token_idx)
                current_token_idx += 1
        
        # 对齐标签
        for i, token in enumerate(bert_tokens):
            # 找到这个token对应的原始字符位置
            char_pos = None
            for j, char_token_idx in enumerate(char_to_token):
                if char_token_idx == i:
                    char_pos = j
                    break
            
            if char_pos is not None and char_pos < len(labels):
                label = labels[char_pos]
            else:
                label = 'O'
            
            tokens.append(token)
            label_ids.append(self.label2id.get(label, 0))
        
        return tokens, label_ids


class ImprovedNERModel(nn.Module):
    """改进的装备制造领域NER模型"""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标签
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss, logits


class ImprovedNERPredictor:
    """改进的NER预测器"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 标签映射
        self.label2id = {
            'O': 0,
            'B-COMPONENT': 1, 'I-COMPONENT': 2,
            'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
            'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
            'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 实体类型映射
        self.entity_type_mapping = {
            'COMPONENT': '部件单元',
            'PERFORMANCE': '性能表征',
            'FAULT_STATE': '故障状态',
            'DETECTION_TOOL': '检测工具'
        }
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取标签数量
            num_labels = len(self.label2id)
            
            # 加载模型
            self.model = ImprovedNERModel('bert-base-chinese', num_labels)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # 预处理文本
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:  # 保留[CLS]和[SEP]的位置
            tokens = tokens[:510]
        
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        pred_labels = predictions[0].cpu().numpy()
        
        # 提取实体
        entities = self._extract_entities(text, tokens[1:-1], pred_labels[1:-1])  # 去掉[CLS]和[SEP]
        
        return entities
    
    def _extract_entities(self, text: str, tokens: List[str], pred_labels: List[int]) -> List[Dict]:
        """改进的实体提取方法"""
        entities = []
        current_entity = None
        
        # 创建token到原始文本的映射
        token_to_text = self._create_token_to_text_mapping(text, tokens)
        
        for i, (token, label_id) in enumerate(zip(tokens, pred_labels)):
            label = self.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # 去掉'B-'前缀
                current_entity = {
                    'name': token_to_text.get(i, token),
                    'type': self.entity_type_mapping.get(entity_type, entity_type),
                    'start_pos': i,
                    'end_pos': i + 1,
                    'original_type': entity_type
                }
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]  # 去掉'I-'前缀
                # 继续当前实体（只有当类型匹配时）
                if entity_type == current_entity['original_type']:
                    current_entity['name'] += token_to_text.get(i, token)
                    current_entity['end_pos'] = i + 1
                else:
                    # 类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            
            elif label == 'O':
                # 非实体标签，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        # 后处理：清理和合并实体
        entities = self._post_process_entities(entities, text)
        
        return entities
    
    def _create_token_to_text_mapping(self, text: str, tokens: List[str]) -> Dict[int, str]:
        """创建token到原始文本的映射"""
        mapping = {}
        text_idx = 0
        
        for token_idx, token in enumerate(tokens):
            # 跳过特殊标记
            if token.startswith('[') and token.endswith(']'):
                continue
            
            # 找到token在原始文本中的对应位置
            if text_idx < len(text):
                # 对于中文字符，通常一个token对应一个字符
                if token in text[text_idx:]:
                    pos = text.find(token, text_idx)
                    if pos != -1:
                        mapping[token_idx] = token
                        text_idx = pos + len(token)
                else:
                    # 处理分词后的子词
                    mapping[token_idx] = token
        
        return mapping
    
    def _post_process_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """后处理实体：清理和合并"""
        processed_entities = []
        
        for entity in entities:
            # 清理实体名称
            name = entity['name']
            
            # 移除标点符号
            name = re.sub(r'[，。！？；：""''（）【】]', '', name)
            
            # 移除空白字符
            name = name.strip()
            
            # 过滤掉太短的实体
            if len(name) >= 1:
                entity['name'] = name
                processed_entities.append(entity)
        
        # 合并相邻的同类型实体
        merged_entities = []
        i = 0
        while i < len(processed_entities):
            current = processed_entities[i]
            merged = False
            
            # 检查是否可以与下一个实体合并
            if i + 1 < len(processed_entities):
                next_entity = processed_entities[i + 1]
                if (current['type'] == next_entity['type'] and 
                    current['end_pos'] == next_entity['start_pos']):
                    # 合并实体
                    current['name'] += next_entity['name']
                    current['end_pos'] = next_entity['end_pos']
                    i += 1  # 跳过下一个实体
                    merged = True
            
            if not merged:
                merged_entities.append(current)
            i += 1
        
        return merged_entities


class ImprovedEntityExtractor:
    """改进的实体抽取器"""
    
    def __init__(self, model_path: str):
        self.predictor = ImprovedNERPredictor(model_path)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体"""
        entities = self.predictor.predict(text)
        return entities
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取实体"""
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results


def test_improved_ner():
    """测试改进的NER模型"""
    text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    
    print(f"测试文本: {text}")
    print(f"文本长度: {len(text)} 字符")
    
    # 创建改进的预测器（需要训练好的模型）
    try:
        model_path = './models/best_ner_model.pth'
        if Path(model_path).exists():
            extractor = ImprovedEntityExtractor(model_path)
            entities = extractor.extract_entities(text)
            
            print("\n抽取的实体:")
            for entity in entities:
                print(f"  - {entity['name']} [{entity['type']}]")
        else:
            print(f"\n模型文件不存在: {model_path}")
            print("请先训练模型再测试")
    except Exception as e:
        print(f"\n测试失败: {e}")


if __name__ == "__main__":
    test_improved_ner()