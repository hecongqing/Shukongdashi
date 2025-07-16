"""
基于字符级别的装备制造领域命名实体识别模型

避免BERT分词导致的标签对齐问题，直接使用字符级别的处理
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


class CharLevelNERDataset(Dataset):
    """字符级别NER数据集"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], max_length: int = 512):
        self.texts = texts
        self.labels = labels
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
        
        # 字符到ID的映射
        self.char2id = self._build_char_vocab()
        self.id2char = {v: k for k, v in self.char2id.items()}
    
    def _build_char_vocab(self):
        """构建字符词汇表"""
        char_set = set()
        for text in self.texts:
            char_set.update(text)
        
        # 添加特殊字符
        char_set.add('[PAD]')
        char_set.add('[UNK]')
        char_set.add('[CLS]')
        char_set.add('[SEP]')
        
        return {char: idx for idx, char in enumerate(sorted(char_set))}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_seq = self.labels[idx]
        
        # 确保标签序列长度与文本长度一致
        if len(label_seq) != len(text):
            if len(label_seq) < len(text):
                label_seq = label_seq + ['O'] * (len(text) - len(label_seq))
            else:
                label_seq = label_seq[:len(text)]
        
        # 字符级别的处理
        chars = list(text)
        char_ids = [self.char2id.get(char, self.char2id['[UNK]']) for char in chars]
        label_ids = [self.label2id.get(label, 0) for label in label_seq]
        
        # 截断到最大长度
        if len(char_ids) > self.max_length - 2:  # 保留[CLS]和[SEP]
            char_ids = char_ids[:self.max_length - 2]
            label_ids = label_ids[:self.max_length - 2]
        
        # 添加特殊标记
        char_ids = [self.char2id['[CLS]']] + char_ids + [self.char2id['[SEP]']]
        label_ids = [self.label2id['O']] + label_ids + [self.label2id['O']]
        
        # 创建attention mask
        attention_mask = [1] * len(char_ids)
        
        # 填充到最大长度
        if len(char_ids) < self.max_length:
            padding_length = self.max_length - len(char_ids)
            char_ids.extend([self.char2id['[PAD]']] * padding_length)
            attention_mask.extend([0] * padding_length)
            label_ids.extend([self.label2id['O']] * padding_length)
        
        return {
            'input_ids': torch.tensor(char_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


class CharLevelNERModel(nn.Module):
    """字符级别NER模型"""
    
    def __init__(self, vocab_size: int, num_labels: int, hidden_size: int = 256, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # 字符嵌入层
        self.char_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size // 2,  # 因为是双向的，所以除以2
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 分类层
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 字符嵌入
        embeddings = self.char_embedding(input_ids)
        
        # LSTM处理
        lstm_output, _ = self.lstm(embeddings)
        
        # Dropout
        lstm_output = self.dropout(lstm_output)
        
        # 分类
        logits = self.classifier(lstm_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标签
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss, logits


class CharLevelNERPredictor:
    """字符级别NER预测器"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.char2id = None
        self.id2char = None
        
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
            
            # 加载字符词汇表
            self.char2id = checkpoint['char2id']
            self.id2char = {v: k for k, v in self.char2id.items()}
            
            # 获取模型参数
            vocab_size = len(self.char2id)
            num_labels = len(self.label2id)
            
            # 加载模型
            self.model = CharLevelNERModel(vocab_size, num_labels)
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
        
        # 字符级别的预处理
        chars = list(text)
        char_ids = [self.char2id.get(char, self.char2id['[UNK]']) for char in chars]
        
        # 截断到最大长度
        max_length = 512
        if len(char_ids) > max_length - 2:
            char_ids = char_ids[:max_length - 2]
        
        # 添加特殊标记
        char_ids = [self.char2id['[CLS]']] + char_ids + [self.char2id['[SEP]']]
        attention_mask = [1] * len(char_ids)
        
        # 填充到最大长度
        if len(char_ids) < max_length:
            padding_length = max_length - len(char_ids)
            char_ids.extend([self.char2id['[PAD]']] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # 转换为tensor
        char_ids = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(char_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        pred_labels = predictions[0].cpu().numpy()
        
        # 提取实体
        entities = self._extract_entities(text, chars, pred_labels[1:-1])  # 去掉[CLS]和[SEP]
        
        return entities
    
    def _extract_entities(self, text: str, chars: List[str], pred_labels: List[int]) -> List[Dict]:
        """提取实体"""
        entities = []
        current_entity = None
        
        for i, (char, label_id) in enumerate(zip(chars, pred_labels)):
            label = self.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # 去掉'B-'前缀
                current_entity = {
                    'name': char,
                    'type': self.entity_type_mapping.get(entity_type, entity_type),
                    'start_pos': i,
                    'end_pos': i + 1,
                    'original_type': entity_type
                }
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]  # 去掉'I-'前缀
                # 继续当前实体（只有当类型匹配时）
                if entity_type == current_entity['original_type']:
                    current_entity['name'] += char
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
        
        # 后处理：清理实体
        entities = self._post_process_entities(entities)
        
        return entities
    
    def _post_process_entities(self, entities: List[Dict]) -> List[Dict]:
        """后处理实体：清理和过滤"""
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
        
        return processed_entities


class CharLevelEntityExtractor:
    """字符级别实体抽取器"""
    
    def __init__(self, model_path: str):
        self.predictor = CharLevelNERPredictor(model_path)
    
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


def test_char_level_ner():
    """测试字符级别NER模型"""
    text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    
    print(f"测试文本: {text}")
    print(f"文本长度: {len(text)} 字符")
    
    # 创建字符级别预测器（需要训练好的模型）
    try:
        model_path = './models/char_level_ner_model.pth'
        if Path(model_path).exists():
            extractor = CharLevelEntityExtractor(model_path)
            entities = extractor.extract_entities(text)
            
            print("\n抽取的实体:")
            for entity in entities:
                print(f"  - {entity['name']} [{entity['type']}]")
        else:
            print(f"\n模型文件不存在: {model_path}")
            print("请先训练字符级别NER模型再测试")
    except Exception as e:
        print(f"\n测试失败: {e}")


if __name__ == "__main__":
    test_char_level_ner()