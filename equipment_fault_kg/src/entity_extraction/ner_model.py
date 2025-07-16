"""
基于深度学习的命名实体识别模型

使用BERT等预训练模型进行装备制造领域的实体识别
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


class EquipmentNERDataset(Dataset):
    """装备制造领域NER数据集"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 实体标签映射 - 统一使用4类标签
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
        label = self.labels[idx]
        
        # 分词和标签对齐
        tokens = self.tokenizer.tokenize(text)
        labels = self.align_labels(tokens, label)
        
        # 截断到最大长度
        if len(tokens) > self.max_length - 2:  # 保留[CLS]和[SEP]
            tokens = tokens[:self.max_length - 2]
            labels = labels[:self.max_length - 2]
        
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = ['O'] + labels + ['O']
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self.label2id.get(label, 0) for label in labels]
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def align_labels(self, tokens: List[str], original_labels: List[str]) -> List[str]:
        """将原始标签与分词后的token对齐"""
        aligned_labels = []
        label_idx = 0
        
        for token in tokens:
            if label_idx < len(original_labels):
                aligned_labels.append(original_labels[label_idx])
                label_idx += 1
            else:
                aligned_labels.append('O')
        
        return aligned_labels


class EquipmentNERModel(nn.Module):
    """装备制造领域NER模型"""
    
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
            loss_fct = nn.CrossEntropyLoss()
            # 只计算非padding位置的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        return loss, logits


class NERModel:
    """NER模型管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model']['name']
        self.max_length = config['model']['max_length']
        self.batch_size = config['model']['batch_size']
        self.learning_rate = config['model']['learning_rate']
        self.epochs = config['model']['epochs']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = None
        
        # 实体类型
        self.entity_types = config['entity_types']
        
    def prepare_data(self, texts: List[str], labels: List[List[str]] = None) -> Tuple[DataLoader, DataLoader]:
        """准备训练和验证数据"""
        if labels is None:
            # 如果没有标签，创建伪标签用于演示
            labels = [['O'] * len(text) for text in texts]
        
        # 分割数据
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 创建数据集
        train_dataset = EquipmentNERDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = EquipmentNERDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练模型"""
        num_labels = len(train_loader.dataset.label2id)
        self.model = EquipmentNERModel(self.model_name, num_labels).to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        best_f1 = 0
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证阶段
            val_f1 = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model('models/best_ner_model.pth')
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask)
                
                # 获取预测结果
                preds = torch.argmax(logits, dim=-1)
                
                # 只保留非padding位置
                active_preds = preds[attention_mask == 1]
                active_labels = labels[attention_mask == 1]
                
                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """预测文本中的实体"""
        if self.model is None:
            self.load_model('models/best_ner_model.pth')
        
        self.model.eval()
        
        # 分词
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # 添加特殊标记
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        pred_labels = [self.model.id2label[pred.item()] for pred in preds[0][1:-1]]  # 去掉[CLS]和[SEP]
        
        # 提取实体
        entities = self.extract_entities(tokens[1:-1], pred_labels)
        
        return entities
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """从标签序列中提取实体"""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # 去掉'B-'前缀
                current_entity = {
                    'type': entity_type,
                    'text': token,
                    'start': i,
                    'end': i
                }
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                entity_type = label[2:]  # 去掉'I-'前缀
                if entity_type == current_entity['type']:
                    current_entity['text'] += token
                    current_entity['end'] = i
            else:
                # 'O'标签，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        # 后处理：过滤明显错误的实体
        filtered_entities = []
        for entity in entities:
            # 过滤标点符号实体
            if entity['text'] in ['。', '，', '；', '：', '！', '？', '、', '（', '）', '【', '】']:
                continue
            
            # 过滤单字符动词
            if len(entity['text']) == 1 and entity['text'] in ['使', '用', '检', '测', '修', '维']:
                continue
            
            # 过滤包含标点符号的实体（保留实体部分）
            if any(punct in entity['text'] for punct in ['。', '，', '；', '：', '！', '？', '、']):
                # 移除末尾的标点符号
                clean_text = entity['text'].rstrip('。，；：！？、')
                if clean_text and clean_text != entity['text']:
                    entity['text'] = clean_text
            
            # 过滤空实体
            if entity['text'].strip():
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def save_model(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        if not Path(path).exists():
            logger.warning(f"模型文件不存在: {path}")
            return
        
        num_labels = 9  # 根据标签数量设置 (4类实体 * 2 + 1个O标签)
        self.model = EquipmentNERModel(self.model_name, num_labels).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"模型已从 {path} 加载")