#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体抽取模型训练程序
使用深度学习方法进行装备制造故障相关实体的识别
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import jieba
import jieba.posseg as pseg

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class EquipmentEntityDataset(Dataset):
    """装备制造故障实体数据集"""
    
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 处理标签
        label_ids = label[:self.max_length]
        if len(label_ids) < self.max_length:
            label_ids += [0] * (self.max_length - len(label_ids))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class BERTEntityExtractor(nn.Module):
    """基于BERT的实体抽取模型"""
    
    def __init__(self, bert_model_name: str, num_labels: int, dropout: float = 0.1):
        super(BERTEntityExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
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
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss, logits

class EntityExtractionTrainer:
    """实体抽取模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
        self.model = None
        self.label2id = config['label2id']
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def prepare_training_data(self, data_file: str) -> Tuple[List[str], List[List[int]]]:
        """准备训练数据"""
        logger.info("开始准备训练数据")
        
        # 读取数据
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            text = item['text']
            entities = item['entities']
            
            # 创建标签序列
            label_seq = [0] * len(text)  # 0表示非实体
            
            for entity in entities:
                start = entity['start']
                end = entity['end']
                entity_type = entity['type']
                
                if entity_type in self.label2id:
                    label_id = self.label2id[entity_type]
                    for i in range(start, end):
                        if i < len(label_seq):
                            label_seq[i] = label_id
            
            texts.append(text)
            labels.append(label_seq)
        
        logger.info(f"准备完成，共 {len(texts)} 个样本")
        return texts, labels
    
    def create_data_loaders(self, texts: List[str], labels: List[List[int]], 
                           batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 创建数据集
        train_dataset = EquipmentEntityDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = EquipmentEntityDataset(val_texts, val_labels, self.tokenizer)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练模型"""
        logger.info("开始训练模型")
        
        # 初始化模型
        self.model = BERTEntityExtractor(
            self.config['bert_model_name'], 
            len(self.label2id)
        ).to(self.device)
        
        # 优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            logger.info(f"开始第 {epoch + 1} 轮训练")
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    loss, logits = self.model(input_ids, attention_mask, labels)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
            
            # 计算指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch + 1}:")
            logger.info(f"  训练损失: {avg_train_loss:.4f}")
            logger.info(f"  验证损失: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(f"../models/best_entity_model.pt")
                logger.info("保存最佳模型")
        
        # 保存最终模型
        self.save_model(f"../models/final_entity_model.pt")
        logger.info("训练完成")
    
    def save_model(self, model_path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label2id': self.label2id,
            'id2label': self.id2label
        }, model_path)
        
        # 保存tokenizer
        tokenizer_path = model_path.replace('.pt', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.label2id = checkpoint['label2id']
        self.id2label = checkpoint['id2label']
        
        self.model = BERTEntityExtractor(
            self.config['bert_model_name'], 
            len(self.label2id)
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """预测文本中的实体"""
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model方法")
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        entities = self.decode_entities(text, predictions[0].cpu().numpy())
        return entities
    
    def decode_entities(self, text: str, predictions: np.ndarray) -> List[Dict[str, Any]]:
        """解码实体预测结果"""
        entities = []
        current_entity = None
        
        for i, pred in enumerate(predictions):
            if pred == 0:  # 非实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                entity_type = self.id2label[pred]
                if current_entity is None or current_entity['type'] != entity_type:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'start': i,
                        'end': i + 1,
                        'text': text[i] if i < len(text) else ''
                    }
                else:
                    current_entity['end'] = i + 1
                    if i < len(text):
                        current_entity['text'] += text[i]
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

def create_sample_data():
    """创建示例训练数据"""
    sample_data = [
        {
            "text": "数控机床主轴出现异常振动，检查发现主轴轴承磨损严重",
            "entities": [
                {"type": "equipment", "start": 0, "end": 4, "text": "数控机床"},
                {"type": "component", "start": 4, "end": 6, "text": "主轴"},
                {"type": "fault", "start": 8, "end": 12, "text": "异常振动"},
                {"type": "component", "start": 18, "end": 22, "text": "主轴轴承"},
                {"type": "fault", "start": 22, "end": 26, "text": "磨损严重"}
            ]
        },
        {
            "text": "FANUC系统报警ALM401，伺服电机过载保护",
            "entities": [
                {"type": "brand", "start": 0, "end": 5, "text": "FANUC"},
                {"type": "system", "start": 6, "end": 8, "text": "系统"},
                {"type": "error_code", "start": 9, "end": 15, "text": "ALM401"},
                {"type": "component", "start": 16, "end": 20, "text": "伺服电机"},
                {"type": "fault", "start": 20, "end": 24, "text": "过载保护"}
            ]
        }
    ]
    
    # 保存示例数据
    os.makedirs("../data", exist_ok=True)
    with open("../data/sample_entity_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    return "../data/sample_entity_data.json"

def main():
    """主函数"""
    # 创建示例数据
    data_file = create_sample_data()
    
    # 配置参数
    config = {
        'bert_model_name': 'bert-base-chinese',
        'learning_rate': 2e-5,
        'epochs': 5,
        'batch_size': 16,
        'max_length': 512,
        'label2id': {
            'equipment': 1,
            'component': 2,
            'fault': 3,
            'brand': 4,
            'system': 5,
            'error_code': 6,
            'solution': 7
        }
    }
    
    # 创建训练器
    trainer = EntityExtractionTrainer(config)
    
    # 准备数据
    texts, labels = trainer.prepare_training_data(data_file)
    
    # 创建数据加载器
    train_loader, val_loader = trainer.create_data_loaders(texts, labels, config['batch_size'])
    
    # 训练模型
    trainer.train_model(train_loader, val_loader)
    
    # 测试模型
    test_text = "西门子数控系统出现E1234报警，检查发现X轴伺服驱动器故障"
    entities = trainer.predict_entities(test_text)
    
    print(f"测试文本: {test_text}")
    print("识别到的实体:")
    for entity in entities:
        print(f"  {entity['type']}: {entity['text']}")

if __name__ == "__main__":
    main()