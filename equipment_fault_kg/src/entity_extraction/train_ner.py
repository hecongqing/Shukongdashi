import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import pickle

from .data_processor import EntityDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 标签映射
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
        
        # 分词
        tokens = []
        label_ids = []
        
        # 添加[CLS]标记
        tokens.append('[CLS]')
        label_ids.append(self.label2id['O'])
        
        # 处理文本和标签
        for i, char in enumerate(text):
            if i < len(label_seq):
                label = label_seq[i]
            else:
                label = 'O'
            
            # 分词
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            
            tokens.extend(sub_tokens)
            
            # 标签处理：第一个子词使用原标签，后续子词使用I-标签
            for j, sub_token in enumerate(sub_tokens):
                if j == 0:
                    label_ids.append(self.label2id[label])
                else:
                    # 将B-标签转换为I-标签
                    if label.startswith('B-'):
                        label_ids.append(self.label2id[label.replace('B-', 'I-')])
                    else:
                        label_ids.append(self.label2id[label])
        
        # 添加[SEP]标记
        tokens.append('[SEP]')
        label_ids.append(self.label2id['O'])
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            label_ids = label_ids[:self.max_length]
        else:
            # 填充
            padding_length = self.max_length - len(tokens)
            tokens.extend(['[PAD]'] * padding_length)
            # 对于填充位置使用特殊忽略标签 -100，避免在损失计算中影响 "O" 标签
            label_ids.extend([-100] * padding_length)
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            # 使用长整型保持与 CrossEntropyLoss 兼容，同时保留 -100 作为 ignore_index
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

class NERModel(nn.Module):
    """NER模型类"""
    
    def __init__(self, bert_model_name: str, num_labels: int, dropout: float = 0.1):
        super(NERModel, self).__init__()
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
            # 使用 -100 作为忽略索引（与上方填充标签一致），避免将 'O' 标签置为忽略
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return loss, logits

class NERTrainer:
    """NER训练器类"""
    
    def __init__(self, model_name: str = 'bert-base-chinese', device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(self, data: List[Dict], test_size: float = 0.2):
        """准备训练数据"""
        texts = [sample['text'] for sample in data]
        labels = [sample['labels'] for sample in data]
        
        # 分割训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # 创建数据集
        train_dataset = NERDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = NERDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, 
              batch_size: int = 16, epochs: int = 10, 
              learning_rate: float = 2e-5, warmup_steps: int = 500):
        """训练模型"""
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        num_labels = len(train_dataset.label2id)
        self.model = NERModel(self.model_name, num_labels)
        self.model.to(self.device)
        
        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        # 训练循环
        best_f1 = 0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # 验证阶段
            val_f1 = self.evaluate(val_loader)
            logger.info(f"Validation F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model('best_ner_model.pth')
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
    
    def evaluate(self, data_loader) -> float:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                
                # 收集非PAD位置的预测和标签
                for i in range(input_ids.size(0)):
                    for j in range(input_ids.size(1)):
                        if attention_mask[i][j] == 1 and labels[i][j] != 0:  # 非PAD且非O标签
                            all_preds.append(preds[i][j].item())
                            all_labels.append(labels[i][j].item())
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
    
    def save_model(self, model_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'label2id': self.model.classifier.out_features
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_labels = checkpoint['label2id']
        self.model = NERModel(self.model_name, num_labels)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_path}")

def train_ner_model(data_path: str, output_dir: str = './models'):
    """训练NER模型的主函数"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    processor = EntityDataProcessor()
    raw_data = processor.load_data(data_path)
    ner_data = processor.convert_to_ner_format(raw_data)
    
    # 保存处理后的数据
    processor.save_ner_data(ner_data, os.path.join(output_dir, 'ner_data.json'))
    
    # 统计信息
    stats = processor.get_entity_statistics(raw_data)
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 训练模型
    trainer = NERTrainer()
    train_dataset, val_dataset = trainer.prepare_data(ner_data)
    
    # 开始训练
    trainer.train(train_dataset, val_dataset)
    
    # 保存最终模型
    trainer.save_model(os.path.join(output_dir, 'final_ner_model.pth'))
    
    logger.info("NER model training completed!")

if __name__ == "__main__":
    # 测试训练流程
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
    
    # 保存示例数据
    with open('sample_data.json', 'w', encoding='utf-8') as f:
        for sample in sample_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 训练模型
    train_ner_model('sample_data.json', './ner_models')