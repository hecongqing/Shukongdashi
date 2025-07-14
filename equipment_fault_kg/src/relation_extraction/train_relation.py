import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import pickle

from .data_processor import RelationDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationDataset(Dataset):
    """关系抽取数据集类"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 关系类型映射
        self.relation2id = {
            "部件故障": 0,
            "性能故障": 1,
            "检测工具": 2,
            "组成": 3
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        head_entity = sample['head_entity']
        tail_entity = sample['tail_entity']
        relation_type = sample['relation_type']
        label = sample.get('label', 1)  # 默认为正样本
        
        # 构建输入文本
        input_text = f"{head_entity} [SEP] {tail_entity} [SEP] {text}"
        
        # 编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 获取关系ID
        relation_id = self.relation2id.get(relation_type, 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'relation_ids': torch.tensor(relation_id, dtype=torch.long)
        }

class RelationModel(nn.Module):
    """关系抽取模型类"""
    
    def __init__(self, bert_model_name: str, num_relations: int, dropout: float = 0.1):
        super(RelationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 关系分类器
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
        
        # 二分类器（正负样本）
        self.binary_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, labels=None, relation_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        
        # 关系分类logits
        relation_logits = self.relation_classifier(pooled_output)
        
        # 二分类logits
        binary_logits = self.binary_classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # 二分类损失
            binary_loss_fct = nn.CrossEntropyLoss()
            binary_loss = binary_loss_fct(binary_logits, labels)
            
            # 关系分类损失（仅对正样本）
            relation_loss = 0
            if relation_ids is not None:
                relation_loss_fct = nn.CrossEntropyLoss()
                relation_loss = relation_loss_fct(relation_logits, relation_ids)
            
            # 总损失
            loss = binary_loss + relation_loss
        
        return loss, binary_logits, relation_logits

class RelationTrainer:
    """关系抽取训练器类"""
    
    def __init__(self, model_name: str = 'bert-base-chinese', device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        
    def prepare_data(self, data: List[Dict], test_size: float = 0.2):
        """准备训练数据"""
        # 分割训练集和验证集
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
        
        # 创建数据集
        train_dataset = RelationDataset(train_data, self.tokenizer)
        val_dataset = RelationDataset(val_data, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, 
              batch_size: int = 16, epochs: int = 10, 
              learning_rate: float = 2e-5, warmup_steps: int = 500):
        """训练模型"""
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        num_relations = len(train_dataset.relation2id)
        self.model = RelationModel(self.model_name, num_relations)
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
                relation_ids = batch['relation_ids'].to(self.device)
                
                optimizer.zero_grad()
                loss, _, _ = self.model(input_ids, attention_mask, labels, relation_ids)
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
                self.save_model('best_relation_model.pth')
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
    
    def evaluate(self, data_loader) -> float:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_relation_preds = []
        all_relation_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                relation_ids = batch['relation_ids'].to(self.device)
                
                _, binary_logits, relation_logits = self.model(input_ids, attention_mask)
                
                # 二分类预测
                binary_preds = torch.argmax(binary_logits, dim=-1)
                all_preds.extend(binary_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 关系分类预测（仅对正样本）
                relation_preds = torch.argmax(relation_logits, dim=-1)
                for i, label in enumerate(labels):
                    if label == 1:  # 正样本
                        all_relation_preds.append(relation_preds[i].cpu().numpy())
                        all_relation_labels.append(relation_ids[i].cpu().numpy())
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 打印详细评估报告
        logger.info("Binary Classification Report:")
        logger.info(classification_report(all_labels, all_preds))
        
        if all_relation_preds:
            relation_f1 = f1_score(all_relation_labels, all_relation_preds, average='weighted')
            logger.info(f"Relation Classification F1: {relation_f1:.4f}")
        
        return f1
    
    def save_model(self, model_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'relation2id': self.model.relation_classifier.out_features
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_relations = checkpoint['relation2id']
        self.model = RelationModel(self.model_name, num_relations)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_path}")

def train_relation_model(data_path: str, output_dir: str = './models'):
    """训练关系抽取模型的主函数"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据预处理
    processor = RelationDataProcessor()
    raw_data = processor.load_data(data_path)
    
    # 提取关系样本
    relation_samples = processor.extract_relation_samples(raw_data)
    
    # 转换为分类格式
    classification_data = processor.convert_to_classification_format(relation_samples)
    
    # 创建负样本
    negative_samples = processor.create_negative_samples(relation_samples, negative_ratio=1.0)
    
    # 合并正负样本
    all_data = classification_data + negative_samples
    
    # 保存处理后的数据
    processor.save_data(all_data, os.path.join(output_dir, 'relation_data.json'))
    
    # 统计信息
    stats = processor.get_relation_statistics(relation_samples)
    with open(os.path.join(output_dir, 'relation_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 训练模型
    trainer = RelationTrainer()
    train_dataset, val_dataset = trainer.prepare_data(all_data)
    
    # 开始训练
    trainer.train(train_dataset, val_dataset)
    
    # 保存最终模型
    trainer.save_model(os.path.join(output_dir, 'final_relation_model.pth'))
    
    logger.info("Relation extraction model training completed!")

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
    with open('sample_relation_data.json', 'w', encoding='utf-8') as f:
        for sample in sample_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 训练模型
    train_relation_model('sample_relation_data.json', './relation_models')