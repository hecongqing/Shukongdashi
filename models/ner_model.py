"""
命名实体识别(NER)模型
基于BERT-BiLSTM-CRF架构实现中文命名实体识别
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer, 
    BertModel, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from loguru import logger
import pickle

from config.settings import CONFIG


class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], 
                 tokenizer: BertTokenizer, label2id: Dict[str, int], 
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 对齐标签
        tokens = self.tokenizer.tokenize(text)
        aligned_labels = self._align_labels(tokens, labels)
        
        # 转换标签为ID
        label_ids = [self.label2id.get(label, self.label2id['O']) for label in aligned_labels]
        
        # 填充标签
        if len(label_ids) < self.max_length:
            label_ids.extend([self.label2id['O']] * (self.max_length - len(label_ids)))
        else:
            label_ids = label_ids[:self.max_length]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def _align_labels(self, tokens: List[str], labels: List[str]) -> List[str]:
        """对齐BERT token和标签"""
        aligned_labels = []
        
        # 简化的对齐策略
        if len(tokens) == len(labels):
            aligned_labels = labels
        elif len(tokens) > len(labels):
            # 如果token数量多于标签，用'O'填充
            aligned_labels = labels + ['O'] * (len(tokens) - len(labels))
        else:
            # 如果token数量少于标签，截断标签
            aligned_labels = labels[:len(tokens)]
        
        # 添加[CLS]和[SEP]标签
        aligned_labels = ['O'] + aligned_labels + ['O']
        
        return aligned_labels


class BertBiLSTMCRF(nn.Module):
    """BERT-BiLSTM-CRF模型"""
    
    def __init__(self, bert_model_name: str, num_labels: int, 
                 hidden_dim: int = 128, dropout: float = 0.1):
        super(BertBiLSTMCRF, self).__init__()
        
        self.num_labels = num_labels
        
        # BERT层
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            self.bert_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if hidden_dim > 1 else 0
        )
        
        # 分类层
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF层
        self.crf = CRF(num_labels, batch_first=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM编码
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        # 分类
        logits = self.classifier(lstm_output)
        
        outputs = (logits,)
        
        if labels is not None:
            # 计算CRF损失
            log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
            loss = -log_likelihood
            outputs = (loss,) + outputs
        
        return outputs
    
    def predict(self, input_ids, attention_mask):
        """预测序列标签"""
        with torch.no_grad():
            # BERT编码
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            sequence_output = bert_output.last_hidden_state
            
            # BiLSTM编码
            lstm_output, _ = self.lstm(sequence_output)
            
            # 分类
            logits = self.classifier(lstm_output)
            
            # CRF预测
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            
        return predictions


class NERTrainer:
    """NER模型训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["model"]["ner"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
        
        # 标签映射
        self.labels = self.config["labels"]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        # 模型保存路径
        self.model_path = Path(self.config["model_path"])
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[List[str]]]:
        """加载训练数据"""
        texts, labels = [], []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                texts.append(item['text'])
                labels.append(item['labels'])
                
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
            # 假设CSV格式：text列包含文本，labels列包含标签（用空格分隔）
            for _, row in df.iterrows():
                texts.append(row['text'])
                labels.append(row['labels'].split())
        
        else:
            # 假设是CoNLL格式
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_text, current_labels = [], []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_text:
                        texts.append(''.join(current_text))
                        labels.append(current_labels)
                        current_text, current_labels = [], []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_text.append(parts[0])
                        current_labels.append(parts[1])
            
            # 添加最后一个句子
            if current_text:
                texts.append(''.join(current_text))
                labels.append(current_labels)
        
        logger.info(f"Loaded {len(texts)} training examples")
        return texts, labels
    
    def create_dataloader(self, texts: List[str], labels: List[List[str]], 
                         batch_size: int, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = NERDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.config["max_length"]
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """数据批处理函数"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def train(self, train_data_path: str, val_data_path: str = None):
        """训练模型"""
        logger.info("Starting NER model training...")
        
        # 加载数据
        train_texts, train_labels = self.load_data(train_data_path)
        
        if val_data_path:
            val_texts, val_labels = self.load_data(val_data_path)
        else:
            # 从训练数据中分割验证集
            split_idx = int(len(train_texts) * 0.8)
            val_texts = train_texts[split_idx:]
            val_labels = train_labels[split_idx:]
            train_texts = train_texts[:split_idx]
            train_labels = train_labels[:split_idx]
        
        # 创建数据加载器
        train_dataloader = self.create_dataloader(
            train_texts, train_labels, self.config["batch_size"], shuffle=True
        )
        val_dataloader = self.create_dataloader(
            val_texts, val_labels, self.config["batch_size"], shuffle=False
        )
        
        # 初始化模型
        model = BertBiLSTMCRF(
            bert_model_name=self.config["model_name"],
            num_labels=len(self.labels)
        )
        model.to(self.device)
        
        # 优化器和调度器
        optimizer = AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            eps=1e-8
        )
        
        total_steps = len(train_dataloader) * self.config["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_val_f1 = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 训练
            model.train()
            total_train_loss = 0
            
            for batch in train_dataloader:
                # 移到GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs[0]
                total_train_loss += loss.item()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            
            # 验证
            val_loss, val_f1 = self.evaluate(model, val_dataloader)
            val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model(model)
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        
        # 绘制训练曲线
        self.plot_training_curve(train_losses, val_losses)
        
        logger.info("Training completed!")
        return model
    
    def evaluate(self, model, dataloader) -> Tuple[float, float]:
        """评估模型"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 计算损失
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0]
                total_loss += loss.item()
                
                # 预测
                predictions = model.predict(input_ids, attention_mask)
                
                # 收集预测和真实标签
                for i, pred in enumerate(predictions):
                    mask = attention_mask[i].cpu().numpy()
                    true_labels = labels[i].cpu().numpy()
                    
                    # 只考虑有效位置（mask=1）
                    valid_length = sum(mask)
                    pred = pred[:valid_length]
                    true_labels = true_labels[:valid_length]
                    
                    all_predictions.extend(pred)
                    all_labels.extend(true_labels)
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算F1分数
        f1_score = self.calculate_f1(all_predictions, all_labels)
        
        return avg_loss, f1_score
    
    def calculate_f1(self, predictions: List[int], labels: List[int]) -> float:
        """计算F1分数"""
        from sklearn.metrics import f1_score
        
        # 转换ID为标签
        pred_labels = [self.id2label[pred] for pred in predictions]
        true_labels = [self.id2label[label] for label in labels]
        
        return f1_score(true_labels, pred_labels, average='weighted')
    
    def save_model(self, model):
        """保存模型"""
        # 保存模型权重
        torch.save(model.state_dict(), self.model_path / "model.pth")
        
        # 保存配置
        config_data = {
            "model_config": self.config,
            "label2id": self.label2id,
            "id2label": self.id2label
        }
        
        with open(self.model_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(self.model_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, model_path: str = None):
        """加载模型"""
        if model_path is None:
            model_path = self.model_path
        else:
            model_path = Path(model_path)
        
        # 加载配置
        with open(model_path / "config.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.label2id = config_data["label2id"]
        self.id2label = config_data["id2label"]
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(str(model_path))
        
        # 加载模型
        model = BertBiLSTMCRF(
            bert_model_name=self.config["model_name"],
            num_labels=len(self.label2id)
        )
        
        model.load_state_dict(torch.load(model_path / "model.pth", map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def predict(self, text: str, model=None) -> List[Tuple[str, str]]:
        """预测单个文本的实体"""
        if model is None:
            model = self.load_model()
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config["max_length"],
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        predictions = model.predict(input_ids, attention_mask)
        
        # 解码结果
        tokens = self.tokenizer.tokenize(text)
        pred_labels = [self.id2label[pred] for pred in predictions[0]]
        
        # 提取实体
        entities = self.extract_entities(tokens, pred_labels[:len(tokens)])
        
        return entities
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> List[Tuple[str, str]]:
        """从BIO标签提取实体"""
        entities = []
        current_entity = ""
        current_label = ""
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token.replace('##', '')
                current_label = label[2:]
            elif label.startswith('I-') and current_label == label[2:]:
                # 继续当前实体
                current_entity += token.replace('##', '')
            else:
                # 结束当前实体
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = ""
                current_label = ""
        
        # 添加最后一个实体
        if current_entity:
            entities.append((current_entity, current_label))
        
        return entities
    
    def plot_training_curve(self, train_losses: List[float], val_losses: List[float]):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('NER Model Training Curve')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.model_path / "training_curve.png")
        plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NER Model Training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--predict", help="Text to predict")
    parser.add_argument("--model-path", help="Model path for loading")
    
    args = parser.parse_args()
    
    trainer = NERTrainer()
    
    if args.train:
        trainer.train(args.train_data, args.val_data)
    
    if args.predict:
        model = trainer.load_model(args.model_path)
        entities = trainer.predict(args.predict, model)
        
        print("Detected entities:")
        for entity, label in entities:
            print(f"  {entity}: {label}")


if __name__ == "__main__":
    main()