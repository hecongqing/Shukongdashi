import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import numpy as np
import json
import logging
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

from .data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERDataset(Dataset):
    """NER数据集类"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
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
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # 分词
        tokens = self.tokenizer.tokenize(text)
        token_labels = []
        
        # 将字符级标签转换为token级标签
        char_to_token = {}
        token_idx = 0
        for i, char in enumerate(text):
            if token_idx < len(tokens):
                if self.tokenizer.convert_tokens_to_string([tokens[token_idx]]).strip() == char:
                    char_to_token[i] = token_idx
                    token_labels.append(labels[i] if i < len(labels) else 'O')
                    token_idx += 1
                elif char == ' ':
                    continue
                else:
                    # 处理中文分词
                    if token_idx < len(tokens):
                        char_to_token[i] = token_idx
                        token_labels.append(labels[i] if i < len(labels) else 'O')
                        token_idx += 1
        
        # 截断
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
            token_labels = token_labels[:self.max_length - 2]
        
        # 添加特殊token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        token_labels = ['O'] + token_labels + ['O']
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self.label2id.get(label, 0) for label in token_labels]
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length
        label_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
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

class NERTrainer:
    """NER训练器类"""
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              output_dir: str, batch_size: int = 16, epochs: int = 10,
              learning_rate: float = 2e-5, warmup_steps: int = 500):
        """训练NER模型"""
        
        # 创建数据集
        train_dataset = NERDataset(train_data, self.tokenizer)
        val_dataset = NERDataset(val_data, self.tokenizer)
        
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        # 训练循环
        best_f1 = 0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练
            self.model.train()
            total_loss = 0
            train_progress = tqdm(train_loader, desc="Training")
            
            for batch in train_progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                train_progress.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # 验证
            val_f1 = self.evaluate(val_loader)
            logger.info(f"Validation F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(output_dir)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
        
        return best_f1
    
    def evaluate(self, data_loader) -> float:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(input_ids, attention_mask)
                
                # 获取预测结果
                preds = torch.argmax(logits, dim=2)
                
                # 只保留非padding位置
                active_preds = preds[attention_mask == 1]
                active_labels = labels[attention_mask == 1]
                
                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
    
    def save_model(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'ner_model.pth'))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存标签映射
        label_mapping = {
            'label2id': {k: v for k, v in self.model.classifier.state_dict().items()},
            'id2label': {v: k for k, v in self.model.classifier.state_dict().items()}
        }
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """加载模型"""
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # 加载标签映射
        with open(os.path.join(model_dir, 'label_mapping.json'), 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # 初始化模型
        num_labels = len(label_mapping['label2id'])
        self.model = NERModel(self.model_name, num_labels)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'ner_model.pth')))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_dir}")
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # 分词
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:  # 保留[CLS]和[SEP]的位置
            tokens = tokens[:510]
        
        # 添加特殊token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=2)
        
        # 解码预测结果
        pred_labels = preds[0].cpu().numpy()
        entities = self._decode_entities(text, tokens, pred_labels)
        
        return entities
    
    def _decode_entities(self, text: str, tokens: List[str], labels: List[int]) -> List[Dict]:
        """解码实体"""
        entities = []
        current_entity = None
        
        for i, (token, label_id) in enumerate(zip(tokens, labels)):
            if i == 0 or i == len(tokens) - 1:  # 跳过[CLS]和[SEP]
                continue
            
            label = self.model.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'start': i - 1,  # 减去[CLS]
                    'end': i,
                    'text': token
                }
            
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                # 继续当前实体
                current_entity['end'] = i
                current_entity['text'] += token.replace('##', '')
            
            else:
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities

def main():
    """主函数，用于训练NER模型"""
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据（假设数据文件路径）
    train_file = "data/train.json"
    if os.path.exists(train_file):
        data = processor.load_data(train_file)
        
        # 转换为NER格式
        ner_data = processor.convert_to_ner_format(data)
        
        # 划分数据集
        train_data, val_data, test_data = processor.split_data(ner_data)
        
        # 初始化训练器
        trainer = NERTrainer()
        
        # 训练模型
        output_dir = "models/ner_model"
        best_f1 = trainer.train(train_data, val_data, output_dir)
        
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
    else:
        logger.warning(f"Training data file {train_file} not found. Please provide the data file.")

if __name__ == "__main__":
    main()