import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import json
import logging
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
import random

from ..entity_extraction.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class REDataset(Dataset):
    """关系抽取数据集类"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, 
                 entity_types: List[str] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 关系类型映射
        self.relation2id = {
            '部件故障': 0,
            '性能故障': 1,
            '检测工具': 2,
            '组成': 3,
            'no_relation': 4  # 无关系
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        # 实体类型
        self.entity_types = entity_types or ['COMPONENT', 'PERFORMANCE', 'FAULT_STATE', 'DETECTION_TOOL']
        
        # 生成训练样本
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict]:
        """生成训练样本"""
        samples = []
        
        for item in self.data:
            text = item['text']
            entities = item.get('entities', [])
            relations = item.get('relations', [])
            
            # 创建实体对
            entity_pairs = []
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities):
                    if i != j:  # 排除自己
                        entity_pairs.append((entity1, entity2))
            
            # 为每个实体对创建样本
            for head, tail in entity_pairs:
                # 检查是否存在关系
                relation_type = 'no_relation'
                for rel in relations:
                    if (rel['head'].name == head.name and rel['tail'].name == tail.name and
                        rel['head'].start == head.start and rel['tail'].start == tail.start):
                        relation_type = rel['relation_type']
                        break
                
                samples.append({
                    'text': text,
                    'head': head,
                    'tail': tail,
                    'relation': relation_type
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        head = sample['head']
        tail = sample['tail']
        relation = sample['relation']
        
        # 标记实体位置
        marked_text = self._mark_entities(text, head, tail)
        
        # 分词
        tokens = self.tokenizer.tokenize(marked_text)
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # 添加特殊token
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        
        # 标签
        label = self.relation2id.get(relation, 4)  # 默认为no_relation
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'head_type': head.type,
            'tail_type': tail.type
        }
    
    def _mark_entities(self, text: str, head: object, tail: object) -> str:
        """在文本中标记实体位置"""
        # 使用特殊标记包围实体
        marked_text = text[:head.start] + f"[HEAD_{head.type}]" + text[head.start:head.end] + f"[/HEAD_{head.type}]" + text[head.end:]
        
        # 调整tail的位置
        if head.end <= tail.start:
            tail_start = tail.start + len(f"[HEAD_{head.type}]") + len(f"[/HEAD_{head.type}]")
            tail_end = tail.end + len(f"[HEAD_{head.type}]") + len(f"[/HEAD_{head.type}]")
        else:
            tail_start = tail.start
            tail_end = tail.end
        
        marked_text = marked_text[:tail_start] + f"[TAIL_{tail.type}]" + marked_text[tail_start:tail_end] + f"[/TAIL_{tail.type}]" + marked_text[tail_end:]
        
        return marked_text

class REModel(nn.Module):
    """关系抽取模型类"""
    
    def __init__(self, bert_model_name: str, num_relations: int, dropout: float = 0.1):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return loss, logits

class RETrainer:
    """关系抽取训练器类"""
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              output_dir: str, batch_size: int = 16, epochs: int = 10,
              learning_rate: float = 2e-5, warmup_steps: int = 500):
        """训练关系抽取模型"""
        
        # 创建数据集
        train_dataset = REDataset(train_data, self.tokenizer)
        val_dataset = REDataset(val_data, self.tokenizer)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        num_relations = len(train_dataset.relation2id)
        self.model = REModel(self.model_name, num_relations)
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
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算F1分数（排除no_relation）
        filtered_preds = []
        filtered_labels = []
        for pred, label in zip(all_preds, all_labels):
            if label != 4:  # 不是no_relation
                filtered_preds.append(pred)
                filtered_labels.append(label)
        
        if filtered_preds:
            f1 = f1_score(filtered_labels, filtered_preds, average='weighted')
        else:
            f1 = 0.0
        
        return f1
    
    def save_model(self, output_dir: str):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(output_dir, 're_model.pth'))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存关系映射
        relation_mapping = {
            'relation2id': self.model.relation2id if hasattr(self.model, 'relation2id') else {},
            'id2relation': self.model.id2relation if hasattr(self.model, 'id2relation') else {}
        }
        with open(os.path.join(output_dir, 'relation_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump(relation_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """加载模型"""
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # 加载关系映射
        with open(os.path.join(model_dir, 'relation_mapping.json'), 'r', encoding='utf-8') as f:
            relation_mapping = json.load(f)
        
        # 初始化模型
        num_relations = len(relation_mapping['relation2id'])
        self.model = REModel(self.model_name, num_relations)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 're_model.pth')))
        self.model.to(self.device)
        self.model.eval()
        
        # 设置关系映射
        self.model.relation2id = relation_mapping['relation2id']
        self.model.id2relation = relation_mapping['id2relation']
        
        logger.info(f"Model loaded from {model_dir}")
    
    def predict(self, text: str, entities: List[Dict]) -> List[Dict]:
        """预测文本中实体间的关系"""
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        relations = []
        
        # 为每对实体预测关系
        for i, head in enumerate(entities):
            for j, tail in enumerate(entities):
                if i != j:
                    # 标记实体
                    marked_text = self._mark_entities(text, head, tail)
                    
                    # 分词
                    tokens = self.tokenizer.tokenize(marked_text)
                    if len(tokens) > 510:
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
                        pred = torch.argmax(logits, dim=1)
                        relation_id = pred.item()
                    
                    # 获取关系类型
                    relation_type = self.model.id2relation.get(relation_id, 'no_relation')
                    
                    # 只保留有效关系
                    if relation_type != 'no_relation':
                        relations.append({
                            'head': head,
                            'tail': tail,
                            'relation': relation_type,
                            'confidence': torch.softmax(logits, dim=1).max().item()
                        })
        
        return relations
    
    def _mark_entities(self, text: str, head: Dict, tail: Dict) -> str:
        """在文本中标记实体位置"""
        # 使用特殊标记包围实体
        marked_text = text[:head['start']] + f"[HEAD_{head['type']}]" + text[head['start']:head['end']] + f"[/HEAD_{head['type']}]" + text[head['end']:]
        
        # 调整tail的位置
        if head['end'] <= tail['start']:
            tail_start = tail['start'] + len(f"[HEAD_{head['type']}]") + len(f"[/HEAD_{head['type']}]")
            tail_end = tail['end'] + len(f"[HEAD_{head['type']}]") + len(f"[/HEAD_{head['type']}]")
        else:
            tail_start = tail['start']
            tail_end = tail['end']
        
        marked_text = marked_text[:tail_start] + f"[TAIL_{tail['type']}]" + marked_text[tail_start:tail_end] + f"[/TAIL_{tail['type']}]" + marked_text[tail_end:]
        
        return marked_text

def main():
    """主函数，用于训练关系抽取模型"""
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据（假设数据文件路径）
    train_file = "data/train.json"
    if os.path.exists(train_file):
        data = processor.load_data(train_file)
        
        # 转换为关系抽取格式
        re_data = processor.convert_to_re_format(data)
        
        # 划分数据集
        train_data, val_data, test_data = processor.split_data(re_data)
        
        # 初始化训练器
        trainer = RETrainer()
        
        # 训练模型
        output_dir = "models/re_model"
        best_f1 = trainer.train(train_data, val_data, output_dir)
        
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
    else:
        logger.warning(f"Training data file {train_file} not found. Please provide the data file.")

if __name__ == "__main__":
    main()