"""
专用于故障数据集的命名实体识别模型

支持4种实体类型：
- 部件单元 (Component Unit)
- 性能表征 (Performance Characteristic)  
- 故障状态 (Fault State)
- 检测工具 (Detection Tool)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from loguru import logger
import pickle
from tqdm import tqdm


class FaultNERDataset(Dataset):
    """故障数据集的NER数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 实体标签映射 - 针对故障数据集的4种实体类型
        self.label2id = {
            'O': 0,
            'B-部件单元': 1, 'I-部件单元': 2,
            'B-性能表征': 3, 'I-性能表征': 4,
            'B-故障状态': 5, 'I-故障状态': 6,
            'B-检测工具': 7, 'I-检测工具': 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
        
        # 预处理数据
        self.processed_data = self._process_data()
    
    def _process_data(self) -> List[Dict]:
        """处理原始数据，将实体标注转换为BIO标签"""
        processed = []
        
        for item in self.data:
            text = item['text']
            spo_list = item.get('spo_list', [])
            
            # 初始化标签序列
            labels = ['O'] * len(text)
            
            # 根据spo_list标注实体
            for spo in spo_list:
                h_entity = spo['h']
                t_entity = spo['t']
                relation = spo['relation']
                
                # 标注头实体
                h_start, h_end = h_entity['pos']
                entity_type = self._get_entity_type_from_relation(h_entity['name'], relation, is_head=True)
                if entity_type:
                    self._mark_entity(labels, h_start, h_end, entity_type)
                
                # 标注尾实体
                t_start, t_end = t_entity['pos']
                entity_type = self._get_entity_type_from_relation(t_entity['name'], relation, is_head=False)
                if entity_type:
                    self._mark_entity(labels, t_start, t_end, entity_type)
            
            processed.append({
                'text': text,
                'labels': labels
            })
        
        return processed
    
    def _get_entity_type_from_relation(self, entity_name: str, relation: str, is_head: bool) -> str:
        """根据关系和实体位置推断实体类型"""
        if relation == "部件故障":
            return "部件单元" if is_head else "故障状态"
        elif relation == "性能故障":
            return "性能表征" if is_head else "故障状态"
        elif relation == "检测工具":
            return "检测工具" if is_head else "性能表征"
        elif relation == "组成":
            return "部件单元"  # 组成关系中头尾都是部件单元
        
        # 根据实体名称进行启发式判断
        return self._heuristic_entity_type(entity_name)
    
    def _heuristic_entity_type(self, entity_name: str) -> str:
        """启发式判断实体类型"""
        # 检测工具关键词
        tool_keywords = ['互感器', '保护器', '测试仪', '检测器', '传感器', '监测仪', '探测器']
        # 性能表征关键词  
        performance_keywords = ['压力', '温度', '转速', '电流', '电压', '流量', '振动', '噪音']
        # 故障状态关键词
        fault_keywords = ['漏油', '断裂', '变形', '卡滞', '损坏', '磨损', '老化', '故障']
        
        if any(keyword in entity_name for keyword in tool_keywords):
            return "检测工具"
        elif any(keyword in entity_name for keyword in performance_keywords):
            return "性能表征"
        elif any(keyword in entity_name for keyword in fault_keywords):
            return "故障状态"
        else:
            return "部件单元"  # 默认为部件单元
    
    def _mark_entity(self, labels: List[str], start: int, end: int, entity_type: str):
        """在标签序列中标记实体"""
        if start < len(labels) and end <= len(labels):
            labels[start] = f'B-{entity_type}'
            for i in range(start + 1, end):
                if i < len(labels):
                    labels[i] = f'I-{entity_type}'
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        text = item['text']
        labels = item['labels']
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # 对齐标签
        aligned_labels = self._align_labels_with_tokens(
            text, labels, encoding['offset_mapping'][0]
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def _align_labels_with_tokens(self, text: str, labels: List[str], offsets) -> List[int]:
        """对齐标签与tokenizer的分词结果"""
        aligned_labels = []
        
        for offset in offsets:
            start, end = offset.tolist()
            if start == 0 and end == 0:  # [CLS] or [SEP]
                aligned_labels.append(self.label2id['O'])
            else:
                # 使用字符级别的标签
                char_label = labels[start] if start < len(labels) else 'O'
                aligned_labels.append(self.label2id.get(char_label, self.label2id['O']))
        
        return aligned_labels


class FaultNERModel(nn.Module):
    """故障NER模型"""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


class FaultNERTrainer:
    """故障NER训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.model = None
        self.dataset = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(self, data_path: str):
        """加载训练数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
        self.dataset = FaultNERDataset(data, self.tokenizer, self.config['max_length'])
        logger.info(f"Loaded {len(data)} samples")
        
        # 初始化模型
        self.model = FaultNERModel(
            self.config['model_name'], 
            self.dataset.num_labels,
            self.config['dropout']
        ).to(self.device)
    
    def train(self, train_data_path: str, val_data_path: str = None):
        """训练模型"""
        # 加载数据
        self.load_data(train_data_path)
        
        # 数据分割
        if val_data_path:
            with open(val_data_path, 'r', encoding='utf-8') as f:
                val_data = [json.loads(line.strip()) for line in f]
            val_dataset = FaultNERDataset(val_data, self.tokenizer, self.config['max_length'])
        else:
            # 从训练集中分割验证集
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # 优化器和调度器
        optimizer = AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_f1 = 0
        for epoch in range(self.config['epochs']):
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # 验证
            val_f1 = self._evaluate(val_loader)
            logger.info(f"Validation F1: {val_f1:.4f}")
            
            # 保存最好的模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(self.config['output_dir'])
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                # 只计算非padding部分
                mask = attention_mask.bool()
                predictions = predictions[mask].cpu().numpy()
                labels = labels[mask].cpu().numpy()
                
                all_preds.extend(predictions)
                all_labels.extend(labels)
        
        # 计算F1分数（忽略O标签）
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """预测单个文本的实体"""
        self.model.eval()
        
        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets = encoding['offset_mapping'][0]
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1)[0]
        
        # 解析实体
        entities = []
        current_entity = None
        
        for i, (pred_id, offset) in enumerate(zip(predictions, offsets)):
            start, end = offset.tolist()
            if start == 0 and end == 0:  # 特殊token
                continue
                
            label = self.dataset.id2label[pred_id.item()]
            
            if label.startswith('B-'):
                # 保存之前的实体
                if current_entity:
                    entities.append(current_entity)
                
                # 开始新实体
                entity_type = label[2:]
                current_entity = {
                    'text': text[start:end],
                    'type': entity_type,
                    'start': start,
                    'end': end
                }
            elif label.startswith('I-') and current_entity:
                # 继续当前实体
                if label[2:] == current_entity['type']:
                    current_entity['text'] = text[current_entity['start']:end]
                    current_entity['end'] = end
            else:
                # O标签，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后的实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def save_model(self, output_dir: str):
        """保存模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        
        # 保存配置
        with open(f"{output_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        # 保存标签映射
        with open(f"{output_dir}/label_mapping.pkl", 'wb') as f:
            pickle.dump({
                'label2id': self.dataset.label2id,
                'id2label': self.dataset.id2label
            }, f)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """加载模型"""
        # 加载配置
        with open(f"{model_dir}/config.json", 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载标签映射
        with open(f"{model_dir}/label_mapping.pkl", 'rb') as f:
            label_mapping = pickle.load(f)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # 加载模型
        num_labels = len(label_mapping['label2id'])
        self.model = FaultNERModel(
            self.config['model_name'], 
            num_labels,
            self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=self.device))
        
        # 创建虚拟数据集用于标签映射
        class DummyDataset:
            def __init__(self, label2id, id2label):
                self.label2id = label2id
                self.id2label = id2label
                self.num_labels = len(label2id)
        
        self.dataset = DummyDataset(label_mapping['label2id'], label_mapping['id2label'])
        
        logger.info(f"Model loaded from {model_dir}")
    
    def _collate_fn(self, batch):
        """数据整理函数"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }