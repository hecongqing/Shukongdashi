"""
专用于故障数据集的关系抽取模型

支持4种关系类型：
- 部件故障 (Component Fault): 部件单元 -> 故障状态
- 性能故障 (Performance Fault): 性能表征 -> 故障状态  
- 检测工具 (Detection Tool): 检测工具 -> 性能表征
- 组成 (Composition): 部件单元 -> 部件单元
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from loguru import logger
import pickle
from tqdm import tqdm
import itertools


class FaultRelationDataset(Dataset):
    """故障关系抽取数据集"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, is_training: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # 关系类型映射
        self.relation2id = {
            'No_Relation': 0,
            '部件故障': 1,
            '性能故障': 2,
            '检测工具': 3,
            '组成': 4
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.num_relations = len(self.relation2id)
        
        # 处理数据
        self.processed_data = self._process_data()
    
    def _process_data(self) -> List[Dict]:
        """处理原始数据，生成实体对和关系标签"""
        processed = []
        
        for item in self.data:
            text = item['text']
            spo_list = item.get('spo_list', [])
            
            # 提取所有实体
            entities = self._extract_entities_from_spo(spo_list)
            
            if self.is_training:
                # 训练时：根据spo_list生成正负样本
                positive_pairs = []
                for spo in spo_list:
                    h_entity = spo['h']
                    t_entity = spo['t']
                    relation = spo['relation']
                    
                    positive_pairs.append({
                        'text': text,
                        'h_entity': h_entity,
                        't_entity': t_entity,
                        'relation': relation
                    })
                
                # 生成负样本（随机实体对，标记为No_Relation）
                negative_pairs = self._generate_negative_samples(text, entities, positive_pairs)
                
                processed.extend(positive_pairs)
                processed.extend(negative_pairs)
            else:
                # 推理时：生成所有可能的实体对
                for i, h_entity in enumerate(entities):
                    for j, t_entity in enumerate(entities):
                        if i != j:  # 不同的实体
                            processed.append({
                                'text': text,
                                'h_entity': h_entity,
                                't_entity': t_entity,
                                'relation': 'No_Relation'  # 推理时先默认为无关系
                            })
        
        return processed
    
    def _extract_entities_from_spo(self, spo_list: List[Dict]) -> List[Dict]:
        """从spo_list中提取所有实体"""
        entities = []
        seen_entities = set()
        
        for spo in spo_list:
            for entity_key in ['h', 't']:
                entity = spo[entity_key]
                entity_id = (entity['name'], tuple(entity['pos']))
                
                if entity_id not in seen_entities:
                    entities.append(entity)
                    seen_entities.add(entity_id)
        
        return entities
    
    def _generate_negative_samples(self, text: str, entities: List[Dict], positive_pairs: List[Dict]) -> List[Dict]:
        """生成负样本"""
        negative_pairs = []
        
        # 创建正样本的集合用于去重
        positive_set = set()
        for pair in positive_pairs:
            h_pos = tuple(pair['h_entity']['pos'])
            t_pos = tuple(pair['t_entity']['pos'])
            positive_set.add((h_pos, t_pos))
        
        # 生成负样本
        for i, h_entity in enumerate(entities):
            for j, t_entity in enumerate(entities):
                if i != j:
                    h_pos = tuple(h_entity['pos'])
                    t_pos = tuple(t_entity['pos'])
                    
                    if (h_pos, t_pos) not in positive_set:
                        negative_pairs.append({
                            'text': text,
                            'h_entity': h_entity,
                            't_entity': t_entity,
                            'relation': 'No_Relation'
                        })
        
        # 限制负样本数量，避免数据不平衡
        max_negative = len(positive_pairs) * 2
        if len(negative_pairs) > max_negative:
            negative_pairs = negative_pairs[:max_negative]
        
        return negative_pairs
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        text = item['text']
        h_entity = item['h_entity']
        t_entity = item['t_entity']
        relation = item['relation']
        
        # 构造输入文本：[CLS] text [SEP] h_entity [SEP] t_entity [SEP]
        h_name = h_entity['name']
        t_name = t_entity['name']
        
        # 标记实体位置
        h_start, h_end = h_entity['pos']
        t_start, t_end = t_entity['pos']
        
        # 构造输入序列
        input_text = f"{text} [SEP] {h_name} [SEP] {t_name}"
        
        # 分词
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 关系标签
        relation_id = self.relation2id[relation]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_label': torch.tensor(relation_id, dtype=torch.long),
            'text': text,
            'h_entity': h_name,
            't_entity': t_name,
            'h_pos': (h_start, h_end),
            't_pos': (t_start, t_end)
        }


class FaultRelationModel(nn.Module):
    """故障关系分类模型"""
    
    def __init__(self, model_name: str, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
        self.num_relations = num_relations
    
    def forward(self, input_ids, attention_mask, relation_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token的表示
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if relation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, relation_labels)
        
        return {'loss': loss, 'logits': logits}


class FaultRelationTrainer:
    """故障关系抽取训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.model = None
        self.dataset = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_data(self, data_path: str, is_training: bool = True):
        """加载训练数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        
        self.dataset = FaultRelationDataset(
            data, self.tokenizer, self.config['max_length'], is_training
        )
        logger.info(f"Loaded {len(data)} samples, generated {len(self.dataset)} pairs")
        
        # 初始化模型
        if self.model is None:
            self.model = FaultRelationModel(
                self.config['model_name'], 
                self.dataset.num_relations,
                self.config['dropout']
            ).to(self.device)
    
    def train(self, train_data_path: str, val_data_path: str = None):
        """训练模型"""
        # 加载训练数据
        self.load_data(train_data_path, is_training=True)
        
        # 数据分割
        if val_data_path:
            with open(val_data_path, 'r', encoding='utf-8') as f:
                val_data = [json.loads(line.strip()) for line in f]
            val_dataset = FaultRelationDataset(
                val_data, self.tokenizer, self.config['max_length'], is_training=True
            )
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
            val_metrics = self._evaluate(val_loader)
            logger.info(f"Validation - F1: {val_metrics['f1']:.4f}, "
                       f"Precision: {val_metrics['precision']:.4f}, "
                       f"Recall: {val_metrics['recall']:.4f}")
            
            # 保存最好的模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
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
            relation_labels = batch['relation_label'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask, relation_labels)
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
                relation_labels = batch['relation_label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(relation_labels.cpu().numpy())
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def predict(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """预测文本中实体间的关系"""
        self.model.eval()
        
        relations = []
        
        # 生成所有实体对
        for i, h_entity in enumerate(entities):
            for j, t_entity in enumerate(entities):
                if i != j:  # 不同的实体
                    # 构造输入
                    h_name = h_entity['text']
                    t_name = t_entity['text']
                    input_text = f"{text} [SEP] {h_name} [SEP] {t_name}"
                    
                    # 分词
                    encoding = self.tokenizer(
                        input_text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.config['max_length'],
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_ids, attention_mask)
                        logits = outputs['logits']
                        probabilities = torch.softmax(logits, dim=-1)
                        predicted_id = torch.argmax(logits, dim=-1).item()
                        confidence = probabilities[0][predicted_id].item()
                    
                    predicted_relation = self.dataset.id2relation[predicted_id]
                    
                    # 只保留有意义的关系（非No_Relation且置信度较高）
                    if predicted_relation != 'No_Relation' and confidence > 0.5:
                        relations.append({
                            'h': {
                                'name': h_name,
                                'type': h_entity['type'],
                                'pos': [h_entity['start'], h_entity['end']]
                            },
                            't': {
                                'name': t_name,
                                'type': t_entity['type'],
                                'pos': [t_entity['start'], t_entity['end']]
                            },
                            'relation': predicted_relation,
                            'confidence': confidence
                        })
        
        return relations
    
    def predict_file(self, input_file: str, entity_file: str = None) -> List[Dict[str, Any]]:
        """对文件中的数据进行关系预测"""
        # 加载文本数据
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [json.loads(line.strip()) for line in f]
        
        # 加载实体数据（如果提供）
        entities_data = {}
        if entity_file:
            with open(entity_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    entities_data[data['ID']] = data.get('entities', [])
        
        results = []
        
        for item in texts:
            text_id = item['ID']
            text = item['text']
            
            # 获取实体
            if text_id in entities_data:
                entities = entities_data[text_id]
            else:
                # 如果没有提供实体文件，需要先进行实体识别
                logger.warning(f"No entities found for {text_id}, skipping relation extraction")
                continue
            
            # 预测关系
            relations = self.predict(text, entities)
            
            results.append({
                'ID': text_id,
                'text': text,
                'relations': relations
            })
        
        return results
    
    def save_model(self, output_dir: str):
        """保存模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save(self.model.state_dict(), f"{output_dir}/model.pt")
        
        # 保存配置
        with open(f"{output_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        # 保存关系映射
        with open(f"{output_dir}/relation_mapping.pkl", 'wb') as f:
            pickle.dump({
                'relation2id': self.dataset.relation2id,
                'id2relation': self.dataset.id2relation
            }, f)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """加载模型"""
        # 加载配置
        with open(f"{model_dir}/config.json", 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 加载关系映射
        with open(f"{model_dir}/relation_mapping.pkl", 'rb') as f:
            relation_mapping = pickle.load(f)
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        
        # 加载模型
        num_relations = len(relation_mapping['relation2id'])
        self.model = FaultRelationModel(
            self.config['model_name'], 
            num_relations,
            self.config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=self.device))
        
        # 创建虚拟数据集用于关系映射
        class DummyDataset:
            def __init__(self, relation2id, id2relation):
                self.relation2id = relation2id
                self.id2relation = id2relation
                self.num_relations = len(relation2id)
        
        self.dataset = DummyDataset(relation_mapping['relation2id'], relation_mapping['id2relation'])
        
        logger.info(f"Model loaded from {model_dir}")
    
    def _collate_fn(self, batch):
        """数据整理函数"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        relation_labels = torch.stack([item['relation_label'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'relation_label': relation_labels
        }