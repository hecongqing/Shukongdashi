"""
关系抽取模型
基于BERT架构实现实体关系抽取
支持多种关系抽取任务：分类、序列标注等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
import re
from sklearn.metrics import classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from loguru import logger

from config.settings import CONFIG


class RelationDataset(Dataset):
    """关系抽取数据集"""
    
    def __init__(self, texts: List[str], entity_pairs: List[Tuple[str, str]], 
                 relations: List[str], tokenizer: BertTokenizer, 
                 relation2id: Dict[str, int], max_length: int = 512):
        self.texts = texts
        self.entity_pairs = entity_pairs
        self.relations = relations
        self.tokenizer = tokenizer
        self.relation2id = relation2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        entity1, entity2 = self.entity_pairs[idx]
        relation = self.relations[idx]
        
        # 标记实体位置
        marked_text = self._mark_entities(text, entity1, entity2)
        
        # 编码文本
        encoding = self.tokenizer(
            marked_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 关系标签
        relation_id = self.relation2id.get(relation, self.relation2id.get('其他', 0))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_id': torch.tensor(relation_id, dtype=torch.long),
            'text': text,
            'entity1': entity1,
            'entity2': entity2,
            'relation': relation
        }
    
    def _mark_entities(self, text: str, entity1: str, entity2: str) -> str:
        """在文本中标记实体位置"""
        # 使用特殊标记标记实体
        # 实体1用 [E1] entity [/E1] 标记
        # 实体2用 [E2] entity [/E2] 标记
        
        marked_text = text
        
        # 先标记较长的实体，避免重叠问题
        entities = [(entity1, "[E1]", "[/E1]"), (entity2, "[E2]", "[/E2]")]
        entities.sort(key=lambda x: len(x[0]), reverse=True)
        
        for entity, start_marker, end_marker in entities:
            if entity in marked_text:
                # 只替换第一个匹配项
                marked_text = marked_text.replace(
                    entity, 
                    f"{start_marker} {entity} {end_marker}", 
                    1
                )
        
        return marked_text


class RelationClassifier(nn.Module):
    """基于BERT的关系分类器"""
    
    def __init__(self, bert_model_name: str, num_relations: int, 
                 hidden_dim: int = 256, dropout: float = 0.1):
        super(RelationClassifier, self).__init__()
        
        self.num_relations = num_relations
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_relations)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, relation_id=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS]token的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        outputs = (logits,)
        
        if relation_id is not None:
            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, relation_id)
            outputs = (loss,) + outputs
        
        return outputs


class EntityMarkerRelationExtractor(nn.Module):
    """基于实体标记的关系抽取器"""
    
    def __init__(self, bert_model_name: str, num_relations: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
        super(EntityMarkerRelationExtractor, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size
        
        # 实体表示层
        self.entity_dense = nn.Linear(self.bert_dim, hidden_dim)
        
        # 关系分类层
        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask, entity1_mask=None, 
                entity2_mask=None, relation_id=None):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 提取实体表示
        if entity1_mask is not None and entity2_mask is not None:
            entity1_repr = self._extract_entity_representation(
                sequence_output, entity1_mask
            )
            entity2_repr = self._extract_entity_representation(
                sequence_output, entity2_mask
            )
        else:
            # 使用[CLS]表示作为备选
            entity1_repr = outputs.pooler_output
            entity2_repr = outputs.pooler_output
        
        # 实体表示变换
        entity1_repr = self.entity_dense(entity1_repr)
        entity2_repr = self.entity_dense(entity2_repr)
        
        # 连接实体表示
        combined_repr = torch.cat([entity1_repr, entity2_repr], dim=1)
        
        # 关系分类
        logits = self.relation_classifier(combined_repr)
        
        outputs = (logits,)
        
        if relation_id is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, relation_id)
            outputs = (loss,) + outputs
        
        return outputs
    
    def _extract_entity_representation(self, sequence_output, entity_mask):
        """提取实体的表示"""
        # 使用平均池化提取实体表示
        entity_mask = entity_mask.unsqueeze(-1).float()
        entity_repr = (sequence_output * entity_mask).sum(1) / entity_mask.sum(1)
        return entity_repr


class RelationExtractionTrainer:
    """关系抽取模型训练器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["model"]["relation_extraction"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
        
        # 添加特殊标记
        special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # 关系映射
        self.relations = self.config["relations"]
        self.relation2id = {rel: i for i, rel in enumerate(self.relations)}
        self.id2relation = {i: rel for rel, i in self.relation2id.items()}
        
        # 模型保存路径
        self.model_path = Path(self.config["model_path"])
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """加载训练数据"""
        texts, entity_pairs, relations = [], [], []
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                texts.append(item['text'])
                entity_pairs.append((item['entity1'], item['entity2']))
                relations.append(item['relation'])
        
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
            for _, row in df.iterrows():
                texts.append(row['text'])
                entity_pairs.append((row['entity1'], row['entity2']))
                relations.append(row['relation'])
        
        else:
            # 自定义格式：每行包含 text \t entity1 \t entity2 \t relation
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        texts.append(parts[0])
                        entity_pairs.append((parts[1], parts[2]))
                        relations.append(parts[3])
        
        logger.info(f"Loaded {len(texts)} relation examples")
        return texts, entity_pairs, relations
    
    def create_dataloader(self, texts: List[str], entity_pairs: List[Tuple[str, str]], 
                         relations: List[str], batch_size: int, 
                         shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = RelationDataset(
            texts=texts,
            entity_pairs=entity_pairs,
            relations=relations,
            tokenizer=self.tokenizer,
            relation2id=self.relation2id,
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
        relation_ids = torch.stack([item['relation_id'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'relation_ids': relation_ids,
            'texts': [item['text'] for item in batch],
            'entity1s': [item['entity1'] for item in batch],
            'entity2s': [item['entity2'] for item in batch],
            'relations': [item['relation'] for item in batch]
        }
    
    def train(self, train_data_path: str, val_data_path: str = None, 
              model_type: str = "classifier"):
        """训练模型"""
        logger.info("Starting relation extraction model training...")
        
        # 加载数据
        train_texts, train_entity_pairs, train_relations = self.load_data(train_data_path)
        
        if val_data_path:
            val_texts, val_entity_pairs, val_relations = self.load_data(val_data_path)
        else:
            # 分割验证集
            split_idx = int(len(train_texts) * 0.8)
            val_texts = train_texts[split_idx:]
            val_entity_pairs = train_entity_pairs[split_idx:]
            val_relations = train_relations[split_idx:]
            train_texts = train_texts[:split_idx]
            train_entity_pairs = train_entity_pairs[:split_idx]
            train_relations = train_relations[:split_idx]
        
        # 创建数据加载器
        train_dataloader = self.create_dataloader(
            train_texts, train_entity_pairs, train_relations,
            self.config["batch_size"], shuffle=True
        )
        val_dataloader = self.create_dataloader(
            val_texts, val_entity_pairs, val_relations,
            self.config["batch_size"], shuffle=False
        )
        
        # 初始化模型
        if model_type == "classifier":
            model = RelationClassifier(
                bert_model_name=self.config["model_name"],
                num_relations=len(self.relations)
            )
        else:
            model = EntityMarkerRelationExtractor(
                bert_model_name=self.config["model_name"],
                num_relations=len(self.relations)
            )
        
        # 调整BERT embedding大小（由于添加了特殊标记）
        model.bert.resize_token_embeddings(len(self.tokenizer))
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
        val_f1s = []
        
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 训练
            model.train()
            total_train_loss = 0
            
            for batch in train_dataloader:
                # 移到GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                relation_ids = batch['relation_ids'].to(self.device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    relation_id=relation_ids
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
            val_loss, val_accuracy, val_f1 = self.evaluate(model, val_dataloader)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model(model)
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
        
        # 绘制训练曲线
        self.plot_training_curve(train_losses, val_losses, val_f1s)
        
        logger.info("Training completed!")
        return model
    
    def evaluate(self, model, dataloader) -> Tuple[float, float, float]:
        """评估模型"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                relation_ids = batch['relation_ids'].to(self.device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    relation_id=relation_ids
                )
                
                loss = outputs[0]
                logits = outputs[1]
                
                total_loss += loss.item()
                
                # 预测
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(relation_ids.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def save_model(self, model):
        """保存模型"""
        # 保存模型权重
        torch.save(model.state_dict(), self.model_path / "model.pth")
        
        # 保存配置
        config_data = {
            "model_config": self.config,
            "relation2id": self.relation2id,
            "id2relation": self.id2relation,
            "relations": self.relations
        }
        
        with open(self.model_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(self.model_path)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, model_path: str = None, model_type: str = "classifier"):
        """加载模型"""
        if model_path is None:
            model_path = self.model_path
        else:
            model_path = Path(model_path)
        
        # 加载配置
        with open(model_path / "config.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.relation2id = config_data["relation2id"]
        self.id2relation = config_data["id2relation"]
        self.relations = config_data["relations"]
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(str(model_path))
        
        # 加载模型
        if model_type == "classifier":
            model = RelationClassifier(
                bert_model_name=self.config["model_name"],
                num_relations=len(self.relations)
            )
        else:
            model = EntityMarkerRelationExtractor(
                bert_model_name=self.config["model_name"],
                num_relations=len(self.relations)
            )
        
        model.bert.resize_token_embeddings(len(self.tokenizer))
        model.load_state_dict(torch.load(model_path / "model.pth", map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def predict(self, text: str, entity1: str, entity2: str, model=None) -> Tuple[str, float]:
        """预测实体关系"""
        if model is None:
            model = self.load_model()
        
        # 准备数据
        dataset = RelationDataset(
            texts=[text],
            entity_pairs=[(entity1, entity2)],
            relations=['其他'],  # 占位符
            tokenizer=self.tokenizer,
            relation2id=self.relation2id,
            max_length=self.config["max_length"]
        )
        
        data = dataset[0]
        input_ids = data['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = data['attention_mask'].unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            probabilities = F.softmax(logits, dim=1)
            
            predicted_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_relation = self.id2relation[predicted_id]
        
        return predicted_relation, confidence
    
    def batch_predict(self, texts: List[str], entity_pairs: List[Tuple[str, str]], 
                     model=None) -> List[Tuple[str, float]]:
        """批量预测实体关系"""
        if model is None:
            model = self.load_model()
        
        # 创建数据加载器
        dummy_relations = ['其他'] * len(texts)
        dataloader = self.create_dataloader(
            texts, entity_pairs, dummy_relations,
            batch_size=self.config["batch_size"], shuffle=False
        )
        
        results = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs[0]
                probabilities = F.softmax(logits, dim=1)
                
                predicted_ids = torch.argmax(logits, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                for pred_id, confidence in zip(predicted_ids, confidences):
                    predicted_relation = self.id2relation[pred_id.item()]
                    results.append((predicted_relation, confidence.item()))
        
        return results
    
    def plot_training_curve(self, train_losses: List[float], val_losses: List[float], 
                           val_f1s: List[float]):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 损失曲线
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1分数曲线
        ax2.plot(val_f1s, label='Validation F1', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_path / "training_curve.png")
        plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Relation Extraction Model Training")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--train-data", help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--model-type", default="classifier", 
                       choices=["classifier", "entity_marker"], 
                       help="Model type")
    parser.add_argument("--predict", help="Text for prediction")
    parser.add_argument("--entity1", help="First entity")
    parser.add_argument("--entity2", help="Second entity")
    parser.add_argument("--model-path", help="Model path for loading")
    
    args = parser.parse_args()
    
    trainer = RelationExtractionTrainer()
    
    if args.train and args.train_data:
        trainer.train(args.train_data, args.val_data, args.model_type)
    
    if args.predict and args.entity1 and args.entity2:
        model = trainer.load_model(args.model_path, args.model_type)
        relation, confidence = trainer.predict(args.predict, args.entity1, args.entity2, model)
        
        print(f"Predicted relation: {relation}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()