import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
import jieba
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger
import json
import pickle
from pathlib import Path

from backend.config.settings import get_settings
from backend.models.entity_models import EntityResult, EntityType
from backend.utils.text_utils import TextPreprocessor

class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.text_preprocessor = TextPreprocessor()
        
        # 预定义的实体类型
        self.entity_types = {
            "EQUIPMENT": "设备",
            "FAULT_SYMPTOM": "故障现象", 
            "FAULT_CAUSE": "故障原因",
            "REPAIR_METHOD": "维修方法",
            "PART": "零部件",
            "OPERATION": "操作",
            "ALARM_CODE": "报警代码",
            "PARAMETER": "参数",
            "LOCATION": "位置",
            "TIME": "时间"
        }
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载预训练模型"""
        try:
            model_path = self.settings.NER_MODEL_PATH
            
            if Path(model_path).exists():
                # 加载自定义训练的模型
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
                self.model = BertForTokenClassification.from_pretrained(model_path)
                
                # 加载标签映射
                with open(f"{model_path}/label_mapping.json", "r") as f:
                    label_mapping = json.load(f)
                    self.label_to_id = label_mapping["label_to_id"]
                    self.id_to_label = label_mapping["id_to_label"]
            else:
                # 使用预训练模型
                model_name = "hfl/chinese-bert-wwm-ext"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                
                # 创建默认标签映射
                self._create_default_labels()
            
            self.model.eval()
            logger.info("实体抽取模型加载成功")
            
        except Exception as e:
            logger.error(f"加载实体抽取模型失败: {e}")
            raise
    
    def _create_default_labels(self):
        """创建默认标签映射"""
        labels = ["O"]  # Outside
        for entity_type in self.entity_types.keys():
            labels.append(f"B-{entity_type}")  # Begin
            labels.append(f"I-{entity_type}")  # Inside
        
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}
    
    def extract_entities(self, text: str) -> List[EntityResult]:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            List[EntityResult]: 抽取的实体列表
        """
        try:
            # 文本预处理
            processed_text = self.text_preprocessor.clean_text(text)
            
            # 分词
            tokens = self.tokenizer.tokenize(processed_text)
            
            # 编码
            encoded = self.tokenizer.encode_plus(
                processed_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # 模型预测
            with torch.no_grad():
                outputs = self.model(**encoded)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # 解码预测结果
            entities = self._decode_predictions(
                tokens, 
                predictions[0].numpy(), 
                processed_text
            )
            
            return entities
            
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []
    
    def _decode_predictions(self, tokens: List[str], predictions: np.ndarray, text: str) -> List[EntityResult]:
        """解码预测结果"""
        entities = []
        current_entity = None
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if pred_id >= len(self.id_to_label):
                continue
                
            label = self.id_to_label[pred_id]
            
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = EntityResult(
                    text=token,
                    entity_type=entity_type,
                    start_pos=i,
                    end_pos=i + 1,
                    confidence=0.9  # 临时设置
                )
            elif label.startswith("I-"):
                if current_entity and current_entity.entity_type == label[2:]:
                    current_entity.text += token
                    current_entity.end_pos = i + 1
        
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract_entities_with_rules(self, text: str) -> List[EntityResult]:
        """
        基于规则的实体抽取
        
        Args:
            text: 输入文本
            
        Returns:
            List[EntityResult]: 抽取的实体列表
        """
        entities = []
        
        # 设备名称模式
        equipment_patterns = [
            r'[A-Z]{2,}[-_]?[A-Z0-9]{2,}',  # 如 FANUC-0i, SIEMENS-840D
            r'数控[机床|车床|铣床|加工中心]+',
            r'[加工中心|车床|铣床|钻床|磨床]+',
        ]
        
        # 故障代码模式
        alarm_patterns = [
            r'[A-Z]{2,}\d{3,}',  # 如 ALM401, ERR1234
            r'报警\d{3,}',
            r'错误代码\d{3,}',
        ]
        
        # 故障现象模式
        symptom_patterns = [
            r'[不能|无法|不正常|异常|故障|错误|报警]+.{0,20}',
            r'.{0,20}[停机|停止|中断|失效|损坏]+',
        ]
        
        # 应用规则
        entities.extend(self._apply_patterns(text, equipment_patterns, "EQUIPMENT"))
        entities.extend(self._apply_patterns(text, alarm_patterns, "ALARM_CODE"))
        entities.extend(self._apply_patterns(text, symptom_patterns, "FAULT_SYMPTOM"))
        
        return entities
    
    def _apply_patterns(self, text: str, patterns: List[str], entity_type: str) -> List[EntityResult]:
        """应用正则表达式模式"""
        entities = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = EntityResult(
                    text=match.group(),
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    def extract_entities_hybrid(self, text: str) -> List[EntityResult]:
        """
        混合实体抽取：结合深度学习和规则
        
        Args:
            text: 输入文本
            
        Returns:
            List[EntityResult]: 抽取的实体列表
        """
        # 深度学习抽取
        ml_entities = self.extract_entities(text)
        
        # 规则抽取
        rule_entities = self.extract_entities_with_rules(text)
        
        # 合并去重
        all_entities = ml_entities + rule_entities
        merged_entities = self._merge_entities(all_entities)
        
        return merged_entities
    
    def _merge_entities(self, entities: List[EntityResult]) -> List[EntityResult]:
        """合并重复实体"""
        if not entities:
            return []
        
        # 按位置排序
        entities.sort(key=lambda x: x.start_pos)
        
        merged = []
        for entity in entities:
            if not merged:
                merged.append(entity)
                continue
            
            last_entity = merged[-1]
            
            # 检查是否重叠
            if (entity.start_pos < last_entity.end_pos and 
                entity.end_pos > last_entity.start_pos):
                # 合并实体，保留置信度更高的
                if entity.confidence > last_entity.confidence:
                    merged[-1] = entity
            else:
                merged.append(entity)
        
        return merged
    
    def batch_extract_entities(self, texts: List[str]) -> List[List[EntityResult]]:
        """
        批量实体抽取
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[EntityResult]]: 每个文本的实体列表
        """
        results = []
        for text in texts:
            entities = self.extract_entities_hybrid(text)
            results.append(entities)
        
        return results
    
    def save_model(self, save_path: str):
        """保存模型"""
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # 保存标签映射
            label_mapping = {
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label
            }
            with open(f"{save_path}/label_mapping.json", "w") as f:
                json.dump(label_mapping, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模型保存成功: {save_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

class EntityExtractionService:
    """实体抽取服务"""
    
    def __init__(self):
        self.extractor = EntityExtractor()
    
    async def extract_entities(self, text: str) -> List[EntityResult]:
        """抽取实体"""
        return self.extractor.extract_entities_hybrid(text)
    
    async def batch_extract_entities(self, texts: List[str]) -> List[List[EntityResult]]:
        """批量抽取实体"""
        return self.extractor.batch_extract_entities(texts)
    
    async def get_entity_types(self) -> Dict[str, str]:
        """获取实体类型"""
        return self.extractor.entity_types
    
    async def update_entity_rules(self, rules: Dict[str, List[str]]):
        """更新实体规则"""
        # 这里可以实现动态更新规则的逻辑
        pass