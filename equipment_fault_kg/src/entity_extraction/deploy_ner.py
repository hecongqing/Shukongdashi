import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import logging
from typing import List, Dict, Tuple
import re

from .train_ner import NERModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERPredictor:
    """NER预测器类"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.label2id = None
        self.id2label = None
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 加载标签映射
            num_labels = checkpoint['label2id']
            self.label2id = {
                'O': 0,
                'B-COMPONENT': 1, 'I-COMPONENT': 2,
                'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
                'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
                'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
            
            # 加载模型
            self.model = NERModel('bert-base-chinese', num_labels)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # 预处理文本
        tokens = []
        char_to_token = []  # 字符到token的映射
        
        # 添加[CLS]标记
        tokens.append('[CLS]')
        char_to_token.append(-1)  # [CLS]不对应任何字符
        
        # 处理文本
        for char in text:
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            
            tokens.extend(sub_tokens)
            # 记录每个字符对应的token位置
            for _ in range(len(sub_tokens)):
                char_to_token.append(len(tokens) - 1)
        
        # 添加[SEP]标记
        tokens.append('[SEP]')
        char_to_token.append(-1)  # [SEP]不对应任何字符
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 预测
        with torch.no_grad():
            _, logits = self.model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
        # 解码预测结果
        pred_labels = predictions[0].cpu().numpy()
        
        # 提取实体
        entities = self._extract_entities(text, pred_labels, char_to_token)
        
        return entities
    
    def _extract_entities(self, text: str, pred_labels: List[int], char_to_token: List[int]) -> List[Dict]:
        """从预测标签中提取实体"""
        entities = []
        current_entity = None
        
        for i, char in enumerate(text):
            if i >= len(char_to_token) or char_to_token[i] == -1:
                continue
            
            token_idx = char_to_token[i]
            if token_idx >= len(pred_labels):
                continue
            
            label_id = pred_labels[token_idx]
            label = self.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'name': char,
                    'type': label[2:],  # 去掉B-前缀
                    'start_pos': i,
                    'end_pos': i + 1
                }
            
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                # 继续当前实体
                current_entity['name'] += char
                current_entity['end_pos'] = i + 1
            
            else:
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """批量预测"""
        results = []
        for text in texts:
            entities = self.predict(text)
            results.append(entities)
        return results

class EntityExtractor:
    """实体抽取器类 - 高级接口"""
    
    def __init__(self, model_path: str):
        self.predictor = NERPredictor(model_path)
        
        # 实体类型映射
        self.entity_type_mapping = {
            'COMPONENT': '部件单元',
            'PERFORMANCE': '性能表征',
            'FAULT_STATE': '故障状态',
            'DETECTION_TOOL': '检测工具'
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体"""
        entities = self.predictor.predict(text)
        
        # 转换格式
        result = []
        for entity in entities:
            result.append({
                'name': entity['name'],
                'type': self.entity_type_mapping.get(entity['type'], entity['type']),
                'start_pos': entity['start_pos'],
                'end_pos': entity['end_pos']
            })
        
        return result
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取实体"""
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results
    
    def get_entities_by_type(self, text: str, entity_type: str) -> List[str]:
        """根据类型获取实体"""
        entities = self.extract_entities(text)
        target_type = None
        
        # 反向映射
        for eng, chn in self.entity_type_mapping.items():
            if chn == entity_type:
                target_type = eng
                break
        
        if not target_type:
            return []
        
        return [entity['name'] for entity in entities if entity['type'] == entity_type]

def create_ner_api():
    """创建NER API服务"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # 全局变量存储模型
    entity_extractor = None
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/load_model', methods=['POST'])
    def load_model():
        global entity_extractor
        try:
            data = request.get_json()
            model_path = data.get('model_path')
            
            if not model_path:
                return jsonify({'error': 'model_path is required'}), 400
            
            entity_extractor = EntityExtractor(model_path)
            return jsonify({'message': 'Model loaded successfully'})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_entities', methods=['POST'])
    def extract_entities():
        global entity_extractor
        
        if not entity_extractor:
            return jsonify({'error': 'Model not loaded. Please load model first.'}), 400
        
        try:
            data = request.get_json()
            text = data.get('text')
            
            if not text:
                return jsonify({'error': 'text is required'}), 400
            
            entities = entity_extractor.extract_entities(text)
            return jsonify({
                'text': text,
                'entities': entities
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_entities_batch', methods=['POST'])
    def extract_entities_batch():
        global entity_extractor
        
        if not entity_extractor:
            return jsonify({'error': 'Model not loaded. Please load model first.'}), 400
        
        try:
            data = request.get_json()
            texts = data.get('texts')
            
            if not texts or not isinstance(texts, list):
                return jsonify({'error': 'texts must be a list'}), 400
            
            results = entity_extractor.extract_entities_batch(texts)
            return jsonify({
                'texts': texts,
                'results': results
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == "__main__":
    # 测试预测器
    sample_text = "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。"
    
    # 注意：这里需要先训练模型才能测试
    # predictor = NERPredictor('./ner_models/best_ner_model.pth')
    # entities = predictor.predict(sample_text)
    # print("预测的实体:")
    # for entity in entities:
    #     print(f"- {entity['name']} ({entity['type']}) at position {entity['start_pos']}-{entity['end_pos']}")
    
    # 创建API服务
    app = create_ner_api()
    app.run(host='0.0.0.0', port=5001, debug=True)