import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import logging
from typing import List, Dict, Tuple
import re
import os

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
            
            # 加载标签映射 - 保持与训练时一致
            self.label2id = {
                'O': 0,
                'B-COMPONENT': 1, 'I-COMPONENT': 2,
                'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
                'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
                'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
            }
            self.id2label = {v: k for k, v in self.label2id.items()}
            
            # 获取标签数量
            num_labels = len(self.label2id)
            
            # 加载模型
            self.model = NERModel('bert-base-chinese', num_labels)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Number of labels: {num_labels}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # 预处理文本 - 与训练时保持一致
        tokens = []
        char_to_token = []  # 字符到token的映射
        
        # 添加[CLS]标记
        tokens.append('[CLS]')
        
        # 处理文本 - 按字符逐个处理，与训练时一致
        for i, char in enumerate(text):
            # 记录当前字符对应的token起始位置
            char_start_token = len(tokens)
            
            # 分词
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            
            tokens.extend(sub_tokens)
            
            # 记录字符对应的token位置（使用第一个sub-token的位置）
            char_to_token.append(char_start_token)
        
        # 添加[SEP]标记
        tokens.append('[SEP]')
        
        # 截断处理
        max_length = 512
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # 填充到最大长度
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([0] * padding_length)  # 0 是 [PAD] token 的 ID
            attention_mask.extend([0] * padding_length)
        
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
            # 检查索引范围
            if i >= len(char_to_token):
                continue
            
            # 获取字符对应的token位置（跳过[CLS]，所以+1）
            token_idx = char_to_token[i] + 1  # +1 是因为第一个token是[CLS]
            
            # 检查token索引范围
            if token_idx >= len(pred_labels):
                continue
            
            label_id = pred_labels[token_idx]
            label = self.id2label.get(label_id, 'O')
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # 去掉B-前缀
                current_entity = {
                    'name': char,
                    'type': entity_type,
                    'start_pos': i,
                    'end_pos': i + 1
                }
            
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]  # 去掉I-前缀
                # 继续当前实体（只有当类型匹配时）
                if entity_type == current_entity['type']:
                    current_entity['name'] += char
                    current_entity['end_pos'] = i + 1
                else:
                    # 类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            
            elif label == 'O':
                # 非实体标签，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        # 后处理：过滤明显错误的实体
        filtered_entities = []
        for entity in entities:
            # 过滤标点符号实体
            if entity['name'] in ['。', '，', '；', '：', '！', '？', '、', '（', '）', '【', '】']:
                continue
            
            # 过滤单字符动词
            if len(entity['name']) == 1 and entity['name'] in ['使', '用', '检', '测', '修', '维']:
                continue
            
            # 过滤包含标点符号的实体（保留实体部分）
            if any(punct in entity['name'] for punct in ['。', '，', '；', '：', '！', '？', '、']):
                # 移除末尾的标点符号
                clean_name = entity['name'].rstrip('。，；：！？、')
                if clean_name and clean_name != entity['name']:
                    entity['name'] = clean_name
                    entity['end_pos'] = entity['start_pos'] + len(clean_name)
            
            # 过滤空实体
            if entity['name'].strip():
                filtered_entities.append(entity)
        
        return filtered_entities
    
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
    
    print(f"测试文本: {sample_text}")
    print(f"文本长度: {len(sample_text)} 字符")
    
    # 测试tokenization
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    print("\n=== Tokenization 测试 ===")
    tokens = ['[CLS]']
    char_to_token = []
    
    for i, char in enumerate(sample_text):
        char_start_token = len(tokens)
        sub_tokens = tokenizer.tokenize(char)
        if not sub_tokens:
            sub_tokens = ['[UNK]']
        tokens.extend(sub_tokens)
        char_to_token.append(char_start_token)
        print(f"字符 '{char}' -> tokens: {sub_tokens}, token_pos: {char_start_token}")
    
    tokens.append('[SEP]')
    print(f"\n总tokens数: {len(tokens)}")
    print(f"字符到token映射长度: {len(char_to_token)}")
    
    # 注意：需要训练好的模型文件才能测试预测
    try:
        model_path = './ner_models/best_ner_model.pth'
        if os.path.exists(model_path):
            print(f"\n=== NER 预测测试 ===")
            predictor = NERPredictor(model_path)
            entities = predictor.predict(sample_text)
            print("预测的实体:")
            for entity in entities:
                print(f"- {entity['name']} ({entity['type']}) at position {entity['start_pos']}-{entity['end_pos']}")
        else:
            print(f"\n模型文件不存在: {model_path}")
            print("请先训练模型再测试预测功能")
    except Exception as e:
        print(f"\n预测测试失败: {e}")
    
    # 创建API服务
    print(f"\n=== 启动 API 服务 ===")
    app = create_ner_api()
    app.run(host='0.0.0.0', port=5001, debug=True)