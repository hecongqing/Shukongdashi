import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import logging
from typing import List, Dict, Tuple
import re
import itertools

from .train_relation import RelationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationPredictor:
    """关系抽取预测器类"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.relation2id = None
        self.id2relation = None
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 加载tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 加载关系映射
            num_relations = checkpoint['relation2id']
            self.relation2id = {
                "部件故障": 0,
                "性能故障": 1,
                "检测工具": 2,
                "组成": 3
            }
            self.id2relation = {v: k for k, v in self.relation2id.items()}
            
            # 加载模型
            self.model = RelationModel('bert-base-chinese', num_relations)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_relation(self, text: str, head_entity: str, tail_entity: str) -> Dict:
        """预测两个实体之间的关系"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        # 构建输入文本
        input_text = f"{head_entity} [SEP] {tail_entity} [SEP] {text}"
        
        # 编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            _, binary_logits, relation_logits = self.model(input_ids, attention_mask)
            
            # 二分类预测
            binary_probs = torch.softmax(binary_logits, dim=-1)
            binary_pred = torch.argmax(binary_logits, dim=-1).item()
            binary_confidence = binary_probs[0][binary_pred].item()
            
            # 关系分类预测
            relation_probs = torch.softmax(relation_logits, dim=-1)
            relation_pred = torch.argmax(relation_logits, dim=-1).item()
            relation_confidence = relation_probs[0][relation_pred].item()
            
            relation_type = self.id2relation.get(relation_pred, "未知")
        
        return {
            'head_entity': head_entity,
            'tail_entity': tail_entity,
            'has_relation': bool(binary_pred),
            'relation_type': relation_type,
            'binary_confidence': binary_confidence,
            'relation_confidence': relation_confidence
        }
    
    def predict_all_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """预测文本中所有实体对之间的关系"""
        if not entities or len(entities) < 2:
            return []
        
        relations = []
        
        # 生成所有实体对
        entity_pairs = list(itertools.combinations(entities, 2))
        
        for head_entity, tail_entity in entity_pairs:
            # 预测关系
            relation = self.predict_relation(
                text, 
                head_entity['name'], 
                tail_entity['name']
            )
            
            # 只保留有关系的预测结果
            if relation['has_relation'] and relation['binary_confidence'] > 0.5:
                relations.append(relation)
        
        return relations

class RelationExtractor:
    """关系抽取器类 - 高级接口"""
    
    def __init__(self, model_path: str):
        self.predictor = RelationPredictor(model_path)
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """抽取文本中的关系"""
        relations = self.predictor.predict_all_relations(text, entities)
        
        # 转换格式
        result = []
        for relation in relations:
            result.append({
                'head_entity': relation['head_entity'],
                'tail_entity': relation['tail_entity'],
                'relation_type': relation['relation_type'],
                'confidence': relation['relation_confidence']
            })
        
        return result
    
    def extract_relations_batch(self, texts: List[str], entities_list: List[List[Dict]]) -> List[List[Dict]]:
        """批量抽取关系"""
        results = []
        for text, entities in zip(texts, entities_list):
            relations = self.extract_relations(text, entities)
            results.append(relations)
        return results
    
    def get_relations_by_type(self, text: str, entities: List[Dict], relation_type: str) -> List[Dict]:
        """根据关系类型获取关系"""
        all_relations = self.extract_relations(text, entities)
        return [rel for rel in all_relations if rel['relation_type'] == relation_type]

class JointExtractor:
    """联合抽取器类 - 结合实体抽取和关系抽取"""
    
    def __init__(self, ner_model_path: str, relation_model_path: str):
        from ..entity_extraction.deploy_ner import EntityExtractor
        
        self.entity_extractor = EntityExtractor(ner_model_path)
        self.relation_extractor = RelationExtractor(relation_model_path)
    
    def extract_spo(self, text: str) -> Dict:
        """抽取SPO三元组"""
        # 抽取实体
        entities = self.entity_extractor.extract_entities(text)
        
        # 抽取关系
        relations = self.relation_extractor.extract_relations(text, entities)
        
        # 构建SPO列表
        spo_list = []
        for relation in relations:
            spo_list.append({
                'h': {'name': relation['head_entity']},
                't': {'name': relation['tail_entity']},
                'relation': relation['relation_type']
            })
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations,
            'spo_list': spo_list
        }
    
    def extract_spo_batch(self, texts: List[str]) -> List[Dict]:
        """批量抽取SPO三元组"""
        results = []
        for text in texts:
            result = self.extract_spo(text)
            results.append(result)
        return results

def create_relation_api():
    """创建关系抽取API服务"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # 全局变量存储模型
    relation_extractor = None
    joint_extractor = None
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/load_model', methods=['POST'])
    def load_model():
        global relation_extractor
        try:
            data = request.get_json()
            model_path = data.get('model_path')
            
            if not model_path:
                return jsonify({'error': 'model_path is required'}), 400
            
            relation_extractor = RelationExtractor(model_path)
            return jsonify({'message': 'Model loaded successfully'})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/load_joint_model', methods=['POST'])
    def load_joint_model():
        global joint_extractor
        try:
            data = request.get_json()
            ner_model_path = data.get('ner_model_path')
            relation_model_path = data.get('relation_model_path')
            
            if not ner_model_path or not relation_model_path:
                return jsonify({'error': 'ner_model_path and relation_model_path are required'}), 400
            
            joint_extractor = JointExtractor(ner_model_path, relation_model_path)
            return jsonify({'message': 'Joint model loaded successfully'})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_relations', methods=['POST'])
    def extract_relations():
        global relation_extractor
        
        if not relation_extractor:
            return jsonify({'error': 'Model not loaded. Please load model first.'}), 400
        
        try:
            data = request.get_json()
            text = data.get('text')
            entities = data.get('entities', [])
            
            if not text:
                return jsonify({'error': 'text is required'}), 400
            
            relations = relation_extractor.extract_relations(text, entities)
            return jsonify({
                'text': text,
                'entities': entities,
                'relations': relations
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_spo', methods=['POST'])
    def extract_spo():
        global joint_extractor
        
        if not joint_extractor:
            return jsonify({'error': 'Joint model not loaded. Please load joint model first.'}), 400
        
        try:
            data = request.get_json()
            text = data.get('text')
            
            if not text:
                return jsonify({'error': 'text is required'}), 400
            
            result = joint_extractor.extract_spo(text)
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_spo_batch', methods=['POST'])
    def extract_spo_batch():
        global joint_extractor
        
        if not joint_extractor:
            return jsonify({'error': 'Joint model not loaded. Please load joint model first.'}), 400
        
        try:
            data = request.get_json()
            texts = data.get('texts')
            
            if not texts or not isinstance(texts, list):
                return jsonify({'error': 'texts must be a list'}), 400
            
            results = joint_extractor.extract_spo_batch(texts)
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
    sample_entities = [
        {'name': '发动机盖', 'type': '部件单元', 'start_pos': 14, 'end_pos': 18},
        {'name': '抖动', 'type': '故障状态', 'start_pos': 24, 'end_pos': 26},
        {'name': '发动机盖锁', 'type': '部件单元', 'start_pos': 46, 'end_pos': 51},
        {'name': '松旷', 'type': '故障状态', 'start_pos': 58, 'end_pos': 60}
    ]
    
    # 注意：这里需要先训练模型才能测试
    # predictor = RelationPredictor('./relation_models/best_relation_model.pth')
    # relation = predictor.predict_relation(sample_text, '发动机盖', '抖动')
    # print("预测的关系:")
    # print(json.dumps(relation, ensure_ascii=False, indent=2))
    
    # 创建API服务
    app = create_relation_api()
    app.run(host='0.0.0.0', port=5002, debug=True)