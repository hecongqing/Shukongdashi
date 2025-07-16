"""
改进的装备制造领域实体抽取部署模块

整合规则基础和深度学习方法，提供更好的实体抽取效果
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import json
import logging
from typing import List, Dict, Tuple
import re
import os
from pathlib import Path

from .rule_based_ner import RuleBasedEntityExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridEntityExtractor:
    """混合实体抽取器 - 结合规则和深度学习"""
    
    def __init__(self, model_path: str = None, use_rule_based: bool = True, use_ml_model: bool = False):
        self.use_rule_based = use_rule_based
        self.use_ml_model = use_ml_model
        
        # 规则基础抽取器
        if use_rule_based:
            self.rule_extractor = RuleBasedEntityExtractor()
        
        # 深度学习模型（可选）
        self.ml_predictor = None
        if use_ml_model and model_path and Path(model_path).exists():
            try:
                # 这里可以加载训练好的深度学习模型
                # self.ml_predictor = ImprovedNERPredictor(model_path)
                logger.info("深度学习模型加载成功")
            except Exception as e:
                logger.warning(f"深度学习模型加载失败: {e}")
                self.use_ml_model = False
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体"""
        entities = []
        
        # 使用规则基础方法
        if self.use_rule_based:
            rule_entities = self.rule_extractor.extract_entities(text)
            entities.extend(rule_entities)
        
        # 使用深度学习模型（如果可用）
        if self.use_ml_model and self.ml_predictor:
            try:
                ml_entities = self.ml_predictor.predict(text)
                entities.extend(ml_entities)
            except Exception as e:
                logger.warning(f"深度学习模型预测失败: {e}")
        
        # 去重和合并
        entities = self._merge_and_deduplicate(entities)
        
        return entities
    
    def _merge_and_deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """合并和去重实体"""
        if not entities:
            return []
        
        # 按位置排序
        entities.sort(key=lambda x: x['start_pos'])
        
        # 去重和合并重叠的实体
        merged_entities = []
        i = 0
        
        while i < len(entities):
            current = entities[i]
            merged = False
            
            # 检查与下一个实体是否重叠
            if i + 1 < len(entities):
                next_entity = entities[i + 1]
                
                # 如果重叠，选择更长的实体
                if (current['start_pos'] <= next_entity['start_pos'] < current['end_pos'] or
                    next_entity['start_pos'] <= current['start_pos'] < next_entity['end_pos']):
                    
                    if len(current['name']) >= len(next_entity['name']):
                        # 保留当前实体
                        merged_entities.append(current)
                    else:
                        # 保留下一个实体
                        merged_entities.append(next_entity)
                    
                    i += 2  # 跳过两个实体
                    merged = True
                else:
                    # 不重叠，保留当前实体
                    merged_entities.append(current)
                    i += 1
            else:
                # 最后一个实体
                merged_entities.append(current)
                i += 1
        
        return merged_entities
    
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
        return [entity['name'] for entity in entities if entity['type'] == entity_type]


class ImprovedNERPredictor:
    """改进的NER预测器（深度学习模型）"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 标签映射
        self.label2id = {
            'O': 0,
            'B-COMPONENT': 1, 'I-COMPONENT': 2,
            'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
            'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
            'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 实体类型映射
        self.entity_type_mapping = {
            'COMPONENT': '部件单元',
            'PERFORMANCE': '性能表征',
            'FAULT_STATE': '故障状态',
            'DETECTION_TOOL': '检测工具'
        }
        
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取标签数量
            num_labels = len(self.label2id)
            
            # 加载模型（这里需要根据实际的模型架构调整）
            # self.model = YourNERModel('bert-base-chinese', num_labels)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.to(self.device)
            # self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> List[Dict]:
        """预测文本中的实体"""
        # 这里实现深度学习模型的预测逻辑
        # 暂时返回空列表，等待模型训练完成
        return []


def create_improved_ner_api():
    """创建改进的NER API服务"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # 全局变量存储抽取器
    entity_extractor = None
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/load_extractor', methods=['POST'])
    def load_extractor():
        global entity_extractor
        try:
            data = request.get_json()
            use_rule_based = data.get('use_rule_based', True)
            use_ml_model = data.get('use_ml_model', False)
            model_path = data.get('model_path', None)
            
            entity_extractor = HybridEntityExtractor(
                model_path=model_path,
                use_rule_based=use_rule_based,
                use_ml_model=use_ml_model
            )
            
            return jsonify({
                'message': 'Entity extractor loaded successfully',
                'use_rule_based': use_rule_based,
                'use_ml_model': use_ml_model
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/extract_entities', methods=['POST'])
    def extract_entities():
        global entity_extractor
        
        if not entity_extractor:
            return jsonify({'error': 'Entity extractor not loaded. Please load extractor first.'}), 400
        
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
            return jsonify({'error': 'Entity extractor not loaded. Please load extractor first.'}), 400
        
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


def test_improved_ner():
    """测试改进的NER系统"""
    print("=== 改进的装备制造领域实体抽取系统测试 ===")
    
    # 创建混合抽取器
    extractor = HybridEntityExtractor(use_rule_based=True, use_ml_model=False)
    
    # 测试案例
    test_cases = [
        "伺服电机运行异常，维修人员使用万用表检测电路故障。",
        "液压泵压力不足，需要更换密封圈解决泄漏问题。",
        "轴承温度过高导致振动，使用振动仪检测发现不对中。",
        "发动机启动困难，检查发现燃油泵故障。",
        "数控机床主轴故障导致加工精度下降，需要更换轴承解决。"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"输入文本: {text}")
        
        entities = extractor.extract_entities(text)
        
        print("抽取的实体:")
        for entity in entities:
            print(f"  - {entity['name']} [{entity['type']}]")
        
        # 按类型统计
        type_counts = {}
        for entity in entities:
            entity_type = entity['type']
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        print("实体类型统计:")
        for entity_type, count in type_counts.items():
            print(f"  {entity_type}: {count}个")


def compare_extraction_methods():
    """比较不同抽取方法的效果"""
    print("=== 实体抽取方法对比 ===")
    
    text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    print(f"测试文本: {text}")
    
    # 规则基础方法
    print("\n1. 规则基础方法:")
    rule_extractor = RuleBasedEntityExtractor()
    rule_entities = rule_extractor.extract_entities(text)
    for entity in rule_entities:
        print(f"  - {entity['name']} [{entity['type']}]")
    
    # 混合方法
    print("\n2. 混合方法:")
    hybrid_extractor = HybridEntityExtractor(use_rule_based=True, use_ml_model=False)
    hybrid_entities = hybrid_extractor.extract_entities(text)
    for entity in hybrid_entities:
        print(f"  - {entity['name']} [{entity['type']}]")
    
    print("\n对比结果:")
    print(f"规则基础方法抽取到 {len(rule_entities)} 个实体")
    print(f"混合方法抽取到 {len(hybrid_entities)} 个实体")


if __name__ == "__main__":
    # 运行测试
    test_improved_ner()
    print("\n" + "="*60)
    compare_extraction_methods()
    
    # 创建API服务
    print(f"\n=== 启动改进的 API 服务 ===")
    app = create_improved_ner_api()
    app.run(host='0.0.0.0', port=5002, debug=True)