#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取模型评估脚本
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.entity_extraction.data_processor import DataProcessor
from src.entity_extraction.trainer import NERTrainer
from src.relation_extraction.trainer import RETrainer
from src.deployment.pipeline import InformationExtractionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, ner_model_path: str = None, re_model_path: str = None):
        self.ner_trainer = None
        self.re_trainer = None
        self.pipeline = None
        
        # 加载模型
        if ner_model_path and os.path.exists(ner_model_path):
            self.load_ner_model(ner_model_path)
        
        if re_model_path and os.path.exists(re_model_path):
            self.load_re_model(re_model_path)
        
        if ner_model_path and re_model_path:
            self.pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
    
    def load_ner_model(self, model_path: str):
        """加载NER模型"""
        try:
            self.ner_trainer = NERTrainer()
            self.ner_trainer.load_model(model_path)
            logger.info(f"NER model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
    
    def load_re_model(self, model_path: str):
        """加载关系抽取模型"""
        try:
            self.re_trainer = RETrainer()
            self.re_trainer.load_model(model_path)
            logger.info(f"RE model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load RE model: {e}")
    
    def evaluate_ner(self, test_data: List[Dict]) -> Dict:
        """评估NER模型"""
        if self.ner_trainer is None:
            logger.error("NER model not loaded")
            return {}
        
        logger.info("Evaluating NER model...")
        
        all_preds = []
        all_labels = []
        entity_results = []
        
        for sample in test_data:
            text = sample['text']
            true_entities = sample.get('entities', [])
            true_labels = sample.get('labels', [])
            
            # 预测实体
            pred_entities = self.ner_trainer.predict(text)
            
            # 转换为标签序列进行比较
            pred_labels = ['O'] * len(text)
            for entity in pred_entities:
                if entity['start'] < len(text) and entity['end'] <= len(text):
                    pred_labels[entity['start']] = f"B-{entity['type']}"
                    for i in range(entity['start'] + 1, entity['end']):
                        if i < len(pred_labels):
                            pred_labels[i] = f"I-{entity['type']}"
            
            # 对齐标签长度
            min_len = min(len(true_labels), len(pred_labels))
            all_preds.extend(pred_labels[:min_len])
            all_labels.extend(true_labels[:min_len])
            
            # 记录实体级别的结果
            entity_results.append({
                'text': text,
                'true_entities': true_entities,
                'pred_entities': pred_entities
            })
        
        # 计算指标
        metrics = self._calculate_ner_metrics(all_labels, all_preds)
        
        return {
            'metrics': metrics,
            'entity_results': entity_results
        }
    
    def evaluate_re(self, test_data: List[Dict]) -> Dict:
        """评估关系抽取模型"""
        if self.re_trainer is None:
            logger.error("RE model not loaded")
            return {}
        
        logger.info("Evaluating RE model...")
        
        all_preds = []
        all_labels = []
        relation_results = []
        
        for sample in test_data:
            text = sample['text']
            true_entities = sample.get('entities', [])
            true_relations = sample.get('relations', [])
            
            # 预测关系
            pred_relations = self.re_trainer.predict(text, true_entities)
            
            # 构建真实关系标签
            true_relation_labels = []
            pred_relation_labels = []
            
            # 为每对实体创建标签
            for i, entity1 in enumerate(true_entities):
                for j, entity2 in enumerate(true_entities):
                    if i != j:
                        # 检查真实关系
                        true_relation = 'no_relation'
                        for rel in true_relations:
                            if (rel['head'].name == entity1.name and 
                                rel['tail'].name == entity2.name):
                                true_relation = rel['relation_type']
                                break
                        
                        # 检查预测关系
                        pred_relation = 'no_relation'
                        for rel in pred_relations:
                            if (rel['head']['text'] == entity1.name and 
                                rel['tail']['text'] == entity2.name):
                                pred_relation = rel['relation']
                                break
                        
                        true_relation_labels.append(true_relation)
                        pred_relation_labels.append(pred_relation)
            
            all_preds.extend(pred_relation_labels)
            all_labels.extend(true_relation_labels)
            
            # 记录关系级别的结果
            relation_results.append({
                'text': text,
                'true_relations': true_relations,
                'pred_relations': pred_relations
            })
        
        # 计算指标
        metrics = self._calculate_re_metrics(all_labels, all_preds)
        
        return {
            'metrics': metrics,
            'relation_results': relation_results
        }
    
    def evaluate_pipeline(self, test_data: List[Dict]) -> Dict:
        """评估端到端管道"""
        if self.pipeline is None:
            logger.error("Pipeline not loaded")
            return {}
        
        logger.info("Evaluating end-to-end pipeline...")
        
        pipeline_results = []
        
        for sample in test_data:
            text = sample['text']
            true_entities = sample.get('entities', [])
            true_relations = sample.get('relations', [])
            
            # 端到端预测
            result = self.pipeline.extract(text)
            pred_entities = result['entities']
            pred_relations = result['relations']
            
            pipeline_results.append({
                'text': text,
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'true_relations': true_relations,
                'pred_relations': pred_relations
            })
        
        # 计算端到端指标
        metrics = self._calculate_pipeline_metrics(pipeline_results)
        
        return {
            'metrics': metrics,
            'pipeline_results': pipeline_results
        }
    
    def _calculate_ner_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict:
        """计算NER指标"""
        # 过滤掉O标签
        filtered_true = []
        filtered_pred = []
        
        for true, pred in zip(true_labels, pred_labels):
            if true != 'O' or pred != 'O':
                filtered_true.append(true)
                filtered_pred.append(pred)
        
        if not filtered_true:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        precision = precision_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        recall = recall_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        f1 = f1_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_re_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict:
        """计算关系抽取指标"""
        # 过滤掉no_relation
        filtered_true = []
        filtered_pred = []
        
        for true, pred in zip(true_labels, pred_labels):
            if true != 'no_relation' or pred != 'no_relation':
                filtered_true.append(true)
                filtered_pred.append(pred)
        
        if not filtered_true:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        precision = precision_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        recall = recall_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        f1 = f1_score(filtered_true, filtered_pred, average='weighted', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_pipeline_metrics(self, results: List[Dict]) -> Dict:
        """计算端到端管道指标"""
        total_entities = 0
        correct_entities = 0
        total_relations = 0
        correct_relations = 0
        
        for result in results:
            true_entities = result['true_entities']
            pred_entities = result['pred_entities']
            true_relations = result['true_relations']
            pred_relations = result['pred_relations']
            
            # 实体准确率
            total_entities += len(true_entities)
            for true_entity in true_entities:
                for pred_entity in pred_entities:
                    if (true_entity.name == pred_entity['text'] and 
                        true_entity.type == pred_entity['type']):
                        correct_entities += 1
                        break
            
            # 关系准确率
            total_relations += len(true_relations)
            for true_relation in true_relations:
                for pred_relation in pred_relations:
                    if (true_relation['head'].name == pred_relation['head']['text'] and
                        true_relation['tail'].name == pred_relation['tail']['text'] and
                        true_relation['relation_type'] == pred_relation['relation']):
                        correct_relations += 1
                        break
        
        entity_accuracy = correct_entities / total_entities if total_entities > 0 else 0.0
        relation_accuracy = correct_relations / total_relations if total_relations > 0 else 0.0
        
        return {
            'entity_accuracy': entity_accuracy,
            'relation_accuracy': relation_accuracy,
            'total_entities': total_entities,
            'correct_entities': correct_entities,
            'total_relations': total_relations,
            'correct_relations': correct_relations
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate information extraction models")
    parser.add_argument("--test_file", type=str, default="data/test.json",
                       help="Path to test data file")
    parser.add_argument("--ner_model", type=str, default="models/ner_model",
                       help="Path to NER model")
    parser.add_argument("--re_model", type=str, default="models/re_model",
                       help="Path to RE model")
    parser.add_argument("--output", type=str, default="results/evaluation_results.json",
                       help="Path to output results file")
    parser.add_argument("--evaluate_ner", action="store_true",
                       help="Evaluate NER model")
    parser.add_argument("--evaluate_re", action="store_true",
                       help="Evaluate RE model")
    parser.add_argument("--evaluate_pipeline", action="store_true",
                       help="Evaluate end-to-end pipeline")
    
    args = parser.parse_args()
    
    # 创建结果目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    if not os.path.exists(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        sys.exit(1)
    
    processor = DataProcessor()
    test_data = processor.load_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # 初始化评估器
    evaluator = ModelEvaluator(args.ner_model, args.re_model)
    
    results = {}
    
    # 评估NER模型
    if args.evaluate_ner or not (args.evaluate_re or args.evaluate_pipeline):
        ner_results = evaluator.evaluate_ner(test_data)
        results['ner'] = ner_results
        logger.info(f"NER Metrics: {ner_results.get('metrics', {})}")
    
    # 评估RE模型
    if args.evaluate_re or not (args.evaluate_ner or args.evaluate_pipeline):
        re_results = evaluator.evaluate_re(test_data)
        results['re'] = re_results
        logger.info(f"RE Metrics: {re_results.get('metrics', {})}")
    
    # 评估端到端管道
    if args.evaluate_pipeline or not (args.evaluate_ner or args.evaluate_re):
        pipeline_results = evaluator.evaluate_pipeline(test_data)
        results['pipeline'] = pipeline_results
        logger.info(f"Pipeline Metrics: {pipeline_results.get('metrics', {})}")
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {args.output}")
    
    # 打印总结
    print("\n=== Evaluation Summary ===")
    for model_type, result in results.items():
        metrics = result.get('metrics', {})
        print(f"\n{model_type.upper()} Model:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()