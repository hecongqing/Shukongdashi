#!/usr/bin/env python3
"""
设备故障知识图谱 - 模型部署主脚本
用于启动实体抽取和关系抽取的API服务
"""

import os
import sys
import argparse
import logging
import json
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.entity_extraction.deploy_ner import EntityExtractor, create_ner_api
from src.relation_extraction.deploy_relation import RelationExtractor, JointExtractor, create_relation_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedAPI:
    """统一的API服务类"""
    
    def __init__(self, ner_model_path: str = None, relation_model_path: str = None):
        self.ner_model_path = ner_model_path
        self.relation_model_path = relation_model_path
        
        self.entity_extractor = None
        self.relation_extractor = None
        self.joint_extractor = None
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'entity_model_loaded': self.entity_extractor is not None,
                'relation_model_loaded': self.relation_extractor is not None,
                'joint_model_loaded': self.joint_extractor is not None
            })
        
        @self.app.route('/load_models', methods=['POST'])
        def load_models():
            try:
                data = request.get_json()
                ner_path = data.get('ner_model_path', self.ner_model_path)
                relation_path = data.get('relation_model_path', self.relation_model_path)
                
                if not ner_path or not relation_path:
                    return jsonify({'error': 'ner_model_path and relation_model_path are required'}), 400
                
                # 加载模型
                self.entity_extractor = EntityExtractor(ner_path)
                self.relation_extractor = RelationExtractor(relation_path)
                self.joint_extractor = JointExtractor(ner_path, relation_path)
                
                return jsonify({'message': 'All models loaded successfully'})
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/extract_entities', methods=['POST'])
        def extract_entities():
            if not self.entity_extractor:
                return jsonify({'error': 'Entity model not loaded'}), 400
            
            try:
                data = request.get_json()
                text = data.get('text')
                
                if not text:
                    return jsonify({'error': 'text is required'}), 400
                
                entities = self.entity_extractor.extract_entities(text)
                return jsonify({
                    'text': text,
                    'entities': entities
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/extract_relations', methods=['POST'])
        def extract_relations():
            if not self.relation_extractor:
                return jsonify({'error': 'Relation model not loaded'}), 400
            
            try:
                data = request.get_json()
                text = data.get('text')
                entities = data.get('entities', [])
                
                if not text:
                    return jsonify({'error': 'text is required'}), 400
                
                relations = self.relation_extractor.extract_relations(text, entities)
                return jsonify({
                    'text': text,
                    'entities': entities,
                    'relations': relations
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/extract_spo', methods=['POST'])
        def extract_spo():
            if not self.joint_extractor:
                return jsonify({'error': 'Joint model not loaded'}), 400
            
            try:
                data = request.get_json()
                text = data.get('text')
                
                if not text:
                    return jsonify({'error': 'text is required'}), 400
                
                result = self.joint_extractor.extract_spo(text)
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/extract_spo_batch', methods=['POST'])
        def extract_spo_batch():
            if not self.joint_extractor:
                return jsonify({'error': 'Joint model not loaded'}), 400
            
            try:
                data = request.get_json()
                texts = data.get('texts')
                
                if not texts or not isinstance(texts, list):
                    return jsonify({'error': 'texts must be a list'}), 400
                
                results = self.joint_extractor.extract_spo_batch(texts)
                return jsonify({
                    'texts': texts,
                    'results': results
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/demo', methods=['GET'])
        def demo():
            """演示页面"""
            demo_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>设备故障知识图谱 - 信息抽取演示</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    textarea { width: 100%; height: 100px; margin: 10px 0; }
                    button { padding: 10px 20px; margin: 10px 5px; }
                    .result { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                    .entity { color: blue; }
                    .relation { color: green; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>设备故障知识图谱 - 信息抽取演示</h1>
                    
                    <h3>输入故障文本:</h3>
                    <textarea id="inputText">故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。</textarea>
                    
                    <div>
                        <button onclick="extractEntities()">抽取实体</button>
                        <button onclick="extractRelations()">抽取关系</button>
                        <button onclick="extractSPO()">抽取SPO三元组</button>
                    </div>
                    
                    <div id="result" class="result"></div>
                </div>
                
                <script>
                    async function extractEntities() {
                        const text = document.getElementById('inputText').value;
                        const response = await fetch('/extract_entities', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: text})
                        });
                        const result = await response.json();
                        displayResult('实体抽取结果', result);
                    }
                    
                    async function extractRelations() {
                        const text = document.getElementById('inputText').value;
                        const response = await fetch('/extract_entities', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: text})
                        });
                        const entityResult = await response.json();
                        
                        const relationResponse = await fetch('/extract_relations', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: text, entities: entityResult.entities})
                        });
                        const relationResult = await relationResponse.json();
                        displayResult('关系抽取结果', relationResult);
                    }
                    
                    async function extractSPO() {
                        const text = document.getElementById('inputText').value;
                        const response = await fetch('/extract_spo', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({text: text})
                        });
                        const result = await response.json();
                        displayResult('SPO三元组抽取结果', result);
                    }
                    
                    function displayResult(title, data) {
                        const resultDiv = document.getElementById('result');
                        resultDiv.innerHTML = '<h3>' + title + '</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    }
                </script>
            </body>
            </html>
            """
            return demo_html

def start_ner_api(ner_model_path: str, port: int = 5001):
    """启动NER API服务"""
    app = create_ner_api()
    
    # 加载模型
    if ner_model_path and os.path.exists(ner_model_path):
        with app.test_client() as client:
            response = client.post('/load_model', json={'model_path': ner_model_path})
            if response.status_code == 200:
                logger.info("NER模型加载成功")
            else:
                logger.warning("NER模型加载失败")
    
    app.run(host='0.0.0.0', port=port, debug=False)

def start_relation_api(relation_model_path: str, port: int = 5002):
    """启动关系抽取API服务"""
    app = create_relation_api()
    
    # 加载模型
    if relation_model_path and os.path.exists(relation_model_path):
        with app.test_client() as client:
            response = client.post('/load_model', json={'model_path': relation_model_path})
            if response.status_code == 200:
                logger.info("关系抽取模型加载成功")
            else:
                logger.warning("关系抽取模型加载失败")
    
    app.run(host='0.0.0.0', port=port, debug=False)

def start_unified_api(ner_model_path: str, relation_model_path: str, port: int = 5000):
    """启动统一API服务"""
    api = UnifiedAPI(ner_model_path, relation_model_path)
    
    # 自动加载模型
    if ner_model_path and relation_model_path:
        try:
            api.entity_extractor = EntityExtractor(ner_model_path)
            api.relation_extractor = RelationExtractor(relation_model_path)
            api.joint_extractor = JointExtractor(ner_model_path, relation_model_path)
            logger.info("所有模型加载成功")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")
    
    api.app.run(host='0.0.0.0', port=port, debug=False)

def main():
    parser = argparse.ArgumentParser(description="部署实体抽取和关系抽取模型")
    parser.add_argument("--ner_model_path", type=str, help="NER模型路径")
    parser.add_argument("--relation_model_path", type=str, help="关系抽取模型路径")
    parser.add_argument("--port", type=int, default=5000, help="API服务端口")
    parser.add_argument("--mode", choices=['unified', 'separate'], default='unified', 
                       help="部署模式: unified(统一服务) 或 separate(分离服务)")
    
    args = parser.parse_args()
    
    if args.mode == 'unified':
        # 统一API服务
        logger.info("启动统一API服务...")
        start_unified_api(args.ner_model_path, args.relation_model_path, args.port)
    else:
        # 分离API服务
        if args.ner_model_path:
            logger.info("启动NER API服务...")
            ner_thread = threading.Thread(
                target=start_ner_api, 
                args=(args.ner_model_path, 5001)
            )
            ner_thread.daemon = True
            ner_thread.start()
        
        if args.relation_model_path:
            logger.info("启动关系抽取API服务...")
            relation_thread = threading.Thread(
                target=start_relation_api, 
                args=(args.relation_model_path, 5002)
            )
            relation_thread.daemon = True
            relation_thread.start()
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("服务停止")

if __name__ == "__main__":
    main()