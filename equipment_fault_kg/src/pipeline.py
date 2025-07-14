"""
故障知识图谱信息抽取流水线

集成实体抽取和关系抽取，提供端到端的信息抽取服务
"""

import argparse
import json
from pathlib import Path
from loguru import logger
from flask import Flask, request, jsonify
from typing import List, Dict, Any
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'entity_extraction'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'relation_extraction'))

from entity_extraction.fault_ner_model import FaultNERTrainer
from relation_extraction.fault_relation_model import FaultRelationTrainer


class FaultKGPipeline:
    """故障知识图谱信息抽取流水线"""
    
    def __init__(self, ner_model_dir: str, relation_model_dir: str):
        """
        初始化流水线
        
        Args:
            ner_model_dir: 实体识别模型目录
            relation_model_dir: 关系抽取模型目录
        """
        # 加载实体识别模型
        self.ner_trainer = FaultNERTrainer({})
        self.ner_trainer.load_model(ner_model_dir)
        logger.info(f"实体识别模型加载完成: {ner_model_dir}")
        
        # 加载关系抽取模型
        self.relation_trainer = FaultRelationTrainer({})
        self.relation_trainer.load_model(relation_model_dir)
        logger.info(f"关系抽取模型加载完成: {relation_model_dir}")
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        对文本进行完整的信息抽取
        
        Args:
            text: 输入文本
            
        Returns:
            包含实体和关系的抽取结果
        """
        # 1. 实体抽取
        entities = self.ner_trainer.predict(text)
        logger.debug(f"提取到 {len(entities)} 个实体")
        
        # 2. 关系抽取
        relations = []
        if len(entities) > 1:  # 至少需要2个实体才能有关系
            relations = self.relation_trainer.predict(text, entities)
            logger.debug(f"提取到 {len(relations)} 个关系")
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations,
            'entity_count': len(entities),
            'relation_count': len(relations)
        }
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量抽取
        
        Args:
            texts: 文本列表
            
        Returns:
            抽取结果列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.extract(text)
                result['id'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"处理第 {i} 个文本时出错: {str(e)}")
                results.append({
                    'id': i,
                    'text': text,
                    'entities': [],
                    'relations': [],
                    'entity_count': 0,
                    'relation_count': 0,
                    'error': str(e)
                })
        return results
    
    def extract_file(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        处理文件中的数据
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）
            
        Returns:
            抽取结果列表
        """
        logger.info(f"开始处理文件: {input_file}")
        
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text_id = data.get('ID', f'line_{line_num}')
                    text = data.get('text', '')
                    
                    if not text:
                        continue
                    
                    # 进行信息抽取
                    result = self.extract(text)
                    result['ID'] = text_id
                    results.append(result)
                    
                    if line_num % 100 == 0:
                        logger.info(f"已处理 {line_num} 行")
                
                except Exception as e:
                    logger.error(f"处理第 {line_num} 行时出错: {str(e)}")
                    continue
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存到: {output_file}")
        
        return results
    
    def convert_to_spo_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将抽取结果转换为竞赛要求的SPO格式
        
        Args:
            result: 抽取结果
            
        Returns:
            SPO格式的结果
        """
        spo_list = []
        
        for relation in result['relations']:
            spo = {
                'h': relation['h'],
                't': relation['t'],
                'relation': relation['relation']
            }
            spo_list.append(spo)
        
        return {
            'ID': result.get('ID', ''),
            'text': result['text'],
            'spo_list': spo_list
        }
    
    def generate_submission(self, input_file: str, output_file: str):
        """
        生成竞赛提交文件
        
        Args:
            input_file: 测试文件路径
            output_file: 提交文件路径
        """
        logger.info(f"生成提交文件: {input_file} -> {output_file}")
        
        results = self.extract_file(input_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                spo_result = self.convert_to_spo_format(result)
                f.write(json.dumps(spo_result, ensure_ascii=False) + '\n')
        
        logger.info(f"提交文件已生成: {output_file}")


def create_app(ner_model_dir: str, relation_model_dir: str) -> Flask:
    """创建Flask应用"""
    app = Flask(__name__)
    pipeline = FaultKGPipeline(ner_model_dir, relation_model_dir)
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({"status": "healthy"})
    
    @app.route('/extract', methods=['POST'])
    def extract():
        """信息抽取接口"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({"error": "文本不能为空"}), 400
            
            result = pipeline.extract(text)
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"抽取信息时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch_extract', methods=['POST'])
    def batch_extract():
        """批量信息抽取接口"""
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({"error": "文本列表不能为空"}), 400
            
            results = pipeline.batch_extract(texts)
            
            return jsonify({
                "results": results,
                "total_count": len(results)
            })
        
        except Exception as e:
            logger.error(f"批量抽取信息时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/extract_spo', methods=['POST'])
    def extract_spo():
        """SPO格式信息抽取接口"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({"error": "文本不能为空"}), 400
            
            result = pipeline.extract(text)
            spo_result = pipeline.convert_to_spo_format(result)
            
            return jsonify(spo_result)
        
        except Exception as e:
            logger.error(f"抽取SPO信息时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app


def main():
    parser = argparse.ArgumentParser(description="故障知识图谱信息抽取流水线")
    parser.add_argument("--ner_model", type=str, required=True, help="实体识别模型目录")
    parser.add_argument("--relation_model", type=str, required=True, help="关系抽取模型目录")
    parser.add_argument("--mode", type=str, choices=['api', 'cli', 'submission'], default='api', help="运行模式")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="API服务器主机")
    parser.add_argument("--port", type=int, default=5002, help="API服务器端口")
    parser.add_argument("--text", type=str, help="CLI模式下要处理的文本")
    parser.add_argument("--input_file", type=str, help="输入文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        # API模式
        logger.info(f"启动知识图谱抽取API服务")
        logger.info(f"实体识别模型: {args.ner_model}")
        logger.info(f"关系抽取模型: {args.relation_model}")
        app = create_app(args.ner_model, args.relation_model)
        app.run(host=args.host, port=args.port, debug=False)
    
    elif args.mode == 'cli':
        # CLI模式
        pipeline = FaultKGPipeline(args.ner_model, args.relation_model)
        
        if args.text:
            # 处理单个文本
            result = pipeline.extract(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.input_file:
            # 处理文件
            results = pipeline.extract_file(args.input_file, args.output_file)
            if not args.output_file:
                for result in results:
                    print(json.dumps(result, ensure_ascii=False))
        
        else:
            parser.error("CLI模式下必须提供 --text 或 --input_file 参数")
    
    elif args.mode == 'submission':
        # 提交模式 - 生成竞赛提交文件
        if not args.input_file or not args.output_file:
            parser.error("提交模式下必须提供 --input_file 和 --output_file 参数")
        
        pipeline = FaultKGPipeline(args.ner_model, args.relation_model)
        pipeline.generate_submission(args.input_file, args.output_file)


if __name__ == "__main__":
    main()