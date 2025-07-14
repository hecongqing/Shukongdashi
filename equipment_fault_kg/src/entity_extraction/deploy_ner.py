"""
故障NER模型部署和推理脚本
"""

import argparse
import json
from pathlib import Path
from loguru import logger
from flask import Flask, request, jsonify
from typing import List, Dict, Any

from fault_ner_model import FaultNERTrainer


class NERService:
    """NER服务类"""
    
    def __init__(self, model_dir: str):
        self.trainer = FaultNERTrainer({})
        self.trainer.load_model(model_dir)
        logger.info(f"模型加载完成: {model_dir}")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体"""
        return self.trainer.predict(text)
    
    def batch_extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """批量提取实体"""
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results


def create_app(model_dir: str) -> Flask:
    """创建Flask应用"""
    app = Flask(__name__)
    ner_service = NERService(model_dir)
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({"status": "healthy"})
    
    @app.route('/extract', methods=['POST'])
    def extract():
        """实体提取接口"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({"error": "文本不能为空"}), 400
            
            entities = ner_service.extract_entities(text)
            
            return jsonify({
                "text": text,
                "entities": entities,
                "count": len(entities)
            })
        
        except Exception as e:
            logger.error(f"提取实体时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch_extract', methods=['POST'])
    def batch_extract():
        """批量实体提取接口"""
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({"error": "文本列表不能为空"}), 400
            
            results = ner_service.batch_extract(texts)
            
            return jsonify({
                "results": [
                    {
                        "text": text,
                        "entities": entities,
                        "count": len(entities)
                    }
                    for text, entities in zip(texts, results)
                ]
            })
        
        except Exception as e:
            logger.error(f"批量提取实体时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app


def main():
    parser = argparse.ArgumentParser(description="部署故障NER模型")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--mode", type=str, choices=['api', 'cli'], default='api', help="运行模式")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="API服务器主机")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口")
    parser.add_argument("--text", type=str, help="CLI模式下要处理的文本")
    parser.add_argument("--input_file", type=str, help="输入文件路径（JSON Lines格式）")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        # API模式
        logger.info(f"启动NER API服务，模型目录: {args.model_dir}")
        app = create_app(args.model_dir)
        app.run(host=args.host, port=args.port, debug=False)
    
    elif args.mode == 'cli':
        # CLI模式
        ner_service = NERService(args.model_dir)
        
        if args.text:
            # 处理单个文本
            entities = ner_service.extract_entities(args.text)
            result = {
                "text": args.text,
                "entities": entities,
                "count": len(entities)
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.input_file:
            # 处理文件
            logger.info(f"处理文件: {args.input_file}")
            
            results = []
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if text:
                            entities = ner_service.extract_entities(text)
                            result = {
                                "ID": data.get('ID', f'line_{line_num}'),
                                "text": text,
                                "entities": entities,
                                "count": len(entities)
                            }
                            results.append(result)
                            
                            if line_num % 100 == 0:
                                logger.info(f"已处理 {line_num} 行")
                    
                    except Exception as e:
                        logger.error(f"处理第 {line_num} 行时出错: {str(e)}")
                        continue
            
            # 保存结果
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                logger.info(f"结果已保存到: {args.output_file}")
            else:
                for result in results:
                    print(json.dumps(result, ensure_ascii=False))
        
        else:
            parser.error("CLI模式下必须提供 --text 或 --input_file 参数")


if __name__ == "__main__":
    main()