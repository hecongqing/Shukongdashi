"""
故障关系抽取模型部署和推理脚本
"""

import argparse
import json
from pathlib import Path
from loguru import logger
from flask import Flask, request, jsonify
from typing import List, Dict, Any

from fault_relation_model import FaultRelationTrainer


class RelationExtractionService:
    """关系抽取服务类"""
    
    def __init__(self, model_dir: str):
        self.trainer = FaultRelationTrainer({})
        self.trainer.load_model(model_dir)
        logger.info(f"关系抽取模型加载完成: {model_dir}")
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """抽取关系"""
        return self.trainer.predict(text, entities)
    
    def batch_extract(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量抽取关系"""
        results = []
        for item in data:
            text = item['text']
            entities = item['entities']
            relations = self.extract_relations(text, entities)
            results.append({
                'ID': item.get('ID', ''),
                'text': text,
                'entities': entities,
                'relations': relations
            })
        return results


def create_app(model_dir: str) -> Flask:
    """创建Flask应用"""
    app = Flask(__name__)
    relation_service = RelationExtractionService(model_dir)
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({"status": "healthy"})
    
    @app.route('/extract_relations', methods=['POST'])
    def extract_relations():
        """关系抽取接口"""
        try:
            data = request.get_json()
            text = data.get('text', '')
            entities = data.get('entities', [])
            
            if not text:
                return jsonify({"error": "文本不能为空"}), 400
            
            if not entities:
                return jsonify({"error": "实体列表不能为空"}), 400
            
            relations = relation_service.extract_relations(text, entities)
            
            return jsonify({
                "text": text,
                "entities": entities,
                "relations": relations,
                "relation_count": len(relations)
            })
        
        except Exception as e:
            logger.error(f"抽取关系时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/batch_extract_relations', methods=['POST'])
    def batch_extract_relations():
        """批量关系抽取接口"""
        try:
            data = request.get_json()
            items = data.get('items', [])
            
            if not items:
                return jsonify({"error": "数据列表不能为空"}), 400
            
            results = relation_service.batch_extract(items)
            
            return jsonify({
                "results": results,
                "total_count": len(results)
            })
        
        except Exception as e:
            logger.error(f"批量抽取关系时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return app


def main():
    parser = argparse.ArgumentParser(description="部署故障关系抽取模型")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--mode", type=str, choices=['api', 'cli'], default='api', help="运行模式")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="API服务器主机")
    parser.add_argument("--port", type=int, default=5001, help="API服务器端口")
    parser.add_argument("--text", type=str, help="CLI模式下要处理的文本")
    parser.add_argument("--entities", type=str, help="CLI模式下的实体列表（JSON格式）")
    parser.add_argument("--input_file", type=str, help="输入文件路径（JSON Lines格式）")
    parser.add_argument("--entity_file", type=str, help="实体文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        # API模式
        logger.info(f"启动关系抽取API服务，模型目录: {args.model_dir}")
        app = create_app(args.model_dir)
        app.run(host=args.host, port=args.port, debug=False)
    
    elif args.mode == 'cli':
        # CLI模式
        relation_service = RelationExtractionService(args.model_dir)
        
        if args.text and args.entities:
            # 处理单个文本
            try:
                entities = json.loads(args.entities)
                relations = relation_service.extract_relations(args.text, entities)
                result = {
                    "text": args.text,
                    "entities": entities,
                    "relations": relations,
                    "relation_count": len(relations)
                }
                print(json.dumps(result, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                parser.error("实体列表必须是有效的JSON格式")
        
        elif args.input_file:
            # 处理文件
            logger.info(f"处理文件: {args.input_file}")
            
            # 如果有entity_file，先加载实体数据
            entities_data = {}
            if args.entity_file:
                logger.info(f"加载实体文件: {args.entity_file}")
                with open(args.entity_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entities_data[data['ID']] = data.get('entities', [])
                        except json.JSONDecodeError:
                            continue
            
            # 处理输入文件
            results = []
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        text_id = data.get('ID', f'line_{line_num}')
                        text = data.get('text', '')
                        
                        if not text:
                            continue
                        
                        # 获取实体
                        if text_id in entities_data:
                            entities = entities_data[text_id]
                        elif 'entities' in data:
                            entities = data['entities']
                        else:
                            logger.warning(f"没有找到 {text_id} 的实体信息，跳过关系抽取")
                            continue
                        
                        # 抽取关系
                        relations = relation_service.extract_relations(text, entities)
                        
                        result = {
                            "ID": text_id,
                            "text": text,
                            "entities": entities,
                            "relations": relations,
                            "relation_count": len(relations)
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
            parser.error("CLI模式下必须提供 --text 和 --entities 参数，或者 --input_file 参数")


if __name__ == "__main__":
    main()