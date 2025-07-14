#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取服务部署脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.deployment.pipeline import run_server, InformationExtractionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_models():
    """检查模型文件是否存在"""
    ner_model_path = "models/ner_model"
    re_model_path = "models/re_model"
    
    ner_exists = os.path.exists(os.path.join(ner_model_path, "ner_model.pth"))
    re_exists = os.path.exists(os.path.join(re_model_path, "re_model.pth"))
    
    if not ner_exists:
        logger.warning(f"NER model not found at {ner_model_path}")
    
    if not re_exists:
        logger.warning(f"RE model not found at {re_model_path}")
    
    return ner_exists, re_exists

def test_pipeline():
    """测试信息抽取管道"""
    logger.info("Testing information extraction pipeline...")
    
    # 初始化管道
    ner_model_path = "models/ner_model"
    re_model_path = "models/re_model"
    
    try:
        pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
        
        # 测试文本
        test_text = "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。"
        
        # 执行抽取
        result = pipeline.extract(test_text)
        
        logger.info("Pipeline test successful!")
        logger.info(f"Extracted {len(result['entities'])} entities and {len(result['relations'])} relations")
        
        # 打印结果
        print("\n=== Test Results ===")
        print(f"Text: {test_text}")
        print(f"\nEntities ({len(result['entities'])}):")
        for entity in result['entities']:
            print(f"  - {entity['text']} ({entity['type']}) at [{entity['start']}, {entity['end']}]")
        
        print(f"\nRelations ({len(result['relations'])}):")
        for relation in result['relations']:
            print(f"  - {relation['head']['text']} --{relation['relation']}--> {relation['tail']['text']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Deploy information extraction service")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind the server to")
    parser.add_argument("--test", action="store_true",
                       help="Test the pipeline before starting server")
    parser.add_argument("--check-models", action="store_true",
                       help="Check if model files exist")
    
    args = parser.parse_args()
    
    # 检查模型
    if args.check_models:
        ner_exists, re_exists = check_models()
        if not ner_exists or not re_exists:
            logger.error("Some models are missing. Please train the models first.")
            sys.exit(1)
        logger.info("All models found!")
        return
    
    # 测试管道
    if args.test:
        if not test_pipeline():
            logger.error("Pipeline test failed. Please check the models.")
            sys.exit(1)
        return
    
    # 检查模型文件
    ner_exists, re_exists = check_models()
    if not ner_exists or not re_exists:
        logger.warning("Some models are missing. The service may not work properly.")
    
    # 启动服务器
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info("API documentation available at: http://localhost:8000/docs")
    
    try:
        run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()