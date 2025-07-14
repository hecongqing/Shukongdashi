#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取系统测试脚本
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.entity_extraction.data_processor import DataProcessor
from src.deployment.pipeline import InformationExtractionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_processor():
    """测试数据处理器"""
    logger.info("Testing data processor...")
    
    # 创建示例数据
    sample_data = [
        {
            "ID": "AT0001",
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "spo_list": [
                {
                    "h": {"name": "发动机盖", "pos": [14, 18]},
                    "t": {"name": "抖动", "pos": [24, 26]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "发动机盖锁", "pos": [46, 51]},
                    "t": {"name": "松旷", "pos": [58, 60]},
                    "relation": "部件故障"
                }
            ]
        }
    ]
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 测试实体提取
    entities = processor.extract_entities_from_spo(sample_data[0]['spo_list'])
    logger.info(f"Extracted {len(entities)} entities")
    for entity in entities:
        logger.info(f"  - {entity.name} ({entity.type}) at [{entity.start}, {entity.end}]")
    
    # 测试NER格式转换
    ner_data = processor.convert_to_ner_format(sample_data)
    logger.info(f"Converted to {len(ner_data)} NER samples")
    
    # 测试RE格式转换
    re_data = processor.convert_to_re_format(sample_data)
    logger.info(f"Converted to {len(re_data)} RE samples")
    
    return True

def test_pipeline():
    """测试信息抽取管道"""
    logger.info("Testing information extraction pipeline...")
    
    # 检查模型文件
    ner_model_path = "models/ner_model"
    re_model_path = "models/re_model"
    
    ner_exists = os.path.exists(os.path.join(ner_model_path, "ner_model.pth"))
    re_exists = os.path.exists(os.path.join(re_model_path, "re_model.pth"))
    
    if not ner_exists or not re_exists:
        logger.warning("Model files not found. Skipping pipeline test.")
        return False
    
    try:
        # 初始化管道
        pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
        
        # 测试文本
        test_texts = [
            "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸。",
            "减振器活塞与缸体发卡，工作阻力过大诊断排除。"
        ]
        
        for i, text in enumerate(test_texts):
            logger.info(f"\nTesting text {i+1}: {text}")
            
            # 执行抽取
            result = pipeline.extract(text)
            
            logger.info(f"Extracted {len(result['entities'])} entities and {len(result['relations'])} relations")
            
            # 打印实体
            if result['entities']:
                logger.info("Entities:")
                for entity in result['entities']:
                    logger.info(f"  - {entity['text']} ({entity['type']}) at [{entity['start']}, {entity['end']}]")
            
            # 打印关系
            if result['relations']:
                logger.info("Relations:")
                for relation in result['relations']:
                    logger.info(f"  - {relation['head']['text']} --{relation['relation']}--> {relation['tail']['text']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False

def test_api():
    """测试API服务"""
    logger.info("Testing API service...")
    
    try:
        import requests
        
        # 测试健康检查
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            logger.info("API health check passed")
        else:
            logger.warning("API health check failed")
            return False
        
        # 测试信息抽取
        test_data = {
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。"
        }
        
        response = requests.post("http://localhost:8000/extract", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API extraction successful: {len(result['entities'])} entities, {len(result['relations'])} relations")
            return True
        else:
            logger.error(f"API extraction failed: {response.status_code}")
            return False
            
    except ImportError:
        logger.warning("requests not installed. Skipping API test.")
        return False
    except requests.exceptions.ConnectionError:
        logger.warning("API server not running. Skipping API test.")
        return False

def main():
    """主测试函数"""
    logger.info("Starting system tests...")
    
    # 创建必要的目录
    directories = ["data", "models/ner_model", "models/re_model", "results"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 运行测试
    tests = [
        ("Data Processor", test_data_processor),
        ("Pipeline", test_pipeline),
        ("API", test_api)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} test passed")
            else:
                logger.warning(f"⚠ {test_name} test failed")
        except Exception as e:
            logger.error(f"✗ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # 打印测试总结
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning("⚠ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()