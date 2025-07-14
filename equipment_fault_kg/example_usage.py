#!/usr/bin/env python3
"""
设备故障知识图谱信息抽取系统使用示例
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

def example_data_processing():
    """示例：数据处理"""
    logger.info("=== 数据处理示例 ===")
    
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
        },
        {
            "ID": "AT0002",
            "text": "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸，因此将影响发动机的正常运转，使得发动机出现加速不良的症状。",
            "spo_list": [
                {
                    "h": {"name": "燃油泵", "pos": [0, 3]},
                    "t": {"name": "损坏", "pos": [15, 17]},
                    "relation": "部件故障"
                },
                {
                    "h": {"name": "发动机", "pos": [25, 28]},
                    "t": {"name": "加速不良", "pos": [45, 49]},
                    "relation": "部件故障"
                }
            ]
        }
    ]
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 提取实体
    print("\n1. 实体提取:")
    for i, sample in enumerate(sample_data):
        entities = processor.extract_entities_from_spo(sample['spo_list'])
        print(f"   样本 {i+1}: 提取到 {len(entities)} 个实体")
        for entity in entities:
            print(f"     - {entity.name} ({entity.type}) 位置: [{entity.start}, {entity.end}]")
    
    # 转换为NER格式
    ner_data = processor.convert_to_ner_format(sample_data)
    print(f"\n2. NER格式转换: 生成 {len(ner_data)} 个NER样本")
    
    # 转换为RE格式
    re_data = processor.convert_to_re_format(sample_data)
    print(f"3. RE格式转换: 生成 {len(re_data)} 个RE样本")
    
    # 保存处理后的数据
    processor.save_processed_data(ner_data, "data/example_ner.json")
    processor.save_processed_data(re_data, "data/example_re.json")
    print("4. 数据已保存到 data/ 目录")

def example_model_inference():
    """示例：模型推理"""
    logger.info("\n=== 模型推理示例 ===")
    
    # 检查模型文件
    ner_model_path = "models/ner_model"
    re_model_path = "models/re_model"
    
    if not os.path.exists(os.path.join(ner_model_path, "ner_model.pth")):
        logger.warning("NER模型文件不存在，请先训练模型")
        return
    
    if not os.path.exists(os.path.join(re_model_path, "re_model.pth")):
        logger.warning("RE模型文件不存在，请先训练模型")
        return
    
    try:
        # 初始化管道
        pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
        
        # 测试文本
        test_texts = [
            "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸。",
            "减振器活塞与缸体发卡，工作阻力过大诊断排除。",
            "当液面变低时，需要检查燃油泵的工作状态。",
            "使用漏电测试仪检测电流异常情况。"
        ]
        
        print("\n1. 端到端信息抽取:")
        for i, text in enumerate(test_texts):
            print(f"\n   文本 {i+1}: {text}")
            
            # 执行抽取
            result = pipeline.extract(text)
            
            # 显示实体
            if result['entities']:
                print(f"   实体 ({len(result['entities'])}):")
                for entity in result['entities']:
                    print(f"     - {entity['text']} ({entity['type']}) 位置: [{entity['start']}, {entity['end']}]")
            else:
                print("   实体: 无")
            
            # 显示关系
            if result['relations']:
                print(f"   关系 ({len(result['relations'])}):")
                for relation in result['relations']:
                    print(f"     - {relation['head']['text']} --{relation['relation']}--> {relation['tail']['text']}")
            else:
                print("   关系: 无")
        
        print("\n2. 单独实体抽取:")
        entities = pipeline.extract_entities(test_texts[0])
        print(f"   从文本中抽取到 {len(entities)} 个实体")
        
        print("\n3. 单独关系抽取:")
        if entities:
            relations = pipeline.extract_relations(test_texts[0], entities)
            print(f"   从实体中抽取到 {len(relations)} 个关系")
        
    except Exception as e:
        logger.error(f"模型推理失败: {e}")

def example_api_usage():
    """示例：API使用"""
    logger.info("\n=== API使用示例 ===")
    
    try:
        import requests
        
        # 检查服务是否运行
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                logger.warning("API服务未运行，请先启动服务: python deploy.py")
                return
        except requests.exceptions.RequestException:
            logger.warning("API服务未运行，请先启动服务: python deploy.py")
            return
        
        print("\n1. 单文本抽取:")
        test_data = {
            "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。"
        }
        
        response = requests.post("http://localhost:8000/extract", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   成功抽取 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")
            
            # 显示结果
            if result['entities']:
                print("   实体:")
                for entity in result['entities']:
                    print(f"     - {entity['text']} ({entity['type']})")
            
            if result['relations']:
                print("   关系:")
                for relation in result['relations']:
                    print(f"     - {relation['head']['text']} --{relation['relation']}--> {relation['tail']['text']}")
        else:
            print(f"   API调用失败: {response.status_code}")
        
        print("\n2. 批量抽取:")
        texts = [
            "燃油泵损坏导致发动机无法启动。",
            "减振器活塞与缸体发卡，工作阻力过大。"
        ]
        
        response = requests.post("http://localhost:8000/extract_batch", json=texts)
        if response.status_code == 200:
            results = response.json()["results"]
            print(f"   成功处理 {len(results)} 个文本")
            
            for i, result in enumerate(results):
                print(f"   文本 {i+1}: {len(result['entities'])} 个实体, {len(result['relations'])} 个关系")
        else:
            print(f"   批量API调用失败: {response.status_code}")
        
        print("\n3. 健康检查:")
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   服务状态: {health['status']}")
            print(f"   NER模型: {'已加载' if health['ner_model_loaded'] else '未加载'}")
            print(f"   RE模型: {'已加载' if health['re_model_loaded'] else '未加载'}")
        
    except ImportError:
        logger.warning("requests库未安装，无法演示API使用")
    except Exception as e:
        logger.error(f"API使用示例失败: {e}")

def main():
    """主函数"""
    logger.info("设备故障知识图谱信息抽取系统使用示例")
    logger.info("=" * 60)
    
    # 创建必要的目录
    directories = ["data", "models/ner_model", "models/re_model", "results"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 运行示例
    example_data_processing()
    example_model_inference()
    example_api_usage()
    
    logger.info("\n" + "=" * 60)
    logger.info("示例运行完成！")
    logger.info("\n使用说明:")
    logger.info("1. 数据处理: 使用 DataProcessor 类处理训练数据")
    logger.info("2. 模型训练: 运行 python train_models.py")
    logger.info("3. 模型推理: 使用 InformationExtractionPipeline 类")
    logger.info("4. API服务: 运行 python deploy.py 启动服务")
    logger.info("5. 系统测试: 运行 python test_system.py 验证系统")

if __name__ == "__main__":
    main()