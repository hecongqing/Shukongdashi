#!/usr/bin/env python3
"""
设备故障知识图谱 - 系统测试脚本
用于测试实体抽取和关系抽取系统的功能
"""

import os
import sys
import json
import logging
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.entity_extraction.deploy_ner import EntityExtractor
from src.relation_extraction.deploy_relation import RelationExtractor, JointExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """系统测试类"""
    
    def __init__(self):
        self.test_texts = [
            "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
            "燃油泵的作用是将燃油加压输送到喷油器，当燃油泵损坏后，燃油将不能正常喷入发动机气缸，因此将影响发动机的正常运转，使得发动机出现加速不良的症状。",
            "使用漏电测试仪检测电流异常，发现保护器动作频繁，需要更换零序互感器。",
            "减振器活塞与缸体发卡，工作阻力过大诊断排除。"
        ]
        
        self.expected_entities = {
            "发动机盖": "部件单元",
            "抖动": "故障状态",
            "发动机盖锁": "部件单元",
            "松旷": "故障状态",
            "燃油泵": "部件单元",
            "损坏": "故障状态",
            "发动机": "部件单元",
            "加速不良": "故障状态",
            "漏电测试仪": "检测工具",
            "电流": "性能表征",
            "保护器": "检测工具",
            "零序互感器": "检测工具",
            "减振器活塞": "部件单元",
            "缸体": "部件单元",
            "发卡": "故障状态",
            "工作阻力": "性能表征",
            "过大": "故障状态"
        }
        
        self.expected_relations = [
            ("发动机盖", "抖动", "部件故障"),
            ("发动机盖锁", "松旷", "部件故障"),
            ("燃油泵", "损坏", "部件故障"),
            ("发动机", "加速不良", "部件故障"),
            ("漏电测试仪", "电流", "检测工具"),
            ("减振器活塞", "发卡", "部件故障"),
            ("缸体", "发卡", "部件故障"),
            ("工作阻力", "过大", "性能故障")
        ]
    
    def test_entity_extraction(self, model_path: str):
        """测试实体抽取功能"""
        logger.info("开始测试实体抽取功能...")
        
        try:
            # 加载模型
            entity_extractor = EntityExtractor(model_path)
            logger.info("实体抽取模型加载成功")
            
            # 测试单个文本
            test_text = self.test_texts[0]
            entities = entity_extractor.extract_entities(test_text)
            
            logger.info(f"测试文本: {test_text}")
            logger.info(f"抽取的实体: {json.dumps(entities, ensure_ascii=False, indent=2)}")
            
            # 验证结果
            extracted_entities = {entity['name']: entity['type'] for entity in entities}
            correct_count = 0
            total_count = 0
            
            for entity_name, expected_type in self.expected_entities.items():
                if entity_name in test_text:
                    total_count += 1
                    if entity_name in extracted_entities:
                        if extracted_entities[entity_name] == expected_type:
                            correct_count += 1
                        else:
                            logger.warning(f"实体类型不匹配: {entity_name}, 期望: {expected_type}, 实际: {extracted_entities.get(entity_name, '未找到')}")
                    else:
                        logger.warning(f"未找到实体: {entity_name}")
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            logger.info(f"实体抽取准确率: {accuracy:.2%} ({correct_count}/{total_count})")
            
            return accuracy > 0.5  # 简单阈值判断
            
        except Exception as e:
            logger.error(f"实体抽取测试失败: {e}")
            return False
    
    def test_relation_extraction(self, model_path: str):
        """测试关系抽取功能"""
        logger.info("开始测试关系抽取功能...")
        
        try:
            # 加载模型
            relation_extractor = RelationExtractor(model_path)
            logger.info("关系抽取模型加载成功")
            
            # 测试单个文本
            test_text = self.test_texts[0]
            
            # 先抽取实体
            from src.entity_extraction.deploy_ner import EntityExtractor
            entity_extractor = EntityExtractor(model_path.replace('relation', 'ner'))
            entities = entity_extractor.extract_entities(test_text)
            
            # 抽取关系
            relations = relation_extractor.extract_relations(test_text, entities)
            
            logger.info(f"测试文本: {test_text}")
            logger.info(f"抽取的关系: {json.dumps(relations, ensure_ascii=False, indent=2)}")
            
            # 验证结果
            extracted_relations = [(rel['head_entity'], rel['tail_entity'], rel['relation_type']) 
                                 for rel in relations]
            correct_count = 0
            total_count = 0
            
            for head, tail, expected_relation in self.expected_relations:
                if head in test_text and tail in test_text:
                    total_count += 1
                    if (head, tail, expected_relation) in extracted_relations:
                        correct_count += 1
                    else:
                        logger.warning(f"未找到关系: {head} -> {tail} ({expected_relation})")
            
            accuracy = correct_count / total_count if total_count > 0 else 0
            logger.info(f"关系抽取准确率: {accuracy:.2%} ({correct_count}/{total_count})")
            
            return accuracy > 0.3  # 简单阈值判断
            
        except Exception as e:
            logger.error(f"关系抽取测试失败: {e}")
            return False
    
    def test_joint_extraction(self, ner_model_path: str, relation_model_path: str):
        """测试联合抽取功能"""
        logger.info("开始测试联合抽取功能...")
        
        try:
            # 加载模型
            joint_extractor = JointExtractor(ner_model_path, relation_model_path)
            logger.info("联合抽取模型加载成功")
            
            # 测试单个文本
            test_text = self.test_texts[0]
            result = joint_extractor.extract_spo(test_text)
            
            logger.info(f"测试文本: {test_text}")
            logger.info(f"抽取的SPO: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 验证结果
            if result['entities'] and result['relations']:
                logger.info("联合抽取测试通过")
                return True
            else:
                logger.warning("联合抽取结果为空")
                return False
            
        except Exception as e:
            logger.error(f"联合抽取测试失败: {e}")
            return False
    
    def test_batch_processing(self, ner_model_path: str, relation_model_path: str):
        """测试批量处理功能"""
        logger.info("开始测试批量处理功能...")
        
        try:
            # 加载模型
            joint_extractor = JointExtractor(ner_model_path, relation_model_path)
            
            # 批量处理
            results = joint_extractor.extract_spo_batch(self.test_texts)
            
            logger.info(f"批量处理 {len(self.test_texts)} 个文本")
            logger.info(f"处理结果数量: {len(results)}")
            
            # 验证结果
            success_count = 0
            for i, result in enumerate(results):
                if result['entities'] or result['relations']:
                    success_count += 1
                    logger.info(f"文本 {i+1} 处理成功")
                else:
                    logger.warning(f"文本 {i+1} 处理失败")
            
            success_rate = success_count / len(results)
            logger.info(f"批量处理成功率: {success_rate:.2%} ({success_count}/{len(results)})")
            
            return success_rate > 0.5
            
        except Exception as e:
            logger.error(f"批量处理测试失败: {e}")
            return False
    
    def run_all_tests(self, ner_model_path: str, relation_model_path: str):
        """运行所有测试"""
        logger.info("开始运行系统测试...")
        
        test_results = {}
        
        # 测试实体抽取
        test_results['entity_extraction'] = self.test_entity_extraction(ner_model_path)
        
        # 测试关系抽取
        test_results['relation_extraction'] = self.test_relation_extraction(relation_model_path)
        
        # 测试联合抽取
        test_results['joint_extraction'] = self.test_joint_extraction(ner_model_path, relation_model_path)
        
        # 测试批量处理
        test_results['batch_processing'] = self.test_batch_processing(ner_model_path, relation_model_path)
        
        # 输出测试结果
        logger.info("\n" + "="*50)
        logger.info("测试结果汇总:")
        for test_name, result in test_results.items():
            status = "通过" if result else "失败"
            logger.info(f"{test_name}: {status}")
        
        # 计算总体成功率
        success_count = sum(test_results.values())
        total_count = len(test_results)
        overall_success_rate = success_count / total_count
        
        logger.info(f"总体成功率: {overall_success_rate:.2%} ({success_count}/{total_count})")
        
        if overall_success_rate >= 0.75:
            logger.info("系统测试通过!")
            return True
        else:
            logger.warning("系统测试失败!")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试实体抽取和关系抽取系统")
    parser.add_argument("--ner_model_path", type=str, 
                       default="./models/ner_models/best_ner_model.pth",
                       help="NER模型路径")
    parser.add_argument("--relation_model_path", type=str,
                       default="./models/relation_models/best_relation_model.pth",
                       help="关系抽取模型路径")
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.ner_model_path):
        logger.error(f"NER模型文件不存在: {args.ner_model_path}")
        logger.info("请先运行训练脚本: python train_models.py")
        return False
    
    if not os.path.exists(args.relation_model_path):
        logger.error(f"关系抽取模型文件不存在: {args.relation_model_path}")
        logger.info("请先运行训练脚本: python train_models.py")
        return False
    
    # 运行测试
    tester = SystemTester()
    success = tester.run_all_tests(args.ner_model_path, args.relation_model_path)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)