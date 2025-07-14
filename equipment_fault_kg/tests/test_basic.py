"""
基本功能测试

测试项目的核心功能模块
"""

import unittest
import sys
import os
sys.path.append('../src')

import yaml
from data_collection import DataProcessor
from entity_extraction import NERModel


class TestDataProcessor(unittest.TestCase):
    """测试数据处理器"""
    
    def setUp(self):
        """设置测试环境"""
        with open('../config/config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.processor = DataProcessor(self.config['entity_extraction'])
    
    def test_clean_text(self):
        """测试文本清洗"""
        dirty_text = "  这是一个  测试文本  \n\n  包含多余的空格和换行  "
        clean_text = self.processor.clean_text(dirty_text)
        self.assertIsInstance(clean_text, str)
        self.assertNotIn('\n', clean_text)
    
    def test_extract_equipment_info(self):
        """测试装备信息提取"""
        text = "某工厂数控车床在加工过程中出现主轴异常振动"
        info = self.processor.extract_equipment_info(text)
        self.assertIsInstance(info, dict)
        self.assertIn('equipment_type', info)
        self.assertIn('components', info)
    
    def test_extract_fault_info(self):
        """测试故障信息提取"""
        text = "主轴异常振动，经检查发现轴承磨损严重，更换轴承后故障排除"
        info = self.processor.extract_fault_info(text)
        self.assertIsInstance(info, dict)
        self.assertIn('fault_type', info)
        self.assertIn('symptoms', info)
        self.assertIn('causes', info)
        self.assertIn('solutions', info)


class TestNERModel(unittest.TestCase):
    """测试NER模型"""
    
    def setUp(self):
        """设置测试环境"""
        with open('../config/config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.ner_model = NERModel(self.config['entity_extraction'])
    
    def test_model_initialization(self):
        """测试模型初始化"""
        self.assertIsNotNone(self.ner_model.config)
        self.assertIsNotNone(self.ner_model.tokenizer)
    
    def test_predict_entities(self):
        """测试实体预测"""
        text = "数控车床主轴异常振动"
        entities = self.ner_model.predict(text)
        self.assertIsInstance(entities, list)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_processing(self):
        """端到端处理测试"""
        # 创建测试数据
        test_data = [
            {
                "id": "test_001",
                "title": "数控车床主轴异常振动故障诊断",
                "content": "某工厂数控车床在加工过程中出现主轴异常振动，经检查发现主轴轴承磨损严重，更换轴承后故障排除。",
                "source": "测试数据",
                "url": ""
            }
        ]
        
        # 加载配置
        with open('../config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 数据处理
        processor = DataProcessor(config['entity_extraction'])
        processed_data = processor.process_batch(test_data)
        
        # 验证处理结果
        self.assertEqual(len(processed_data), 1)
        self.assertIn('equipment_info', processed_data[0])
        self.assertIn('fault_info', processed_data[0])
        
        # 实体抽取
        ner_model = NERModel(config['entity_extraction'])
        text = f"{processed_data[0]['title']} {processed_data[0]['content']}"
        entities = ner_model.predict(text)
        
        # 验证实体抽取结果
        self.assertIsInstance(entities, list)


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestNERModel))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print(f"\n测试结果: {result.testsRun} 个测试用例")
    print(f"失败: {len(result.failures)} 个")
    print(f"错误: {len(result.errors)} 个")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")