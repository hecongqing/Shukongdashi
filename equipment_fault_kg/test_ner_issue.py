#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试实体抽取问题
重现用户提到的错误抽取结果
"""

import sys
import os
sys.path.append('./src')

from entity_extraction.deploy_ner import EntityExtractor

def test_entity_extraction():
    """测试实体抽取功能"""
    
    # 测试文本
    test_text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    
    print(f"测试文本: {test_text}")
    print("=" * 50)
    
    try:
        # 尝试加载模型
        model_path = './models/best_ner_model.pth'
        if os.path.exists(model_path):
            print("找到模型文件，开始测试...")
            extractor = EntityExtractor(model_path)
            
            # 抽取实体
            entities = extractor.extract_entities(test_text)
            
            print("抽取结果:")
            for entity in entities:
                print(f"  - {entity['name']} [{entity['type']}]")
                
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型")
            
            # 模拟错误的抽取结果（基于用户描述）
            print("\n模拟错误的抽取结果:")
            wrong_entities = [
                {"name": "伺服电机", "type": "部件单元"},
                {"name": "运行异常，", "type": "故障状态"},
                {"name": "维修", "type": "部件单元"},
                {"name": "使", "type": "故障状态"},
                {"name": "万用表", "type": "检测工具"},
                {"name": "检测", "type": "故障状态"},
                {"name": "电路", "type": "部件单元"},
                {"name": "故障", "type": "故障状态"},
                {"name": "。", "type": "部件单元"}
            ]
            
            for entity in wrong_entities:
                print(f"  - {entity['name']} [{entity['type']}]")
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def analyze_issues():
    """分析问题原因"""
    print("\n" + "=" * 50)
    print("问题分析:")
    print("=" * 50)
    
    print("1. 标签定义不一致问题:")
    print("   - ner_model.py 中使用: B-Equipment, B-Component, B-Fault, B-Cause, B-Solution, B-Symptom, B-Material, B-Tool")
    print("   - train_ner.py 中使用: B-COMPONENT, B-PERFORMANCE, B-FAULT_STATE, B-DETECTION_TOOL")
    print("   - deploy_ner.py 中使用: B-COMPONENT, B-PERFORMANCE, B-FAULT_STATE, B-DETECTION_TOOL")
    
    print("\n2. 实体类型映射问题:")
    print("   - 代码中定义了4种实体类型: 部件单元, 性能表征, 故障状态, 检测工具")
    print("   - 但实际抽取结果显示了错误的分类")
    
    print("\n3. 分词和标签对齐问题:")
    print("   - 可能存在字符级别的标签与token级别的不匹配")
    print("   - 标点符号被错误分类为实体")
    
    print("\n4. 模型训练数据问题:")
    print("   - 可能训练数据质量不高")
    print("   - 标签标注不准确")

def suggest_solutions():
    """建议解决方案"""
    print("\n" + "=" * 50)
    print("解决方案建议:")
    print("=" * 50)
    
    print("1. 统一标签定义:")
    print("   - 在所有文件中使用相同的标签体系")
    print("   - 建议使用: B-COMPONENT, B-PERFORMANCE, B-FAULT_STATE, B-DETECTION_TOOL")
    
    print("\n2. 改进标签对齐:")
    print("   - 确保字符级别的标签正确映射到token级别")
    print("   - 处理标点符号和特殊字符")
    
    print("\n3. 优化实体类型映射:")
    print("   - 重新定义实体类型，使其更符合业务需求")
    print("   - 添加后处理规则过滤明显错误的实体")
    
    print("\n4. 改进训练数据:")
    print("   - 使用更高质量的标注数据")
    print("   - 增加数据清洗和验证步骤")
    
    print("\n5. 添加后处理规则:")
    print("   - 过滤掉标点符号实体")
    print("   - 合并被错误分割的实体")
    print("   - 验证实体的合理性")

if __name__ == "__main__":
    test_entity_extraction()
    analyze_issues()
    suggest_solutions()