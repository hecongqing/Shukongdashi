#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析NER实体抽取问题
不依赖外部库的静态分析
"""

import os
import re

def analyze_label_definitions():
    """分析不同文件中的标签定义"""
    print("=" * 60)
    print("标签定义分析")
    print("=" * 60)
    
    # 分析ner_model.py
    print("\n1. ner_model.py 中的标签定义:")
    ner_model_labels = {
        'O': 0,
        'B-Equipment': 1, 'I-Equipment': 2,
        'B-Component': 3, 'I-Component': 4,
        'B-Fault': 5, 'I-Fault': 6,
        'B-Cause': 7, 'I-Cause': 8,
        'B-Solution': 9, 'I-Solution': 10,
        'B-Symptom': 11, 'I-Symptom': 12,
        'B-Material': 13, 'I-Material': 14,
        'B-Tool': 15, 'I-Tool': 16
    }
    print("   实体类型: Equipment, Component, Fault, Cause, Solution, Symptom, Material, Tool")
    print(f"   标签数量: {len(ner_model_labels)}")
    
    # 分析train_ner.py
    print("\n2. train_ner.py 中的标签定义:")
    train_ner_labels = {
        'O': 0,
        'B-COMPONENT': 1, 'I-COMPONENT': 2,
        'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
        'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
        'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
    }
    print("   实体类型: COMPONENT, PERFORMANCE, FAULT_STATE, DETECTION_TOOL")
    print(f"   标签数量: {len(train_ner_labels)}")
    
    # 分析deploy_ner.py
    print("\n3. deploy_ner.py 中的标签定义:")
    deploy_ner_labels = {
        'O': 0,
        'B-COMPONENT': 1, 'I-COMPONENT': 2,
        'B-PERFORMANCE': 3, 'I-PERFORMANCE': 4,
        'B-FAULT_STATE': 5, 'I-FAULT_STATE': 6,
        'B-DETECTION_TOOL': 7, 'I-DETECTION_TOOL': 8
    }
    print("   实体类型: COMPONENT, PERFORMANCE, FAULT_STATE, DETECTION_TOOL")
    print(f"   标签数量: {len(deploy_ner_labels)}")
    
    # 分析不一致性
    print("\n4. 标签定义不一致性分析:")
    print("   ❌ ner_model.py 与其他两个文件使用不同的标签体系")
    print("   ✅ train_ner.py 和 deploy_ner.py 使用相同的标签体系")
    print("   ❌ 实体类型数量不匹配: ner_model.py(8种) vs 其他文件(4种)")

def analyze_entity_mapping():
    """分析实体类型映射"""
    print("\n" + "=" * 60)
    print("实体类型映射分析")
    print("=" * 60)
    
    # deploy_ner.py 中的映射
    print("\n1. deploy_ner.py 中的实体类型映射:")
    entity_mapping = {
        'COMPONENT': '部件单元',
        'PERFORMANCE': '性能表征',
        'FAULT_STATE': '故障状态',
        'DETECTION_TOOL': '检测工具'
    }
    
    for eng, chn in entity_mapping.items():
        print(f"   {eng} -> {chn}")
    
    # 分析用户提到的错误结果
    print("\n2. 用户提到的错误抽取结果分析:")
    wrong_results = [
        ("伺服电机", "部件单元"),
        ("运行异常，", "故障状态"),
        ("维修", "部件单元"),
        ("使", "故障状态"),
        ("万用表", "检测工具"),
        ("检测", "故障状态"),
        ("电路", "部件单元"),
        ("故障", "故障状态"),
        ("。", "部件单元")
    ]
    
    print("   错误分析:")
    for entity, entity_type in wrong_results:
        if entity == "。" or entity == "，":
            print(f"   ❌ '{entity}' 被错误分类为 {entity_type} (应该是标点符号)")
        elif entity == "使":
            print(f"   ❌ '{entity}' 被错误分类为 {entity_type} (应该是动词)")
        elif entity == "检测":
            print(f"   ❌ '{entity}' 被错误分类为 {entity_type} (应该是动词)")
        elif entity == "运行异常，":
            print(f"   ❌ '{entity}' 包含标点符号，应该只抽取 '运行异常'")
        else:
            print(f"   ✅ '{entity}' 分类为 {entity_type} (可能正确)")

def analyze_tokenization_issues():
    """分析分词和标签对齐问题"""
    print("\n" + "=" * 60)
    print("分词和标签对齐问题分析")
    print("=" * 60)
    
    test_text = "伺服电机运行异常，维修人员使用万用表检测电路故障。"
    print(f"\n测试文本: {test_text}")
    
    # 字符级别的分析
    print("\n1. 字符级别分析:")
    for i, char in enumerate(test_text):
        print(f"   位置{i:2d}: '{char}'")
    
    # 分析可能的分词问题
    print("\n2. 潜在的分词问题:")
    print("   - BERT tokenizer 可能将中文字符分割成多个sub-token")
    print("   - 字符级别的标签需要正确映射到token级别")
    print("   - 标点符号可能被错误处理")
    
    # 分析标签对齐问题
    print("\n3. 标签对齐问题:")
    print("   - 原始标签是按字符级别标注的")
    print("   - BERT分词后，一个字符可能对应多个token")
    print("   - 需要确保标签正确传播到所有相关的token")

def analyze_training_data_issues():
    """分析训练数据问题"""
    print("\n" + "=" * 60)
    print("训练数据问题分析")
    print("=" * 60)
    
    print("\n1. 可能的训练数据问题:")
    print("   - 标注质量不高，存在错误标注")
    print("   - 标点符号被错误标注为实体")
    print("   - 动词被错误标注为实体")
    print("   - 实体边界标注不准确")
    
    print("\n2. 数据预处理问题:")
    print("   - 字符到token的映射可能有问题")
    print("   - 标签传播逻辑可能有bug")
    print("   - 没有过滤掉明显错误的实体")

def suggest_solutions():
    """建议解决方案"""
    print("\n" + "=" * 60)
    print("解决方案建议")
    print("=" * 60)
    
    print("\n1. 立即修复的问题:")
    print("   🔧 统一标签定义 - 选择一种标签体系并在所有文件中使用")
    print("   🔧 修复标签对齐逻辑 - 确保字符标签正确映射到token")
    print("   🔧 添加后处理规则 - 过滤标点符号和明显错误的实体")
    
    print("\n2. 中期改进:")
    print("   📈 改进训练数据质量 - 重新标注或清洗数据")
    print("   📈 优化实体类型定义 - 根据实际业务需求调整")
    print("   📈 添加实体验证规则 - 基于领域知识验证实体合理性")
    
    print("\n3. 长期优化:")
    print("   🚀 使用领域预训练模型 - 如中文BERT或领域特定模型")
    print("   🚀 集成规则和统计方法 - 结合规则和深度学习")
    print("   🚀 添加实体关系抽取 - 利用实体间关系提高准确性")

def create_fix_plan():
    """创建修复计划"""
    print("\n" + "=" * 60)
    print("修复计划")
    print("=" * 60)
    
    print("\n第一步: 统一标签定义")
    print("1. 选择标签体系: 建议使用 train_ner.py 和 deploy_ner.py 中的4类标签")
    print("2. 修改 ner_model.py 中的标签定义")
    print("3. 确保所有文件使用相同的标签映射")
    
    print("\n第二步: 修复标签对齐")
    print("1. 检查字符到token的映射逻辑")
    print("2. 修复标签传播算法")
    print("3. 添加边界检查")
    
    print("\n第三步: 添加后处理")
    print("1. 过滤标点符号实体")
    print("2. 过滤单字符动词实体")
    print("3. 合并被错误分割的实体")
    
    print("\n第四步: 重新训练")
    print("1. 使用修复后的代码重新训练模型")
    print("2. 验证训练效果")
    print("3. 测试实际抽取效果")

if __name__ == "__main__":
    analyze_label_definitions()
    analyze_entity_mapping()
    analyze_tokenization_issues()
    analyze_training_data_issues()
    suggest_solutions()
    create_fix_plan()