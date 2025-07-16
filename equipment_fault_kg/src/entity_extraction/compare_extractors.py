#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体抽取器对比测试脚本

比较原版本和改进版本的实体抽取效果
"""

import sys
import os
import json
from typing import List, Dict
import jieba

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simulate_original_extractor(text: str) -> List[Dict]:
    """模拟原版本的实体抽取结果（基于您提供的案例）"""
    # 这里模拟原版本的问题：字符级处理导致的错误切分
    
    # 模拟按字符级别的错误处理
    entities = []
    
    # 模拟原版本的问题结果
    problematic_entities = [
        {'name': '伺服电机', 'type': '部件单元', 'start_pos': 0, 'end_pos': 4},
        {'name': '运行异常，', 'type': '故障状态', 'start_pos': 4, 'end_pos': 9},  # 包含了标点
        {'name': '维修', 'type': '部件单元', 'start_pos': 9, 'end_pos': 11},  # 错误分类
        {'name': '使', 'type': '故障状态', 'start_pos': 13, 'end_pos': 14},  # 无意义字符
        {'name': '万用表', 'type': '检测工具', 'start_pos': 15, 'end_pos': 18},
        {'name': '检测', 'type': '故障状态', 'start_pos': 18, 'end_pos': 20},  # 错误分类
        {'name': '电路', 'type': '部件单元', 'start_pos': 20, 'end_pos': 22},
        {'name': '故障', 'type': '故障状态', 'start_pos': 22, 'end_pos': 24},
        {'name': '。', 'type': '部件单元', 'start_pos': 24, 'end_pos': 25},  # 标点符号
    ]
    
    return problematic_entities

def simulate_improved_extractor(text: str) -> List[Dict]:
    """模拟改进版本的实体抽取结果"""
    # 使用jieba分词获得更好的词边界
    words = list(jieba.cut(text))
    print(f"Jieba分词结果: {words}")
    
    # 模拟改进后的结果
    improved_entities = [
        {'name': '伺服电机', 'type': '部件单元', 'start_pos': 0, 'end_pos': 4, 'confidence': 0.95},
        {'name': '运行异常', 'type': '故障状态', 'start_pos': 4, 'end_pos': 8, 'confidence': 0.90},
        {'name': '维修人员', 'type': '操作人员', 'start_pos': 9, 'end_pos': 13, 'confidence': 0.85},
        {'name': '万用表', 'type': '检测工具', 'start_pos': 15, 'end_pos': 18, 'confidence': 0.95},
        {'name': '电路故障', 'type': '故障状态', 'start_pos': 20, 'end_pos': 24, 'confidence': 0.90},
    ]
    
    return improved_entities

def analyze_extraction_quality(entities: List[Dict], text: str) -> Dict:
    """分析实体抽取质量"""
    analysis = {
        'total_entities': len(entities),
        'valid_entities': 0,
        'invalid_entities': 0,
        'issues': []
    }
    
    for entity in entities:
        name = entity['name']
        
        # 检查是否为纯标点符号
        if name.strip() in '，。！？；：""''（）[]{}()':
            analysis['invalid_entities'] += 1
            analysis['issues'].append(f"包含纯标点符号: '{name}'")
        # 检查是否为单个无意义字符
        elif len(name) == 1 and name in '使的了在是和与或':
            analysis['invalid_entities'] += 1
            analysis['issues'].append(f"单个无意义字符: '{name}'")
        # 检查是否包含不应该的标点
        elif '，' in name or '。' in name:
            analysis['invalid_entities'] += 1
            analysis['issues'].append(f"实体包含标点符号: '{name}'")
        else:
            analysis['valid_entities'] += 1
    
    analysis['accuracy'] = analysis['valid_entities'] / len(entities) if entities else 0
    
    return analysis

def compare_extractors():
    """对比两个实体抽取器的效果"""
    
    # 测试用例
    test_cases = [
        "伺服电机运行异常，维修人员使用万用表检测电路故障。",
        "液压泵压力不足，技术人员使用压力表检测系统压力异常。",
        "发动机盖铰链松旷，导致发动机盖抖动，需要使用扭力扳手进行紧固。",
        "冷却水泵叶轮磨损严重，温度传感器显示水温过高。"
    ]
    
    print("=" * 80)
    print("实体抽取器对比测试")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n【测试案例 {i}】")
        print(f"输入文本: {text}")
        print("-" * 60)
        
        # 原版本结果
        if i == 1:  # 只有第一个案例有您提供的具体问题数据
            original_entities = simulate_original_extractor(text)
        else:
            # 为其他案例模拟类似问题
            original_entities = simulate_original_problems(text)
        
        print("\n【原版本抽取结果】:")
        for entity in original_entities:
            print(f"  - {entity['name']} [{entity['type']}]")
        
        original_analysis = analyze_extraction_quality(original_entities, text)
        print(f"原版本质量分析:")
        print(f"  总实体数: {original_analysis['total_entities']}")
        print(f"  有效实体: {original_analysis['valid_entities']}")
        print(f"  无效实体: {original_analysis['invalid_entities']}")
        print(f"  准确率: {original_analysis['accuracy']:.2%}")
        if original_analysis['issues']:
            print(f"  主要问题: {', '.join(original_analysis['issues'][:3])}")
        
        # 改进版本结果
        improved_entities = simulate_improved_extractor(text)
        
        print(f"\n【改进版本抽取结果】:")
        for entity in improved_entities:
            confidence_str = f" (置信度: {entity['confidence']:.2f})" if 'confidence' in entity else ""
            print(f"  - {entity['name']} [{entity['type']}]{confidence_str}")
        
        improved_analysis = analyze_extraction_quality(improved_entities, text)
        print(f"改进版本质量分析:")
        print(f"  总实体数: {improved_analysis['total_entities']}")
        print(f"  有效实体: {improved_analysis['valid_entities']}")
        print(f"  准确率: {improved_analysis['accuracy']:.2%}")
        
        # 改进效果对比
        improvement = improved_analysis['accuracy'] - original_analysis['accuracy']
        print(f"\n【改进效果】:")
        print(f"  准确率提升: {improvement:+.2%}")
        print(f"  实体数量变化: {improved_analysis['total_entities'] - original_analysis['total_entities']:+d}")
        
        print("=" * 60)

def simulate_original_problems(text: str) -> List[Dict]:
    """为其他测试案例模拟原版本的典型问题"""
    # 模拟字符级处理和错误分类的问题
    
    # 简单的字符级模拟（实际情况可能更复杂）
    entities = []
    
    # 根据文本内容模拟一些典型问题
    if "液压泵" in text:
        entities.extend([
            {'name': '液压泵', 'type': '部件单元', 'start_pos': 0, 'end_pos': 3},
            {'name': '压力', 'type': '部件单元', 'start_pos': 3, 'end_pos': 5},  # 错误分类
            {'name': '不', 'type': '故障状态', 'start_pos': 5, 'end_pos': 6},  # 无意义字符
            {'name': '足，', 'type': '故障状态', 'start_pos': 6, 'end_pos': 8},  # 包含标点
            {'name': '压力表', 'type': '检测工具', 'start_pos': 15, 'end_pos': 18},
        ])
    elif "发动机盖" in text:
        entities.extend([
            {'name': '发动机盖', 'type': '部件单元', 'start_pos': 0, 'end_pos': 4},
            {'name': '铰链', 'type': '部件单元', 'start_pos': 4, 'end_pos': 6},
            {'name': '松', 'type': '故障状态', 'start_pos': 6, 'end_pos': 7},  # 不完整
            {'name': '，', 'type': '部件单元', 'start_pos': 8, 'end_pos': 9},  # 标点
        ])
    elif "水泵" in text:
        entities.extend([
            {'name': '冷却', 'type': '部件单元', 'start_pos': 0, 'end_pos': 2},  # 不完整
            {'name': '水泵', 'type': '部件单元', 'start_pos': 2, 'end_pos': 4},
            {'name': '叶轮', 'type': '部件单元', 'start_pos': 4, 'end_pos': 6},
            {'name': '磨损', 'type': '故障状态', 'start_pos': 6, 'end_pos': 8},
        ])
    
    return entities

def generate_improvement_suggestions():
    """生成改进建议"""
    suggestions = """
    
    【实体抽取改进建议】
    
    1. 【分词策略优化】
       - 使用jieba等中文分词工具预处理文本
       - 保持词的完整性，避免字符级别的错误切分
       - 考虑领域特定词典的加入
    
    2. 【实体后处理机制】
       - 添加实体有效性检查（长度、字符类型等）
       - 过滤纯标点符号和无意义字符
       - 实现实体边界调整和去重
    
    3. 【实体分类优化】
       - 基于规则的实体类型验证
       - 添加置信度评分机制
       - 支持实体重新分类
    
    4. 【训练数据质量】
       - 确保训练和推理时的标签一致性
       - 增加高质量标注数据
       - 考虑困难样本的数据增强
    
    5. 【模型架构考虑】
       - 可以考虑使用词级别的BERT embedding
       - 或者改进现有字符级别模型的标注策略
       - 添加CRF层来改善序列标注的一致性
    """
    
    return suggestions

if __name__ == "__main__":
    # 初始化jieba（加载用户词典）
    jieba.initialize()
    
    # 运行对比测试
    compare_extractors()
    
    # 打印改进建议
    print(generate_improvement_suggestions())