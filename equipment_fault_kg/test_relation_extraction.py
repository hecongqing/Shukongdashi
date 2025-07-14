#!/usr/bin/env python3
"""
关系抽取模块测试脚本

测试关系抽取的各种功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from relation_extraction import RelationExtractor, RelationValidator, RelationPatterns

def test_relation_patterns():
    """测试关系模式"""
    print("=== 测试关系模式 ===")
    
    patterns = RelationPatterns()
    
    # 测试获取所有模式
    all_patterns = patterns.get_all_patterns()
    print(f"总模式类型数: {len(all_patterns)}")
    
    # 测试获取特定类型模式
    fault_patterns = patterns.get_patterns_by_type('fault_symptom')
    print(f"故障-症状模式数: {len(fault_patterns)}")
    
    # 测试模式统计
    stats = patterns.get_pattern_statistics()
    print(f"模式统计: {stats}")
    
    print()

def test_relation_extraction():
    """测试关系抽取"""
    print("=== 测试关系抽取 ===")
    
    extractor = RelationExtractor()
    
    # 测试文本
    test_texts = [
        "主轴故障导致加工精度下降，需要更换轴承解决。",
        "数控系统出现报警，原因是电源电压不稳定。",
        "刀具磨损严重，影响加工质量，建议更换新刀具。",
        "设备运行异常，维修人员使用万用表检测电路。",
        "伺服电机故障，维修时间约2小时，需要专业技术人员。"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: {text}")
        
        # 提取关系
        relations = extractor.extract_relations(text)
        
        print(f"提取到 {len(relations)} 个关系:")
        for j, relation in enumerate(relations, 1):
            print(f"  {j}. {relation.subject} --{relation.predicate}--> {relation.object}")
            print(f"     类型: {relation.relation_type}, 置信度: {relation.confidence:.2f}")
            print(f"     来源: {relation.source_text}")
    
    print()

def test_relation_extraction_with_entities():
    """测试基于实体的关系抽取"""
    print("=== 测试基于实体的关系抽取 ===")
    
    extractor = RelationExtractor()
    
    # 测试文本和实体
    text = "主轴故障导致加工精度下降，需要更换轴承解决。维修人员使用万用表检测电路。"
    entities = ["主轴", "加工精度", "轴承", "维修人员", "万用表", "电路"]
    
    print(f"文本: {text}")
    print(f"实体: {entities}")
    
    # 基于实体提取关系
    relations = extractor.extract_relations(text, entities)
    
    print(f"\n提取到 {len(relations)} 个关系:")
    for i, relation in enumerate(relations, 1):
        print(f"  {i}. {relation.subject} --{relation.predicate}--> {relation.object}")
        print(f"     类型: {relation.relation_type}, 置信度: {relation.confidence:.2f}")
    
    print()

def test_relation_validation():
    """测试关系验证"""
    print("=== 测试关系验证 ===")
    
    extractor = RelationExtractor()
    validator = RelationValidator()
    
    # 测试文本（包含一些低质量关系）
    test_text = "主轴故障导致加工精度下降。这是一个问题。设备出现故障。"
    
    # 提取关系
    relations = extractor.extract_relations(test_text)
    print(f"原始关系数: {len(relations)}")
    
    # 验证关系
    validated_relations = validator.validate_relations(relations, min_confidence=0.6)
    print(f"验证后关系数: {len(validated_relations)}")
    
    # 获取验证统计
    stats = validator.get_validation_statistics(relations, validated_relations)
    print(f"验证统计: {stats}")
    
    print()

def test_relation_statistics():
    """测试关系统计"""
    print("=== 测试关系统计 ===")
    
    extractor = RelationExtractor()
    
    # 测试文本
    test_text = """
    主轴故障导致加工精度下降，需要更换轴承解决。
    数控系统出现报警，原因是电源电压不稳定。
    刀具磨损严重，影响加工质量，建议更换新刀具。
    设备运行异常，维修人员使用万用表检测电路。
    伺服电机故障，维修时间约2小时，需要专业技术人员。
    """
    
    # 提取关系
    relations = extractor.extract_relations(test_text)
    
    # 获取统计信息
    stats = extractor.get_relation_statistics(relations)
    
    print(f"总关系数: {stats['total_relations']}")
    print(f"关系类型统计: {stats['relation_type_counts']}")
    print(f"谓词统计: {stats['predicate_counts']}")
    print(f"置信度分布: {stats['confidence_distribution']}")
    
    print()

def test_relation_filtering():
    """测试关系过滤"""
    print("=== 测试关系过滤 ===")
    
    extractor = RelationExtractor()
    
    # 测试文本
    test_text = """
    主轴故障导致加工精度下降，需要更换轴承解决。
    数控系统出现报警，原因是电源电压不稳定。
    刀具磨损严重，影响加工质量，建议更换新刀具。
    """
    
    # 提取关系
    relations = extractor.extract_relations(test_text)
    
    # 按类型过滤
    fault_relations = extractor.filter_relations_by_type(relations, 'fault_symptom')
    print(f"故障-症状关系数: {len(fault_relations)}")
    
    # 按置信度过滤
    high_confidence_relations = extractor.filter_relations_by_confidence(relations, 0.7)
    print(f"高置信度关系数: {len(high_confidence_relations)}")
    
    print()

def main():
    """主函数"""
    print("关系抽取模块测试")
    print("=" * 50)
    
    try:
        # 运行各种测试
        test_relation_patterns()
        test_relation_extraction()
        test_relation_extraction_with_entities()
        test_relation_validation()
        test_relation_statistics()
        test_relation_filtering()
        
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()