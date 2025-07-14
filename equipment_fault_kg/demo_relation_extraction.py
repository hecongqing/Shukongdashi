#!/usr/bin/env python3
"""
关系抽取模块演示脚本

展示如何使用关系抽取模块从数控机床故障诊断文本中提取关系
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from relation_extraction import RelationExtractor, RelationValidator, RelationPatterns

def demo_basic_extraction():
    """演示基本关系抽取"""
    print("=== 基本关系抽取演示 ===")
    
    extractor = RelationExtractor()
    
    # 示例文本
    text = """
    数控机床主轴故障导致加工精度下降，需要更换轴承解决。
    数控系统出现报警，原因是电源电压不稳定。
    刀具磨损严重，影响加工质量，建议更换新刀具。
    设备运行异常，维修人员使用万用表检测电路。
    伺服电机故障，维修时间约2小时，需要专业技术人员。
    """
    
    print("输入文本:")
    print(text.strip())
    print()
    
    # 提取关系
    relations = extractor.extract_relations(text)
    
    print(f"提取到 {len(relations)} 个关系:")
    for i, relation in enumerate(relations, 1):
        print(f"{i:2d}. {relation.subject} --{relation.predicate}--> {relation.object}")
        print(f"    类型: {relation.relation_type}, 置信度: {relation.confidence:.2f}")
        print(f"    来源: {relation.source_text[:50]}...")
        print()
    
    return relations

def demo_entity_based_extraction():
    """演示基于实体的关系抽取"""
    print("=== 基于实体的关系抽取演示 ===")
    
    extractor = RelationExtractor()
    
    # 示例文本和已知实体
    text = "主轴故障导致加工精度下降，需要更换轴承解决。维修人员使用万用表检测电路。"
    entities = ["主轴", "加工精度", "轴承", "维修人员", "万用表", "电路"]
    
    print("输入文本:")
    print(text)
    print()
    print("已知实体:")
    print(entities)
    print()
    
    # 基于实体提取关系
    relations = extractor.extract_relations(text, entities)
    
    print(f"基于实体提取到 {len(relations)} 个关系:")
    for i, relation in enumerate(relations, 1):
        print(f"{i:2d}. {relation.subject} --{relation.predicate}--> {relation.object}")
        print(f"    类型: {relation.relation_type}, 置信度: {relation.confidence:.2f}")
        print()
    
    return relations

def demo_relation_validation():
    """演示关系验证"""
    print("=== 关系验证演示 ===")
    
    extractor = RelationExtractor()
    validator = RelationValidator()
    
    # 包含低质量关系的文本
    text = """
    主轴故障导致加工精度下降。
    这是一个问题。
    设备出现故障。
    维修人员使用工具。
    """
    
    print("输入文本:")
    print(text.strip())
    print()
    
    # 提取关系
    relations = extractor.extract_relations(text)
    print(f"原始关系数: {len(relations)}")
    
    # 验证关系
    validated_relations = validator.validate_relations(relations, min_confidence=0.6)
    print(f"验证后关系数: {len(validated_relations)}")
    
    # 显示验证统计
    stats = validator.get_validation_statistics(relations, validated_relations)
    print(f"过滤率: {stats['filter_rate']:.2%}")
    
    print("\n验证通过的关系:")
    for i, relation in enumerate(validated_relations, 1):
        print(f"{i}. {relation.subject} --{relation.predicate}--> {relation.object}")
    
    return validated_relations

def demo_relation_statistics():
    """演示关系统计"""
    print("=== 关系统计演示 ===")
    
    extractor = RelationExtractor()
    
    # 示例文本
    text = """
    主轴故障导致加工精度下降，需要更换轴承解决。
    数控系统出现报警，原因是电源电压不稳定。
    刀具磨损严重，影响加工质量，建议更换新刀具。
    设备运行异常，维修人员使用万用表检测电路。
    伺服电机故障，维修时间约2小时，需要专业技术人员。
    """
    
    # 提取关系
    relations = extractor.extract_relations(text)
    
    # 获取统计信息
    stats = extractor.get_relation_statistics(relations)
    
    print(f"总关系数: {stats['total_relations']}")
    print()
    
    print("关系类型分布:")
    for relation_type, count in stats['relation_type_counts'].items():
        print(f"  {relation_type}: {count}")
    print()
    
    print("常见谓词:")
    for predicate, count in sorted(stats['predicate_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {predicate}: {count}")
    print()
    
    print("置信度分布:")
    for level, count in stats['confidence_distribution'].items():
        print(f"  {level}: {count}")
    
    return stats

def demo_relation_filtering():
    """演示关系过滤"""
    print("=== 关系过滤演示 ===")
    
    extractor = RelationExtractor()
    
    # 示例文本
    text = """
    主轴故障导致加工精度下降，需要更换轴承解决。
    数控系统出现报警，原因是电源电压不稳定。
    刀具磨损严重，影响加工质量，建议更换新刀具。
    设备运行异常，维修人员使用万用表检测电路。
    """
    
    # 提取关系
    relations = extractor.extract_relations(text)
    
    print(f"原始关系数: {len(relations)}")
    print()
    
    # 按置信度过滤
    high_confidence = extractor.filter_relations_by_confidence(relations, 0.7)
    print(f"高置信度关系数 (>=0.7): {len(high_confidence)}")
    
    medium_confidence = extractor.filter_relations_by_confidence(relations, 0.6)
    print(f"中等置信度关系数 (>=0.6): {len(medium_confidence)}")
    
    # 按类型过滤
    maintenance_relations = extractor.filter_relations_by_type(relations, 'maintenance_personnel')
    print(f"维修人员关系数: {len(maintenance_relations)}")
    
    return relations

def demo_pattern_management():
    """演示模式管理"""
    print("=== 模式管理演示 ===")
    
    patterns = RelationPatterns()
    
    # 显示所有模式类型
    all_patterns = patterns.get_all_patterns()
    print(f"支持的关系类型数: {len(all_patterns)}")
    print()
    
    print("关系类型列表:")
    for pattern_type in all_patterns.keys():
        chinese_name = patterns.get_relation_type_name(pattern_type)
        pattern_count = len(all_patterns[pattern_type])
        print(f"  {pattern_type} ({chinese_name}): {pattern_count} 个模式")
    print()
    
    # 显示特定类型的模式
    print("故障-症状关系模式:")
    fault_symptom_patterns = patterns.get_patterns_by_type('fault_symptom')
    for i, pattern in enumerate(fault_symptom_patterns, 1):
        print(f"  {i}. {pattern}")
    print()
    
    # 添加自定义模式
    print("添加自定义模式...")
    patterns.add_custom_pattern('custom_fault', r'([^，。；]*故障[^，。；]*)(引起|造成)([^，。；]*问题[^，。；]*)')
    
    # 验证模式
    is_valid = patterns.validate_pattern(r'([^，。；]*故障[^，。；]*)(引起|造成)([^，。；]*问题[^，。；]*)')
    print(f"自定义模式验证: {'通过' if is_valid else '失败'}")
    
    return patterns

def main():
    """主函数"""
    print("关系抽取模块演示")
    print("=" * 60)
    print()
    
    try:
        # 运行各种演示
        demo_basic_extraction()
        print()
        
        demo_entity_based_extraction()
        print()
        
        demo_relation_validation()
        print()
        
        demo_relation_statistics()
        print()
        
        demo_relation_filtering()
        print()
        
        demo_pattern_management()
        print()
        
        print("演示完成！")
        print()
        print("关系抽取模块功能总结:")
        print("1. 支持基于规则模式的关系抽取")
        print("2. 支持基于启发式方法的关系抽取")
        print("3. 支持基于已知实体的关系抽取")
        print("4. 提供关系验证和过滤功能")
        print("5. 提供关系统计和分析功能")
        print("6. 支持自定义关系模式")
        print("7. 支持多种关系类型：故障-症状、故障-原因、维修-工具等")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()