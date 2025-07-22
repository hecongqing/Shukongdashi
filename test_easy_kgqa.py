"""
EASY KGQA Framework 测试脚本
验证框架的基本功能
"""

import sys
import os

# 添加路径以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from easy_kgqa_framework import EasyAnalyzer
from easy_kgqa_framework.utils.text_processor import SimpleTextProcessor
from easy_kgqa_framework.models.entities import FaultType


def test_text_processor():
    """测试文本处理器"""
    print("=" * 50)
    print("测试文本处理器")
    print("=" * 50)
    
    processor = SimpleTextProcessor()
    
    test_text = "主轴不转，刀库换刀时出现ALM401报警"
    
    # 测试分词
    words = processor.segment_text(test_text)
    print(f"原文本: {test_text}")
    print(f"分词结果: {words}")
    
    # 测试故障元素提取
    elements = processor.extract_fault_elements(test_text)
    print("提取的故障元素:")
    for element in elements:
        print(f"  - {element.content} ({element.element_type.value}) [置信度: {element.confidence}]")
    
    # 测试报警代码提取
    alarm_codes = processor.extract_alarm_codes(test_text)
    print(f"提取的报警代码: {alarm_codes}")
    
    return len(elements) > 0


def test_configuration():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试配置")
    print("=" * 50)
    
    from easy_kgqa_framework.config import config
    
    print(f"Neo4j URI: {config.NEO4J_URI}")
    print(f"实体识别服务URL: {config.ENTITY_SERVICE_URL}")
    print(f"最大查询结果数: {config.MAX_QUERY_RESULTS}")
    
    return True


def test_analyzer_initialization():
    """测试分析器初始化"""
    print("\n" + "=" * 50)
    print("测试分析器初始化")
    print("=" * 50)
    
    try:
        analyzer = EasyAnalyzer()
        print("✅ 分析器初始化成功")
        
        # 测试系统状态
        status = analyzer.get_system_status()
        print(f"Neo4j状态: {status['neo4j']['status']}")
        print(f"实体识别服务状态: {status['entity_service']['status']}")
        
        analyzer.close()
        return True
        
    except Exception as e:
        print(f"❌ 分析器初始化失败: {e}")
        return False


def test_basic_analysis():
    """测试基本分析功能"""
    print("\n" + "=" * 50)
    print("测试基本分析功能")
    print("=" * 50)
    
    try:
        with EasyAnalyzer() as analyzer:
            test_question = "主轴不转是什么原因"
            print(f"测试问题: {test_question}")
            
            # 只使用内部文本处理（避免依赖外部服务）
            result = analyzer.analyze_question(test_question, use_entity_service=False)
            
            print(f"分析结果:")
            print(f"  提取元素数量: {len(result.elements)}")
            print(f"  整体置信度: {result.confidence}")
            
            if result.elements:
                print("  提取的元素:")
                for element in result.elements:
                    print(f"    - {element.content} ({element.element_type.value})")
            
            return True
            
    except Exception as e:
        print(f"❌ 基本分析测试失败: {e}")
        return False


def test_simple_qa():
    """测试简单问答功能"""
    print("\n" + "=" * 50)
    print("测试简单问答功能")
    print("=" * 50)
    
    try:
        with EasyAnalyzer() as analyzer:
            test_question = "什么是主轴"
            print(f"测试问题: {test_question}")
            
            results = analyzer.simple_qa(test_question)
            print(f"查询结果数量: {len(results)}")
            
            if results:
                print("前3个结果:")
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {result.get('name', '未知')}")
            
            return True
            
    except Exception as e:
        print(f"❌ 简单问答测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("EASY KGQA Framework 功能测试")
    print("注意：某些测试可能因为服务未启动而失败，这是正常的")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("文本处理器", test_text_processor()))
    test_results.append(("配置", test_configuration()))
    test_results.append(("分析器初始化", test_analyzer_initialization()))
    test_results.append(("基本分析", test_basic_analysis()))
    test_results.append(("简单问答", test_simple_qa()))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！EASY KGQA Framework 运行正常。")
    elif passed > total // 2:
        print("⚠️ 大部分测试通过，框架基本功能正常。")
    else:
        print("⚠️ 多项测试失败，请检查Neo4j和实体识别服务是否正常运行。")


if __name__ == "__main__":
    main()