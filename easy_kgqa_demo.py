"""
EASY KGQA Framework 演示程序
展示如何使用简化版知识图谱问答框架
"""

import sys
import os

# 添加路径以导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from easy_kgqa_framework import EasyAnalyzer


def demo_basic_usage():
    """基本使用演示"""
    print("=" * 60)
    print("EASY KGQA Framework 基本使用演示")
    print("=" * 60)
    
    # 初始化分析器
    with EasyAnalyzer() as analyzer:
        # 测试问题列表
        test_questions = [
            "主轴不转是什么原因",
            "刀库换刀时出现ALM401报警",
            "液压系统压力不够怎么办",
            "数控机床开机后显示屏无显示",
            "自动换刀时刀链运转不到位"
        ]
        
        # 分析每个问题
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 问题: {question}")
            print("-" * 40)
            
            # 分析问题
            result = analyzer.analyze_question(question)
            
            # 显示提取的元素
            print("提取的故障元素:")
            for element in result.elements:
                print(f"  - {element.content} ({element.element_type.value}) [置信度: {element.confidence}]")
            
            # 显示推理路径
            if result.reasoning_path:
                print("推理路径:")
                for path in result.reasoning_path[:3]:  # 只显示前3个
                    print(f"  - {path}")
            
            # 显示置信度
            print(f"整体置信度: {result.confidence}")


def demo_simple_qa():
    """简单问答演示"""
    print("\n" + "=" * 60)
    print("简单问答功能演示")
    print("=" * 60)
    
    with EasyAnalyzer() as analyzer:
        qa_questions = [
            "什么是主轴",
            "液压系统的作用",
            "常见的报警代码有哪些"
        ]
        
        for question in qa_questions:
            print(f"\n问题: {question}")
            print("-" * 40)
            
            results = analyzer.simple_qa(question)
            if results:
                for i, result in enumerate(results[:3], 1):
                    print(f"{i}. {result.get('name', '')}")
                    if result.get('content'):
                        print(f"   内容: {result['content']}")
            else:
                print("未找到相关信息")


def demo_system_status():
    """系统状态演示"""
    print("\n" + "=" * 60)
    print("系统状态检查")
    print("=" * 60)
    
    with EasyAnalyzer() as analyzer:
        status = analyzer.get_system_status()
        
        print("Neo4j状态:")
        neo4j = status['neo4j']
        print(f"  连接状态: {neo4j['status']}")
        print(f"  数据库URI: {neo4j['uri']}")
        
        if neo4j['statistics']:
            stats = neo4j['statistics']
            print(f"  节点数量: {stats.get('node_count', 0)}")
            print(f"  关系数量: {stats.get('relation_count', 0)}")
            print(f"  标签类型: {stats.get('label_count', 0)}")
        
        print("\n实体识别服务状态:")
        entity = status['entity_service']
        print(f"  连接状态: {entity['status']}")
        print(f"  服务URL: {entity['url']}")
        
        print("\n系统信息:")
        system = status['system']
        print(f"  版本: {system['version']}")
        print(f"  模式: {system['mode']}")


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 60)
    print("交互模式 - 输入 'quit' 退出")
    print("=" * 60)
    
    with EasyAnalyzer() as analyzer:
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见!")
                    break
                
                if not question:
                    continue
                
                print("分析中...")
                result = analyzer.analyze_question(question)
                
                print(f"\n分析结果 (置信度: {result.confidence}):")
                print("提取的元素:")
                for element in result.elements:
                    print(f"  - {element.content} ({element.element_type.value})")
                
                if result.reasoning_path:
                    print("推理路径:")
                    for path in result.reasoning_path[:5]:
                        print(f"  - {path}")
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"处理过程中出错: {e}")


def main():
    """主函数"""
    print("EASY KGQA Framework - 简化版知识图谱问答框架")
    print("专为教学设计，去除了复杂的相似度匹配等功能")
    print("保留了KGQA的核心：文本处理 + 实体识别 + 知识图谱推理")
    
    try:
        # 运行各种演示
        demo_basic_usage()
        demo_simple_qa() 
        demo_system_status()
        
        # 询问是否进入交互模式
        choice = input("\n是否进入交互模式? (y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_mode()
            
    except Exception as e:
        print(f"程序运行出错: {e}")
        print("请检查Neo4j和实体识别服务是否正常运行")


if __name__ == "__main__":
    main()