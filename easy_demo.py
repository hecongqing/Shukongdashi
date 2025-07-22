#!/usr/bin/env python3
"""
简洁版KGQA框架示例程序 - 使用Neo4j和实体识别
Easy KGQA Framework Demo with Neo4j and NER
"""

from easy_kgqa_framework import EasyKGQA, SimpleEntityRecognizer


def demo_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("简洁版知识图谱问答系统演示 (Neo4j + 实体识别)")
    print("=" * 60)
    
    # 1. 初始化系统
    print("\n1. 初始化KGQA系统...")
    try:
        # 注意：需要Neo4j服务运行在本地
        kgqa = EasyKGQA()
        print("✓ Neo4j连接成功")
    except Exception as e:
        print(f"❌ Neo4j连接失败: {e}")
        print("请确保Neo4j服务正在运行 (bolt://localhost:7687)")
        return False
    
    # 2. 清空数据库（演示用）
    print("\n2. 清空旧数据...")
    kgqa.clear_database()
    
    # 3. 添加基础数据
    print("\n3. 添加基础知识...")
    
    # 添加实体
    kgqa.add_entity("数控机床", "设备", "用于精密加工的自动化机床")
    kgqa.add_entity("主轴", "部件", "机床的核心旋转部件")
    kgqa.add_entity("刀具", "工具", "用于切削加工的工具")
    kgqa.add_entity("电机", "部件", "提供动力的电动机")
    kgqa.add_entity("轴承", "部件", "支撑旋转部件的机械元件")
    
    # 添加关系
    kgqa.add_relation("数控机床", "包含", "主轴")
    kgqa.add_relation("数控机床", "使用", "刀具")
    kgqa.add_relation("主轴", "驱动", "刀具")
    kgqa.add_relation("电机", "驱动", "主轴")
    kgqa.add_relation("主轴", "包含", "轴承")
    
    # 添加故障案例
    kgqa.add_fault_case("数控机床", "主轴不转", "电机故障", "检查电机状态并更换")
    kgqa.add_fault_case("数控机床", "精度差", "刀具磨损", "更换刀具")
    kgqa.add_fault_case("数控机床", "异响", "轴承磨损", "更换轴承")
    
    print("✓ 基础知识添加完成")
    
    # 4. 演示实体识别
    print("\n4. 实体识别演示:")
    recognizer = SimpleEntityRecognizer()
    test_texts = [
        "数控机床主轴不转了",
        "电机出现异响需要检查",
        "刀具磨损导致精度差"
    ]
    
    for text in test_texts:
        entities = recognizer.recognize(text)
        entities_by_type = recognizer.recognize_by_type(text)
        print(f"文本: {text}")
        print(f"  实体: {entities}")
        print(f"  分类: {entities_by_type}")
        print()
    
    # 5. 显示统计信息
    print("5. 知识库统计:")
    stats = kgqa.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 6. 问答演示
    print("\n6. 问答演示:")
    
    questions = [
        "什么是数控机床？",
        "主轴不转怎么办？",
        "数控机床包含什么部件？",
        "电机的作用是什么？",
        "轴承异响的原因"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = kgqa.answer_question(question)
        print(f"识别实体: {result['entities_found']}")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")
        if result['sources']:
            print(f"信息来源: {result['sources'][0]['type']}")
    
    return True


def demo_interactive():
    """交互式问答演示"""
    print("\n" + "=" * 60)
    print("交互式问答模式")
    print("输入 'quit' 退出")
    print("=" * 60)
    
    try:
        kgqa = EasyKGQA()
        print("✓ 系统准备就绪")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    while True:
        question = input("\n请输入问题: ").strip()
        
        if question.lower() in ['quit', '退出', 'q']:
            break
            
        if not question:
            continue
            
        result = kgqa.answer_question(question)
        print(f"识别实体: {result['entities_found']}")
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")


if __name__ == "__main__":
    print("请确保Neo4j服务正在运行...")
    print("默认连接: bolt://localhost:7687 (neo4j/password)")
    print()
    
    # 运行基础演示
    if demo_basic_usage():
        print("\n" + "=" * 60)
        choice = input("是否进入交互式模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            demo_interactive()
    
    print("\n演示结束！")