#!/usr/bin/env python3
"""
简洁版KGQA框架示例程序
Easy KGQA Framework Demo
"""

from easy_kgqa_framework import EasyKGQA


def demo_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("简洁版知识图谱问答系统演示")
    print("=" * 50)
    
    # 1. 初始化系统
    print("\n1. 初始化KGQA系统...")
    kgqa = EasyKGQA("demo_kg.db")
    print("✓ 系统初始化完成")
    
    # 2. 添加基础数据
    print("\n2. 添加基础知识...")
    
    # 添加实体
    kgqa.add_entity("数控机床", "设备", "用于精密加工的自动化机床")
    kgqa.add_entity("主轴", "部件", "机床的核心旋转部件")
    kgqa.add_entity("刀具", "工具", "用于切削加工的工具")
    kgqa.add_entity("电机", "部件", "提供动力的电动机")
    
    # 添加关系
    kgqa.add_relation("数控机床", "包含", "主轴")
    kgqa.add_relation("数控机床", "使用", "刀具")
    kgqa.add_relation("主轴", "驱动", "刀具")
    kgqa.add_relation("电机", "驱动", "主轴")
    
    # 添加故障案例
    kgqa.add_fault_case(
        "数控机床", 
        "主轴不转", 
        "电机故障或电源问题", 
        "检查电源连接，检查电机状态，必要时更换电机"
    )
    kgqa.add_fault_case(
        "数控机床", 
        "加工精度差", 
        "刀具磨损或主轴松动", 
        "更换刀具，检查并紧固主轴"
    )
    kgqa.add_fault_case(
        "数控机床", 
        "异常噪音", 
        "轴承磨损或润滑不足", 
        "检查润滑系统，更换磨损的轴承"
    )
    
    print("✓ 基础知识添加完成")
    
    # 3. 显示统计信息
    print("\n3. 知识库统计:")
    stats = kgqa.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 4. 问答演示
    print("\n4. 问答演示:")
    
    questions = [
        "什么是数控机床？",
        "主轴不转怎么办？",
        "数控机床包含什么部件？",
        "加工精度差的原因是什么？",
        "电机的作用是什么？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = kgqa.answer_question(question)
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")
        if result['sources']:
            print(f"信息来源: {result['sources'][0]['type']}")


def demo_interactive():
    """交互式问答演示"""
    print("\n" + "=" * 50)
    print("交互式问答模式")
    print("输入 'quit' 退出")
    print("=" * 50)
    
    kgqa = EasyKGQA("demo_kg.db")
    
    while True:
        question = input("\n请输入问题: ").strip()
        
        if question.lower() in ['quit', '退出', 'q']:
            break
            
        if not question:
            continue
            
        result = kgqa.answer_question(question)
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")


if __name__ == "__main__":
    # 运行基础演示
    demo_basic_usage()
    
    # 运行交互式演示
    demo_interactive()
    
    print("\n演示结束！")