"""
KGQA框架主程序
演示如何使用基于知识图谱的故障诊断问答系统
"""

import logging
import sys
import json
from kgqa_framework import FaultAnalyzer
from kgqa_framework.config import current_config
from kgqa_framework.models.entities import EquipmentInfo, UserQuery


def setup_logging():
    """设置日志配置"""
    current_config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, current_config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(current_config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def demo_fault_analysis():
    """演示故障分析功能"""
    print("=" * 60)
    print("基于知识图谱的故障诊断系统演示")
    print("=" * 60)
    
    # 初始化故障分析器
    try:
        analyzer = FaultAnalyzer(
            neo4j_uri=current_config.NEO4J_URI,
            neo4j_username=current_config.NEO4J_USERNAME,
            neo4j_password=current_config.NEO4J_PASSWORD,
            case_database_path=current_config.CASE_DATABASE_PATH,
            vectorizer_path=current_config.VECTORIZER_PATH,
            stopwords_path=current_config.STOPWORDS_PATH,
            custom_dict_path=current_config.CUSTOM_DICT_PATH,
            enable_web_search=current_config.ENABLE_WEB_SEARCH
        )
        
        print("✓ 故障分析器初始化成功")
        
        # 检查系统状态
        status = analyzer.get_system_status()
        print(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"✗ 故障分析器初始化失败: {e}")
        return
    
    # 演示案例1：基本故障诊断
    print("\n" + "-" * 40)
    print("案例1: 刀库故障诊断")
    print("-" * 40)
    
    result1 = analyzer.analyze_fault(
        fault_description="自动换刀时刀链运转不到位，刀库停止运转",
        brand="发那科",
        model="MATE-TD",
        error_code="ALM401",
        related_phenomena=["机床自动报警", "刀链卡顿"]
    )
    
    print_diagnosis_result("刀库故障", result1)
    
    # 演示案例2：主轴故障诊断
    print("\n" + "-" * 40)
    print("案例2: 主轴故障诊断")
    print("-" * 40)
    
    result2 = analyzer.analyze_fault(
        fault_description="主轴运转时出现异常振动和噪音，温度升高",
        brand="西门子",
        model="840D",
        related_phenomena=["主轴温度报警", "振动超标"]
    )
    
    print_diagnosis_result("主轴故障", result2)
    
    # 演示案例3：电机故障诊断
    print("\n" + "-" * 40)
    print("案例3: 伺服电机故障诊断")
    print("-" * 40)
    
    result3 = analyzer.analyze_fault(
        fault_description="Y轴伺服电机启动后立即停止，显示过载报警",
        brand="发那科",
        error_code="ALM502",
        related_phenomena=["电机过热", "编码器报警"]
    )
    
    print_diagnosis_result("伺服电机故障", result3)
    
    # 演示用户反馈功能
    print("\n" + "-" * 40)
    print("用户反馈演示")
    print("-" * 40)
    
    # 创建用户查询对象
    user_query = UserQuery(
        equipment_info=EquipmentInfo(brand="发那科", model="MATE-TD", error_code="ALM401"),
        fault_description="自动换刀时刀链运转不到位，刀库停止运转",
        related_phenomena=["机床自动报警"],
        user_feedback=None
    )
    
    # 模拟用户选择解决方案并反馈
    chosen_solution = "检查刀库液压系统压力"
    effectiveness_score = 0.9
    
    analyzer.add_user_feedback(user_query, chosen_solution, effectiveness_score)
    print(f"✓ 用户反馈已记录: 解决方案 '{chosen_solution}' 有效性评分: {effectiveness_score}")
    
    # 导出知识库
    print("\n" + "-" * 40)
    print("导出知识库")
    print("-" * 40)
    
    try:
        export_path = "./exported_knowledge.json"
        analyzer.export_knowledge(export_path, format="json")
        print(f"✓ 知识库已导出到: {export_path}")
    except Exception as e:
        print(f"✗ 导出知识库失败: {e}")
    
    # 关闭分析器
    analyzer.close()
    print("\n✓ 演示完成")


def print_diagnosis_result(case_name: str, result):
    """打印诊断结果"""
    print(f"\n【{case_name}诊断结果】")
    print(f"置信度: {result.confidence:.2f}")
    
    print(f"\n可能原因 ({len(result.causes)}):")
    for i, cause in enumerate(result.causes, 1):
        print(f"  {i}. {cause}")
    
    print(f"\n解决方案 ({len(result.solutions)}):")
    for i, solution in enumerate(result.solutions, 1):
        print(f"  {i}. {solution}")
    
    if result.similar_cases:
        print(f"\n相似案例 ({len(result.similar_cases)}):")
        for i, case in enumerate(result.similar_cases, 1):
            print(f"  {i}. 相似度: {case.similarity:.2f} - {case.description[:50]}...")
    
    if result.recommendations:
        print(f"\n进一步检查建议:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print(f"\n推理路径:")
    for i, step in enumerate(result.reasoning_path, 1):
        print(f"  {i}. {step['step']}: {step['description']}")


def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("进入交互模式 - 输入 'quit' 退出")
    print("=" * 60)
    
    try:
        analyzer = FaultAnalyzer(
            neo4j_uri=current_config.NEO4J_URI,
            neo4j_username=current_config.NEO4J_USERNAME,
            neo4j_password=current_config.NEO4J_PASSWORD,
            enable_web_search=current_config.ENABLE_WEB_SEARCH
        )
        print("✓ 系统初始化成功")
        
        while True:
            print("\n" + "-" * 40)
            fault_description = input("请输入故障描述: ").strip()
            
            if fault_description.lower() in ['quit', 'exit', '退出']:
                break
            
            if not fault_description:
                print("故障描述不能为空")
                continue
            
            # 可选输入
            brand = input("设备品牌 (可选): ").strip() or None
            model = input("设备型号 (可选): ").strip() or None
            error_code = input("故障代码 (可选): ").strip() or None
            
            related_phenomena = []
            while True:
                phenomenon = input("相关现象 (回车结束): ").strip()
                if not phenomenon:
                    break
                related_phenomena.append(phenomenon)
            
            print("\n分析中...")
            result = analyzer.analyze_fault(
                fault_description=fault_description,
                brand=brand,
                model=model,
                error_code=error_code,
                related_phenomena=related_phenomena
            )
            
            print_diagnosis_result("故障", result)
            
        analyzer.close()
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"系统错误: {e}")


def main():
    """主函数"""
    setup_logging()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'interactive':
            interactive_mode()
        elif mode == 'demo':
            demo_fault_analysis()
        else:
            print("使用方法:")
            print("  python main.py demo        # 运行演示")
            print("  python main.py interactive # 交互模式")
    else:
        # 默认运行演示
        demo_fault_analysis()


if __name__ == "__main__":
    main()