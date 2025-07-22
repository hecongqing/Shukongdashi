"""
KGQA框架基本功能测试
"""

import sys
import logging
from kgqa_framework.utils.text_processor import TextProcessor
from kgqa_framework.models.entities import FaultType, FaultElement, EquipmentInfo, UserQuery

def test_text_processor():
    """测试文本处理器"""
    print("=" * 50)
    print("测试文本处理器")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # 测试文本
    test_text = "自动换刀时刀链运转不到位，刀库停止运转，机床出现ALM401报警"
    
    print(f"原始文本: {test_text}")
    
    # 测试分句
    sentences = processor.split_sentences(test_text)
    print(f"\n分句结果: {sentences}")
    
    # 测试分词
    words = processor.segment_words(test_text)
    print(f"\n分词结果: {words[:10]}...")  # 只显示前10个
    
    # 测试故障元素提取
    elements = processor.extract_fault_elements(test_text)
    print(f"\n故障元素提取结果:")
    for element in elements:
        print(f"  - {element.element_type.value}: {element.content} (置信度: {element.confidence})")
    
    # 测试关键词提取
    keywords = processor.extract_keywords(test_text, top_k=5)
    print(f"\n关键词提取结果:")
    for keyword, weight in keywords:
        print(f"  - {keyword}: {weight:.3f}")
    
    # 测试相似度计算
    text1 = "主轴运转异常"
    text2 = "主轴运行不正常"
    similarity = processor.calculate_text_similarity(text1, text2)
    print(f"\n文本相似度:")
    print(f"  '{text1}' vs '{text2}': {similarity:.3f}")
    
    return True

def test_entities():
    """测试实体模型"""
    print("\n" + "=" * 50)
    print("测试实体模型")
    print("=" * 50)
    
    # 测试设备信息
    equipment = EquipmentInfo(
        brand="发那科",
        model="MATE-TD", 
        error_code="ALM401"
    )
    print(f"设备信息: {equipment.to_dict()}")
    
    # 测试故障元素
    element = FaultElement(
        content="刀库停止运转",
        element_type=FaultType.PHENOMENON,
        confidence=0.9,
        position=10
    )
    print(f"故障元素: {element.to_dict()}")
    
    # 测试用户查询
    query = UserQuery(
        equipment_info=equipment,
        fault_description="自动换刀时刀链运转不到位",
        related_phenomena=["机床报警", "刀链卡顿"],
        user_feedback=None
    )
    print(f"用户查询: {query.to_dict()}")
    
    return True

def test_mock_analysis():
    """模拟故障分析流程（不依赖外部数据库）"""
    print("\n" + "=" * 50)
    print("模拟故障分析流程")
    print("=" * 50)
    
    # 初始化文本处理器
    processor = TextProcessor()
    
    # 测试案例
    test_cases = [
        {
            "description": "主轴启动时发生异常振动，温度快速升高",
            "brand": "西门子",
            "error_code": None
        },
        {
            "description": "Y轴伺服电机运行时出现异响，位置精度下降",
            "brand": "发那科", 
            "error_code": "ALM502"
        },
        {
            "description": "液压系统压力不稳定，刀库换刀动作缓慢",
            "brand": "海德汉",
            "error_code": "HYD301"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"故障描述: {case['description']}")
        print(f"设备品牌: {case['brand']}")
        print(f"故障代码: {case['error_code']}")
        
        # 提取故障元素
        elements = processor.extract_fault_elements(case['description'])
        print(f"\n提取的故障元素:")
        for element in elements:
            print(f"  - {element.element_type.value}: {element.content}")
        
        # 提取关键词
        keywords = processor.extract_keywords(case['description'], top_k=3)
        print(f"\n关键词:")
        for keyword, weight in keywords:
            print(f"  - {keyword}: {weight:.3f}")
        
        # 模拟推理结果
        mock_causes = []
        mock_solutions = []
        
        # 基于故障元素模拟推理
        for element in elements:
            if element.element_type == FaultType.PHENOMENON:
                if "振动" in element.content:
                    mock_causes.append("轴承磨损")
                    mock_solutions.append("检查轴承状态")
                elif "温度" in element.content:
                    mock_causes.append("冷却系统故障")
                    mock_solutions.append("检查冷却系统")
                elif "异响" in element.content:
                    mock_causes.append("机械松动")
                    mock_solutions.append("检查机械连接")
                elif "压力" in element.content:
                    mock_causes.append("液压系统故障")
                    mock_solutions.append("检查液压油压")
        
        print(f"\n模拟推理结果:")
        print(f"可能原因: {mock_causes}")
        print(f"解决方案: {mock_solutions}")
    
    return True

def test_integration():
    """集成测试（检查各组件是否能正常协作）"""
    print("\n" + "=" * 50)
    print("集成测试")
    print("=" * 50)
    
    try:
        # 测试导入
        from kgqa_framework import FaultAnalyzer
        from kgqa_framework.config import current_config
        
        print("✓ 框架导入成功")
        
        # 测试配置
        config_dict = current_config.to_dict()
        print(f"✓ 配置加载成功，包含 {len(config_dict)} 个配置项")
        
        # 创建必要目录
        current_config.create_directories()
        print("✓ 目录创建成功")
        
        print("\n注意: 完整的故障分析器需要Neo4j数据库连接")
        print("如果需要测试完整功能，请确保:")
        print("1. Neo4j数据库正在运行")
        print("2. 已导入知识图谱数据")
        print("3. 正确配置数据库连接参数")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("KGQA框架基本功能测试")
    print("开始测试...")
    
    # 执行测试
    tests = [
        ("文本处理器", test_text_processor),
        ("实体模型", test_entities),
        ("模拟分析", test_mock_analysis),
        ("集成测试", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n正在测试: {name}")
            result = test_func()
            results.append((name, result, None))
            print(f"✓ {name}测试完成")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name}测试失败: {e}")
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result, error in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
        if error:
            print(f"  错误: {error}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个测试通过, {failed} 个测试失败")
    
    if failed == 0:
        print("🎉 所有基本功能测试通过！")
        print("\n下一步:")
        print("1. 配置Neo4j数据库连接")
        print("2. 运行完整的演示程序: python main.py demo")
        print("3. 启动Django服务: python manage.py runserver")
    else:
        print("⚠️  部分测试失败，请检查相关组件")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)