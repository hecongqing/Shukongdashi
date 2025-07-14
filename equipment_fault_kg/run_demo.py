#!/usr/bin/env python3
"""
装备制造故障知识图谱 - 快速启动演示

这个脚本提供了一个完整的演示，展示知识图谱构建的各个步骤
"""

import os
import sys
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.append('src')

def create_sample_data():
    """创建示例数据"""
    sample_data = [
        {
            "id": "001",
            "title": "数控车床主轴异常振动故障诊断",
            "content": "某工厂数控车床在加工过程中出现主轴异常振动，经检查发现主轴轴承磨损严重，更换轴承后故障排除。",
            "source": "示例数据",
            "url": ""
        },
        {
            "id": "002", 
            "title": "加工中心伺服电机过热故障",
            "content": "加工中心X轴伺服电机在运行过程中出现过热现象，检查发现电机散热风扇故障，更换风扇后恢复正常。",
            "source": "示例数据",
            "url": ""
        },
        {
            "id": "003",
            "title": "铣床进给系统精度下降问题", 
            "content": "铣床进给系统精度下降，经检查发现滚珠丝杠磨损，更换丝杠并重新调整后精度恢复。",
            "source": "示例数据",
            "url": ""
        }
    ]
    
    # 保存到data/raw目录
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/sample_fault_cases.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("✓ 示例数据已创建")
    return sample_data

def run_data_processing():
    """运行数据处理"""
    try:
        from data_collection import DataProcessor
        import yaml
        
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 加载示例数据
        with open('data/raw/sample_fault_cases.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 处理数据
        processor = DataProcessor(config['entity_extraction'])
        processed_data = processor.process_batch(raw_data)
        
        # 保存处理后的数据
        os.makedirs('data/processed', exist_ok=True)
        processor.save_processed_data(processed_data, 'data/processed/processed_fault_cases.json')
        
        print("✓ 数据处理完成")
        return processed_data
        
    except Exception as e:
        print(f"✗ 数据处理失败: {e}")
        return None

def run_entity_extraction(processed_data):
    """运行实体抽取"""
    try:
        from entity_extraction import NERModel
        import yaml
        
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 初始化NER模型
        ner_model = NERModel(config['entity_extraction'])
        
        # 抽取实体
        all_entities = []
        for item in processed_data:
            text = f"{item['title']} {item['content']}"
            entities = ner_model.predict(text)
            all_entities.extend(entities)
        
        # 保存实体
        os.makedirs('data/annotated', exist_ok=True)
        with open('data/annotated/extracted_entities.json', 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 实体抽取完成，共抽取 {len(all_entities)} 个实体")
        return all_entities
        
    except Exception as e:
        print(f"✗ 实体抽取失败: {e}")
        return []

def run_knowledge_graph_construction(entities):
    """运行知识图谱构建"""
    try:
        from neo4j_qa import GraphManager
        import yaml
        
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 初始化图管理器
        graph_manager = GraphManager(config['database']['neo4j'])
        
        # 构建知识图谱
        graph_manager.build_knowledge_graph(entities, [])
        
        # 获取统计信息
        stats = graph_manager.get_statistics()
        
        print("✓ 知识图谱构建完成")
        print(f"  节点统计: {stats.get('nodes', {})}")
        print(f"  关系统计: {stats.get('relations', {})}")
        
        return graph_manager
        
    except Exception as e:
        print(f"✗ 知识图谱构建失败: {e}")
        print("  注意：需要先启动Neo4j数据库")
        return None

def run_qa_demo(graph_manager):
    """运行问答演示"""
    if not graph_manager:
        print("✗ 无法运行问答演示，图管理器未初始化")
        return
    
    print("\n=== 问答演示 ===")
    print("您可以询问以下类型的问题：")
    print("- 某装备的故障信息")
    print("- 某故障的原因") 
    print("- 某故障的解决方案")
    print("输入 'quit' 退出")
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                break
            
            if not question:
                continue
            
            # 简单的问答逻辑
            if "故障" in question and "装备" in question:
                equipment_name = question.split("装备")[1].split("故障")[0] if "装备" in question and "故障" in question else ""
                if equipment_name:
                    results = graph_manager.query_equipment_faults(equipment_name)
                    if results:
                        print(f"\n装备 {equipment_name} 的故障信息：")
                        for result in results:
                            print(f"- 故障：{result['fault']}")
                            print(f"  描述：{result['description']}")
                    else:
                        print(f"未找到装备 {equipment_name} 的故障信息")
            
            elif "原因" in question:
                fault_name = question.split("故障")[1].split("原因")[0] if "故障" in question and "原因" in question else ""
                if fault_name:
                    results = graph_manager.query_fault_causes(fault_name)
                    if results:
                        print(f"\n故障 {fault_name} 的可能原因：")
                        for result in results:
                            print(f"- {result['cause']}")
                            print(f"  描述：{result['description']}")
                    else:
                        print(f"未找到故障 {fault_name} 的原因信息")
            
            else:
                print("抱歉，我无法理解您的问题。请尝试询问装备故障、故障原因或解决方案相关的问题。")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    """主函数"""
    print("=== 装备制造故障知识图谱演示 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例数据...")
    create_sample_data()
    
    # 2. 数据处理
    print("\n2. 数据处理...")
    processed_data = run_data_processing()
    if not processed_data:
        print("数据处理失败，退出演示")
        return
    
    # 3. 实体抽取
    print("\n3. 实体抽取...")
    entities = run_entity_extraction(processed_data)
    
    # 4. 知识图谱构建
    print("\n4. 知识图谱构建...")
    graph_manager = run_knowledge_graph_construction(entities)
    
    # 5. 问答演示
    if graph_manager:
        run_qa_demo(graph_manager)
    
    print("\n=== 演示完成 ===")
    print("\n项目文件说明：")
    print("- src/: 源代码目录")
    print("- config/: 配置文件目录")
    print("- data/: 数据目录")
    print("- notebooks/: Jupyter notebooks")
    print("- models/: 模型文件目录")
    print("\n使用说明：")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 配置Neo4j数据库")
    print("3. 运行主程序: python src/main.py")
    print("4. 查看详细文档: README.md")

if __name__ == "__main__":
    main()