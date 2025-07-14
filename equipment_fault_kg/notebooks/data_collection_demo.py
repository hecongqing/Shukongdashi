"""
装备制造故障知识图谱 - 数据采集演示

本脚本演示如何采集装备制造故障相关的数据
"""

import sys
import os
sys.path.append('../src')

import yaml
import pandas as pd
from data_collection import DataProcessor
from pathlib import Path


def main():
    """主函数"""
    print("=== 装备制造故障知识图谱 - 数据采集演示 ===\n")
    
    # 1. 加载配置
    print("1. 加载配置文件...")
    with open('../config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("配置加载成功\n")
    
    # 2. 创建示例数据
    print("2. 创建示例故障数据...")
    sample_fault_cases = [
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
        },
        {
            "id": "004",
            "title": "钻床液压系统压力不足故障",
            "content": "钻床液压系统压力不足，导致夹紧力不够，检查发现液压油泄漏，修复泄漏点后压力恢复正常。",
            "source": "示例数据",
            "url": ""
        },
        {
            "id": "005",
            "title": "磨床冷却系统故障",
            "content": "磨床冷却系统无法正常工作，工件温度过高，检查发现冷却泵故障，更换泵后系统恢复正常。",
            "source": "示例数据",
            "url": ""
        }
    ]
    
    print(f"创建了 {len(sample_fault_cases)} 条示例数据\n")
    
    # 3. 数据处理
    print("3. 处理数据...")
    processor = DataProcessor(config['entity_extraction'])
    processed_data = processor.process_batch(sample_fault_cases)
    print(f"处理完成，共 {len(processed_data)} 条数据\n")
    
    # 4. 查看处理结果
    print("4. 查看处理结果...")
    df_data = []
    for item in processed_data:
        df_data.append({
            'id': item['id'],
            'title': item['title'],
            'equipment_type': item['equipment_info'].get('equipment_type'),
            'manufacturer': item['equipment_info'].get('manufacturer'),
            'components': ','.join(item['equipment_info'].get('components', [])),
            'fault_type': item['fault_info'].get('fault_type'),
            'symptoms': ';'.join(item['fault_info'].get('symptoms', [])),
            'causes': ';'.join(item['fault_info'].get('causes', [])),
            'solutions': ';'.join(item['fault_info'].get('solutions', []))
        })
    
    df = pd.DataFrame(df_data)
    print("处理后的数据:")
    print(df.to_string(index=False))
    print()
    
    # 5. 生成统计信息
    print("5. 生成统计信息...")
    stats = processor.generate_statistics(processed_data)
    
    print("数据统计信息:")
    print(f"总案例数: {stats['total_cases']}")
    print(f"装备类型: {stats['equipment_types']}")
    print(f"故障类型: {stats['fault_types']}")
    print(f"制造商: {stats['manufacturers']}")
    print()
    
    # 6. 保存数据
    print("6. 保存数据...")
    processor.save_processed_data(processed_data, '../data/processed/sample_fault_cases.json')
    print("数据已保存到 ../data/processed/sample_fault_cases.json")
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    main()