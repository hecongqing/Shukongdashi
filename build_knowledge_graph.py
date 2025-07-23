"""
改进的知识图谱构建脚本
适配用户的实体类型和关系类型
"""

import json
import yaml
import logging
from pathlib import Path
from easy_kgqa_framework.utils.graph_manager import GraphManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            'database': {
                'neo4j': {
                    'uri': 'bolt://localhost:50002',
                    'username': 'neo4j',
                    'password': 'password'
                }
            }
        }

def load_data(file_path: str):
    """加载JSON数据文件"""
    logger.info(f"加载数据文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"第 {line_num} 行JSON解析错误: {e}")
                        continue
    except FileNotFoundError:
        logger.error(f"数据文件 {file_path} 不存在")
        return

def classify_entity_type(entity_name: str, relation: str = "") -> str:
    """
    根据实体名称和关系对实体进行分类
    你可以根据自己的业务逻辑调整这个函数
    """
    # 基于关键词的简单分类规则
    if any(keyword in entity_name for keyword in ['电机', '轴承', '齿轮', '泵', '阀门', '传感器']):
        return "部件单元"
    elif any(keyword in entity_name for keyword in ['故障', '损坏', '失效', '断裂', '磨损']):
        return "故障状态"
    elif any(keyword in entity_name for keyword in ['检测', '测试', '仪器', '工具']):
        return "检测工具"
    elif any(keyword in entity_name for keyword in ['性能', '效率', '温度', '压力', '转速']):
        return "性能表征"
    elif "部件故障" in relation:
        return "部件单元" if "主体" not in entity_name else "主体"
    elif "性能故障" in relation:
        return "性能表征" if "表征" in entity_name else "客体"
    else:
        # 默认分类
        return "主体"

def extract_entities_relations(data_iter):
    """从数据中提取实体和关系"""
    entities_dict = {}
    relations = []
    
    for item in data_iter:
        spo_list = item.get('spo_list', [])
        logger.debug(f"处理条目，包含 {len(spo_list)} 个SPO三元组")
        
        for spo in spo_list:
            h = spo['h']
            t = spo['t'] 
            rel = spo['relation']
            
            # 分类实体类型
            h_type = classify_entity_type(h['name'], rel)
            t_type = classify_entity_type(t['name'], rel)
            
            # 添加头实体
            if h['name'] not in entities_dict:
                entities_dict[h['name']] = {
                    "type": h_type,
                    "text": h['name'],
                    "description": f"实体类型: {h_type}"
                }
            
            # 添加尾实体
            if t['name'] not in entities_dict:
                entities_dict[t['name']] = {
                    "type": t_type,
                    "text": t['name'], 
                    "description": f"实体类型: {t_type}"
                }
            
            # 添加关系
            relations.append({
                "head": h['name'],
                "head_type": h_type,
                "tail": t['name'],
                "tail_type": t_type,
                "relation": rel
            })
    
    logger.info(f"提取到 {len(entities_dict)} 个唯一实体，{len(relations)} 个关系")
    return list(entities_dict.values()), relations

def main():
    """主函数"""
    try:
        # 1. 加载配置
        config = load_config()
        
        # 2. 初始化图谱管理器
        graph_manager = GraphManager(config['database']['neo4j'])
        
        # 3. 测试连接
        if not graph_manager.test_connection():
            logger.error("无法连接到Neo4j数据库，请检查配置和数据库状态")
            return
        
        logger.info("成功连接到Neo4j数据库")
        
        # 4. 处理数据文件
        data_files = [
            "data/train.json",
            # 可以添加更多数据文件
        ]
        
        all_entities = []
        all_relations = []
        
        for file_path in data_files:
            if Path(file_path).exists():
                logger.info(f"处理数据文件: {file_path}")
                data_iter = load_data(file_path)
                entities, relations = extract_entities_relations(data_iter)
                all_entities.extend(entities)
                all_relations.extend(relations)
            else:
                logger.warning(f"数据文件不存在: {file_path}")
        
        # 5. 去重实体（保留第一个出现的）
        unique_entities = {}
        for entity in all_entities:
            if entity['text'] not in unique_entities:
                unique_entities[entity['text']] = entity
        
        final_entities = list(unique_entities.values())
        logger.info(f"去重后实体数量: {len(final_entities)}")
        
        # 6. 构建知识图谱
        if final_entities and all_relations:
            logger.info("开始构建知识图谱...")
            success = graph_manager.build_knowledge_graph(final_entities, all_relations)
            
            if success:
                # 7. 获取统计信息
                stats = graph_manager.get_statistics()
                logger.info("✓ 知识图谱构建完成")
                logger.info(f"  节点统计: {stats.get('nodes', {})}")
                logger.info(f"  关系统计: {stats.get('relations', {})}")
                logger.info(f"  总节点数: {stats.get('total_nodes', 0)}")
                logger.info(f"  总关系数: {stats.get('total_relations', 0)}")
            else:
                logger.error("知识图谱构建失败")
        else:
            logger.warning("没有找到有效的实体或关系数据")
        
        # 8. 关闭连接
        graph_manager.close()
        
    except Exception as e:
        logger.error(f"构建过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()