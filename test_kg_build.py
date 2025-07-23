"""
知识图谱构建测试脚本
"""

from easy_kgqa_framework.utils.graph_manager import GraphManager
from easy_kgqa_framework.core.kg_engine import KnowledgeGraphEngine
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_graph_manager():
    """测试GraphManager"""
    logger.info("测试GraphManager...")
    
    # 数据库配置
    config = {
        'uri': 'bolt://localhost:50002',
        'username': 'neo4j',
        'password': 'password'
    }
    
    # 创建管理器
    manager = GraphManager(config)
    
    # 测试连接
    if not manager.test_connection():
        logger.error("无法连接到Neo4j数据库")
        return False
    
    logger.info("数据库连接成功")
    
    # 测试实体
    test_entities = [
        {
            "type": "部件单元",
            "text": "主轴电机",
            "description": "机床主轴驱动电机"
        },
        {
            "type": "故障状态", 
            "text": "轴承故障",
            "description": "轴承磨损或损坏"
        },
        {
            "type": "检测工具",
            "text": "振动检测仪",
            "description": "用于检测机械振动的设备"
        }
    ]
    
    # 测试关系
    test_relations = [
        {
            "head": "主轴电机",
            "head_type": "部件单元",
            "tail": "轴承故障",
            "tail_type": "故障状态", 
            "relation": "部件故障"
        },
        {
            "head": "振动检测仪",
            "head_type": "检测工具",
            "tail": "轴承故障",
            "tail_type": "故障状态",
            "relation": "检测工具"
        }
    ]
    
    # 构建知识图谱
    logger.info("开始构建测试知识图谱...")
    success = manager.build_knowledge_graph(test_entities, test_relations)
    
    if success:
        logger.info("测试知识图谱构建成功")
        
        # 获取统计信息
        stats = manager.get_statistics()
        logger.info(f"统计信息: {stats}")
        
        # 测试查询
        logger.info("测试查询功能...")
        query_results = manager.query_by_entity_name("主轴电机")
        logger.info(f"查询结果: {len(query_results)} 条")
        
    else:
        logger.error("测试知识图谱构建失败")
    
    # 关闭连接
    manager.close()
    return success

def test_kg_engine():
    """测试KnowledgeGraphEngine"""
    logger.info("测试KnowledgeGraphEngine...")
    
    # 创建引擎
    engine = KnowledgeGraphEngine(
        uri='bolt://localhost:50002',
        username='neo4j',
        password='password'
    )
    
    # 测试连接
    if not engine.test_connection():
        logger.error("引擎无法连接到数据库")
        return False
    
    # 测试查询功能
    logger.info("测试按实体类型查询...")
    nodes = engine.query_by_entity_type("部件单元")
    logger.info(f"找到 {len(nodes)} 个部件单元实体")
    
    logger.info("测试按关系类型查询...")
    relations = engine.query_by_relation_type("部件故障")
    logger.info(f"找到 {len(relations)} 个部件故障关系")
    
    logger.info("测试简单问答...")
    qa_results = engine.simple_qa("主轴电机故障")
    logger.info(f"问答查询返回 {len(qa_results)} 个结果")
    
    # 获取统计信息
    stats = engine.get_statistics()
    logger.info(f"知识图谱统计: {stats}")
    
    # 关闭连接
    engine.close()
    return True

def main():
    """主函数"""
    logger.info("开始知识图谱构建测试...")
    
    try:
        # 测试GraphManager
        success1 = test_graph_manager()
        
        # 测试KnowledgeGraphEngine
        success2 = test_kg_engine()
        
        if success1 and success2:
            logger.info("✓ 所有测试通过")
        else:
            logger.error("✗ 部分测试失败")
            
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()