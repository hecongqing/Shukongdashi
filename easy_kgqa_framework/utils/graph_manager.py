"""
知识图谱管理器
负责知识图谱的构建、更新和统计
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import logging
from ..models.entities import EntityType, RelationType


class GraphManager:
    """知识图谱管理器"""
    
    def __init__(self, config: Dict[str, str]):
        """
        初始化图谱管理器
        
        Args:
            config: 数据库配置字典，包含uri, username, password
        """
        self.uri = config.get('uri', 'bolt://localhost:50002')
        self.username = config.get('username', 'neo4j') 
        self.password = config.get('password', 'password')
        
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.username, self.password)
        )
        self.logger = logging.getLogger(__name__)
        
        # 实体类型映射
        self.entity_type_mapping = {
            "主体": "Subject",
            "客体": "Object", 
            "部件单元": "ComponentUnit",
            "故障状态": "FaultState",
            "性能表征": "PerformanceFeature",
            "检测工具": "DetectionTool",
            "Component": "ComponentUnit",  # 兼容旧格式
            "Fault": "FaultState",         # 兼容旧格式
            "Event": "Event"               # 兼容旧格式
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("数据库已清空")
        except Exception as e:
            self.logger.error(f"清空数据库失败: {e}")
    
    def create_entity(self, entity: Dict[str, Any]) -> bool:
        """
        创建实体节点
        
        Args:
            entity: 实体信息字典，包含text, type, description等
            
        Returns:
            创建是否成功
        """
        try:
            with self.driver.session() as session:
                # 获取标准化的实体类型
                entity_type = self.entity_type_mapping.get(
                    entity.get('type', ''), 
                    'Unknown'
                )
                
                # 创建节点
                query = f"""
                MERGE (n:{entity_type} {{name: $name}})
                SET n.text = $text,
                    n.description = $description,
                    n.type = $type
                RETURN n
                """
                
                result = session.run(query, 
                    name=entity['text'],
                    text=entity['text'],
                    description=entity.get('description', ''),
                    type=entity.get('type', '')
                )
                
                return result.single() is not None
                
        except Exception as e:
            self.logger.error(f"创建实体失败: {e}")
            return False
    
    def create_relation(self, relation: Dict[str, Any]) -> bool:
        """
        创建关系
        
        Args:
            relation: 关系信息字典，包含head, tail, relation, head_type, tail_type
            
        Returns:
            创建是否成功
        """
        try:
            with self.driver.session() as session:
                # 获取标准化的实体类型
                head_type = self.entity_type_mapping.get(
                    relation.get('head_type', ''), 
                    'Unknown'
                )
                tail_type = self.entity_type_mapping.get(
                    relation.get('tail_type', ''), 
                    'Unknown'
                )
                
                # 规范化关系类型
                relation_type = relation['relation'].replace(' ', '_').replace('-', '_')
                
                # 创建关系
                query = f"""
                MATCH (h:{head_type} {{name: $head_name}})
                MATCH (t:{tail_type} {{name: $tail_name}})
                MERGE (h)-[r:{relation_type}]->(t)
                SET r.relation_name = $relation_name
                RETURN r
                """
                
                result = session.run(query,
                    head_name=relation['head'],
                    tail_name=relation['tail'],
                    relation_name=relation['relation']
                )
                
                return result.single() is not None
                
        except Exception as e:
            self.logger.error(f"创建关系失败: {e}")
            return False
    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]], 
                            relations: List[Dict[str, Any]]) -> bool:
        """
        批量构建知识图谱
        
        Args:
            entities: 实体列表
            relations: 关系列表
            
        Returns:
            构建是否成功
        """
        self.logger.info(f"开始构建知识图谱，实体数量: {len(entities)}, 关系数量: {len(relations)}")
        
        # 创建实体
        entity_success_count = 0
        for entity in entities:
            if self.create_entity(entity):
                entity_success_count += 1
        
        self.logger.info(f"成功创建实体: {entity_success_count}/{len(entities)}")
        
        # 创建关系
        relation_success_count = 0
        for relation in relations:
            if self.create_relation(relation):
                relation_success_count += 1
        
        self.logger.info(f"成功创建关系: {relation_success_count}/{len(relations)}")
        
        return entity_success_count > 0 and relation_success_count > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识图谱统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.driver.session() as session:
                # 获取节点统计
                node_stats_query = """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                """
                node_result = session.run(node_stats_query)
                node_stats = {record["label"]: record["count"] for record in node_result}
                
                # 获取关系统计
                rel_stats_query = """
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                """
                rel_result = session.run(rel_stats_query)
                rel_stats = {record["relation_type"]: record["count"] for record in rel_result}
                
                # 获取总数
                total_nodes_result = session.run("MATCH (n) RETURN count(n) as total")
                total_nodes = total_nodes_result.single()["total"]
                
                total_relations_result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
                total_relations = total_relations_result.single()["total"]
                
                return {
                    "nodes": node_stats,
                    "relations": rel_stats,
                    "total_nodes": total_nodes,
                    "total_relations": total_relations
                }
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {
                "nodes": {},
                "relations": {},
                "total_nodes": 0,
                "total_relations": 0
            }
    
    def query_by_entity_name(self, entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        根据实体名称查询相关信息
        
        Args:
            entity_name: 实体名称
            limit: 结果限制数量
            
        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {name: $entity_name})-[r]-(m)
                RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels
                LIMIT $limit
                """
                
                result = session.run(query, entity_name=entity_name, limit=limit)
                
                results = []
                for record in result:
                    results.append({
                        "source_node": {
                            "name": record["n"]["name"],
                            "labels": record["n_labels"],
                            "properties": dict(record["n"])
                        },
                        "relation": {
                            "type": type(record["r"]).__name__,
                            "properties": dict(record["r"])
                        },
                        "target_node": {
                            "name": record["m"]["name"], 
                            "labels": record["m_labels"],
                            "properties": dict(record["m"])
                        }
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"查询实体失败: {e}")
            return []