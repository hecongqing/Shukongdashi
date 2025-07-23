"""
简化版知识图谱引擎
负责与Neo4j数据库的交互，执行基本的图谱查询
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from ..models.entities import KnowledgeGraphNode, FaultElement, FaultType, EntityType, RelationType


class KnowledgeGraphEngine:
    """简化版知识图谱引擎"""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        初始化知识图谱引擎
        
        Args:
            uri: Neo4j数据库URI
            username: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # 关系类型映射（保持原有兼容性，添加新的关系类型）
        self.relation_types = {
            'CX': '操作导致现象',
            'XY': '现象导致原因',
            'XX': '现象关联现象',
            'XB': '现象关联部位',
            'XJ': '现象关联报警',
            # 新增关系类型
            '部件故障': '部件故障',
            '性能故障': '性能故障',
            '检测工具': '检测工具',
            '组成': '组成关系'
        }
        
        # 实体类型映射
        self.entity_types = {
            '主体': 'Subject',
            '客体': 'Object',
            '部件单元': 'ComponentUnit',
            '故障状态': 'FaultState',
            '性能表征': 'PerformanceFeature',
            '检测工具': 'DetectionTool'
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
        except Exception:
            return False
    
    def find_nodes_by_content(self, content: str, node_type: str = None) -> List[KnowledgeGraphNode]:
        """
        根据内容查找节点，支持新的实体类型
        
        Args:
            content: 节点内容
            node_type: 节点类型（可选）
            
        Returns:
            匹配的节点列表
        """
        try:
            with self.driver.session() as session:
                if node_type:
                    # 将中文实体类型转换为英文标签
                    english_type = self.entity_types.get(node_type, node_type)
                    query = """
                    MATCH (n)
                    WHERE $node_type IN labels(n) AND (n.name CONTAINS $content OR n.text CONTAINS $content)
                    RETURN n.name as name, labels(n) as labels, properties(n) as properties
                    LIMIT 10
                    """
                    result = session.run(query, content=content, node_type=english_type)
                else:
                    query = """
                    MATCH (n)
                    WHERE n.name CONTAINS $content OR n.text CONTAINS $content
                    RETURN n.name as name, labels(n) as labels, properties(n) as properties
                    LIMIT 10
                    """
                    result = session.run(query, content=content)
                
                nodes = []
                for record in result:
                    node = KnowledgeGraphNode(
                        id=record["name"],
                        name=record["name"],
                        labels=record["labels"],
                        properties=record["properties"]
                    )
                    nodes.append(node)
                
                return nodes
        except Exception as e:
            print(f"查询节点时出错: {e}")
            return []
    
    def find_related_nodes(self, node_name: str, relation_type: str = None, max_depth: int = 2) -> List[Dict]:
        """
        查找相关节点，支持新的关系类型
        
        Args:
            node_name: 起始节点名称
            relation_type: 关系类型（可选）
            max_depth: 最大搜索深度
            
        Returns:
            相关节点和路径信息
        """
        try:
            with self.driver.session() as session:
                if relation_type:
                    # 规范化关系类型
                    normalized_relation = relation_type.replace(' ', '_').replace('-', '_')
                    query = f"""
                    MATCH path = (start {{name: $node_name}})-[r:{normalized_relation}*1..{max_depth}]->(end)
                    RETURN path, start, end, r
                    LIMIT 20
                    """
                else:
                    query = f"""
                    MATCH path = (start {{name: $node_name}})-[r*1..{max_depth}]->(end)
                    RETURN path, start, end, r
                    LIMIT 20
                    """
                
                result = session.run(query, node_name=node_name)
                
                results = []
                for record in result:
                    path_info = {
                        "start_node": dict(record["start"]),
                        "end_node": dict(record["end"]),
                        "relations": [dict(rel) for rel in record["r"]],
                        "path_length": len(record["r"])
                    }
                    results.append(path_info)
                
                return results
        except Exception as e:
            print(f"查询相关节点时出错: {e}")
            return []
    
    def query_by_fault_elements(self, elements: List[FaultElement]) -> Dict[str, Any]:
        """
        根据故障元素进行查询
        
        Args:
            elements: 故障元素列表
            
        Returns:
            查询结果
        """
        results = {
            "nodes": [],
            "relations": [],
            "reasoning_paths": []
        }
        
        # 为每个元素查找相关节点
        for element in elements:
            # 查找匹配的节点
            nodes = self.find_nodes_by_content(element.content)
            results["nodes"].extend(nodes)
            
            # 查找相关节点
            for node in nodes:
                related = self.find_related_nodes(node.name)
                results["relations"].extend(related)
                
                # 构建推理路径
                for relation in related:
                    path = f"{relation['start_node'].get('name', '')} -> {relation['end_node'].get('name', '')}"
                    results["reasoning_paths"].append(path)
        
        return results
    
    def query_by_entity_type(self, entity_type: str, limit: int = 10) -> List[Dict]:
        """
        根据实体类型查询节点
        
        Args:
            entity_type: 实体类型（中文或英文）
            limit: 限制返回数量
            
        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                # 转换实体类型
                english_type = self.entity_types.get(entity_type, entity_type)
                
                query = f"""
                MATCH (n:{english_type})
                RETURN n.name as name, labels(n) as labels, properties(n) as properties
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                
                results = []
                for record in result:
                    results.append({
                        "name": record["name"],
                        "labels": record["labels"],
                        "properties": record["properties"]
                    })
                
                return results
        except Exception as e:
            print(f"按实体类型查询时出错: {e}")
            return []
    
    def query_by_relation_type(self, relation_type: str, limit: int = 10) -> List[Dict]:
        """
        根据关系类型查询关系
        
        Args:
            relation_type: 关系类型
            limit: 限制返回数量
            
        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                # 规范化关系类型
                normalized_relation = relation_type.replace(' ', '_').replace('-', '_')
                
                query = f"""
                MATCH (start)-[r:{normalized_relation}]->(end)
                RETURN start.name as start_name, end.name as end_name, 
                       type(r) as relation_type, properties(r) as properties,
                       labels(start) as start_labels, labels(end) as end_labels
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                
                results = []
                for record in result:
                    results.append({
                        "start_node": {
                            "name": record["start_name"],
                            "labels": record["start_labels"]
                        },
                        "end_node": {
                            "name": record["end_name"],
                            "labels": record["end_labels"]
                        },
                        "relation": {
                            "type": record["relation_type"],
                            "properties": record["properties"]
                        }
                    })
                
                return results
        except Exception as e:
            print(f"按关系类型查询时出错: {e}")
            return []
    
    def simple_qa(self, question: str) -> List[Dict]:
        """
        简单问答查询，支持新的实体和关系类型
        
        Args:
            question: 用户问题
            
        Returns:
            查询结果
        """
        try:
            with self.driver.session() as session:
                # 简单的模糊匹配查询
                query = """
                MATCH (n)
                WHERE n.name CONTAINS $keyword OR n.text CONTAINS $keyword OR n.description CONTAINS $keyword
                RETURN n.name as name, labels(n) as labels, 
                       coalesce(n.text, n.name) as content,
                       properties(n) as properties
                LIMIT 10
                """
                
                # 提取问题中的关键词（简单分词）
                import jieba
                keywords = jieba.lcut(question)
                
                all_results = []
                for keyword in keywords:
                    if len(keyword) > 1:  # 过滤单字
                        result = session.run(query, keyword=keyword)
                        for record in result:
                            all_results.append({
                                "name": record["name"],
                                "labels": record["labels"],
                                "content": record["content"],
                                "properties": record["properties"]
                            })
                
                return all_results
        except Exception as e:
            print(f"简单问答查询时出错: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取知识图谱统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.driver.session() as session:
                # 获取节点总数
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_count_result.single()["count"]
                
                # 获取关系总数
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_count_result.single()["count"]
                
                # 获取标签类型统计
                label_stats_result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                """)
                label_stats = {record["label"]: record["count"] for record in label_stats_result}
                
                # 获取关系类型统计
                rel_stats_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                """)
                rel_stats = {record["relation_type"]: record["count"] for record in rel_stats_result}
                
                return {
                    "node_count": node_count,
                    "relation_count": rel_count,
                    "nodes": label_stats,
                    "relations": rel_stats
                }
        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return {
                "node_count": 0,
                "relation_count": 0,
                "nodes": {},
                "relations": {}
            }