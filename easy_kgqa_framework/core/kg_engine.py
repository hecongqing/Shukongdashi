"""
简化版知识图谱引擎
负责与Neo4j数据库的交互，执行基本的图谱查询
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from ..models.entities import KnowledgeGraphNode, FaultElement, FaultType


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
        
        # 关系类型映射
        self.relation_types = {
            'CX': '操作导致现象',
            'XY': '现象导致原因',
            'XX': '现象关联现象',
            'XB': '现象关联部位',
            'XJ': '现象关联报警'
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
        根据内容查找节点
        
        Args:
            content: 节点内容
            node_type: 节点类型（可选）
            
        Returns:
            匹配的节点列表
        """
        try:
            with self.driver.session() as session:
                if node_type:
                    query = """
                    MATCH (n)
                    WHERE $node_type IN labels(n) AND n.name CONTAINS $content
                    RETURN n.name as name, labels(n) as labels, properties(n) as properties
                    LIMIT 10
                    """
                    result = session.run(query, content=content, node_type=node_type)
                else:
                    query = """
                    MATCH (n)
                    WHERE n.name CONTAINS $content
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
        查找相关节点
        
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
                    query = f"""
                    MATCH path = (start {{name: $node_name}})-[r:{relation_type}*1..{max_depth}]->(end)
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
    
    def simple_qa(self, question: str) -> List[Dict]:
        """
        简单问答查询
        
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
                WHERE n.name CONTAINS $keyword OR n.content CONTAINS $keyword
                RETURN n.name as name, labels(n) as labels, n.content as content
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
                                "content": record.get("content", "")
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
                
                # 获取标签类型
                label_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in label_result]
                
                return {
                    "node_count": node_count,
                    "relation_count": rel_count,
                    "label_count": len(labels),
                    "labels": labels
                }
        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return {
                "node_count": 0,
                "relation_count": 0,
                "label_count": 0,
                "labels": []
            }