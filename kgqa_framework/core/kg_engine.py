"""
知识图谱引擎
负责与Neo4j数据库的交互，执行图谱查询和推理
"""

from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import logging
from ..models.entities import (
    KnowledgeGraphNode, KnowledgeGraphRelation, 
    FaultElement, FaultType
)


class KnowledgeGraphEngine:
    """知识图谱引擎"""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        初始化知识图谱引擎
        
        Args:
            uri: Neo4j数据库URI
            username: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
        
        # 关系类型映射
        self.relation_types = {
            'CX': '操作导致现象',    # 操作 -> 现象
            'XY': '现象导致原因',    # 现象 -> 原因
            'XX': '现象关联现象',    # 现象 -> 现象
            'XB': '现象关联部位',    # 现象 -> 部位
            'XJ': '现象关联报警'     # 现象 -> 报警
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
    
    def find_nodes_by_content(self, content: str, node_types: List[str] = None) -> List[KnowledgeGraphNode]:
        """
        根据内容查找节点
        
        Args:
            content: 查找内容
            node_types: 节点类型列表
            
        Returns:
            匹配的节点列表
        """
        nodes = []
        
        try:
            with self.driver.session() as session:
                # 构建查询语句
                if node_types:
                    label_filter = " OR ".join([f"n:{label}" for label in node_types])
                    query = f"""
                    MATCH (n) 
                    WHERE ({label_filter}) AND n.title CONTAINS $content
                    RETURN n, labels(n) as labels
                    """
                else:
                    query = """
                    MATCH (n) 
                    WHERE n.title CONTAINS $content
                    RETURN n, labels(n) as labels
                    """
                
                result = session.run(query, content=content)
                
                for record in result:
                    node_data = record["n"]
                    labels = record["labels"]
                    
                    node = KnowledgeGraphNode(
                        id=str(node_data.element_id),
                        label=node_data.get("title", ""),
                        properties=dict(node_data),
                        node_type=labels[0] if labels else "Unknown"
                    )
                    nodes.append(node)
                    
        except Exception as e:
            self.logger.error(f"查找节点失败: {e}")
        
        return nodes
    
    def find_related_nodes(self, node_title: str, relation_types: List[str] = None, 
                          direction: str = "both") -> List[Tuple[KnowledgeGraphNode, str]]:
        """
        查找相关节点
        
        Args:
            node_title: 节点标题
            relation_types: 关系类型列表
            direction: 关系方向 ("outgoing", "incoming", "both")
            
        Returns:
            (相关节点, 关系类型) 元组列表
        """
        related_nodes = []
        
        try:
            with self.driver.session() as session:
                # 构建查询语句
                if direction == "outgoing":
                    relation_pattern = "-[r]->"
                elif direction == "incoming":
                    relation_pattern = "<-[r]-"
                else:
                    relation_pattern = "-[r]-"
                
                if relation_types:
                    relation_filter = "|".join(relation_types)
                    relation_pattern = f"-[r:{relation_filter}]->"
                
                query = f"""
                MATCH (n {{title: $title}}){relation_pattern}(m)
                RETURN m, type(r) as relation_type, labels(m) as labels
                """
                
                result = session.run(query, title=node_title)
                
                for record in result:
                    node_data = record["m"]
                    relation_type = record["relation_type"]
                    labels = record["labels"]
                    
                    node = KnowledgeGraphNode(
                        id=str(node_data.element_id),
                        label=node_data.get("title", ""),
                        properties=dict(node_data),
                        node_type=labels[0] if labels else "Unknown"
                    )
                    related_nodes.append((node, relation_type))
                    
        except Exception as e:
            self.logger.error(f"查找相关节点失败: {e}")
        
        return related_nodes
    
    def find_paths_between_nodes(self, start_title: str, end_title: str, 
                                max_depth: int = 3) -> List[List[Dict]]:
        """
        查找两个节点之间的路径
        
        Args:
            start_title: 起始节点标题
            end_title: 结束节点标题
            max_depth: 最大路径深度
            
        Returns:
            路径列表，每个路径是节点和关系的字典列表
        """
        paths = []
        
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH path = (start {{title: $start_title}})-[*1..{max_depth}]-(end {{title: $end_title}})
                RETURN path
                LIMIT 10
                """
                
                result = session.run(query, start_title=start_title, end_title=end_title)
                
                for record in result:
                    path_data = record["path"]
                    path_info = []
                    
                    # 解析路径中的节点和关系
                    nodes = path_data.nodes
                    relationships = path_data.relationships
                    
                    for i, node in enumerate(nodes):
                        path_info.append({
                            "type": "node",
                            "id": str(node.element_id),
                            "title": node.get("title", ""),
                            "properties": dict(node)
                        })
                        
                        if i < len(relationships):
                            rel = relationships[i]
                            path_info.append({
                                "type": "relationship",
                                "relation_type": rel.type,
                                "properties": dict(rel)
                            })
                    
                    paths.append(path_info)
                    
        except Exception as e:
            self.logger.error(f"查找路径失败: {e}")
        
        return paths
    
    def get_fault_causes_by_phenomena(self, phenomena: List[str]) -> List[Dict]:
        """
        根据故障现象查找可能的原因
        
        Args:
            phenomena: 故障现象列表
            
        Returns:
            故障原因和置信度列表
        """
        causes = []
        
        try:
            with self.driver.session() as session:
                for phenomenon in phenomena:
                    query = """
                    MATCH (p:Xianxiang {title: $phenomenon})-[:XY]->(y:Yuanyin)
                    RETURN y.title as cause, 1.0 as confidence
                    UNION
                    MATCH (p:Xianxiang {title: $phenomenon})-[:XX]->(x:Xianxiang)-[:XY]->(y:Yuanyin)
                    RETURN y.title as cause, 0.8 as confidence
                    """
                    
                    result = session.run(query, phenomenon=phenomenon)
                    
                    for record in result:
                        causes.append({
                            "cause": record["cause"],
                            "confidence": record["confidence"],
                            "related_phenomenon": phenomenon
                        })
                        
        except Exception as e:
            self.logger.error(f"查找故障原因失败: {e}")
        
        return causes
    
    def get_related_phenomena_by_operations(self, operations: List[str]) -> List[Dict]:
        """
        根据操作查找相关现象
        
        Args:
            operations: 操作列表
            
        Returns:
            相关现象列表
        """
        phenomena = []
        
        try:
            with self.driver.session() as session:
                for operation in operations:
                    query = """
                    MATCH (c:Caozuo)-[:CX]->(x:Xianxiang)
                    WHERE c.title CONTAINS $operation
                    RETURN x.title as phenomenon, 0.9 as confidence
                    """
                    
                    result = session.run(query, operation=operation)
                    
                    for record in result:
                        phenomena.append({
                            "phenomenon": record["phenomenon"],
                            "confidence": record["confidence"],
                            "related_operation": operation
                        })
                        
        except Exception as e:
            self.logger.error(f"查找相关现象失败: {e}")
        
        return phenomena
    
    def get_location_phenomena(self, locations: List[str]) -> List[Dict]:
        """
        根据故障部位查找常见现象
        
        Args:
            locations: 故障部位列表
            
        Returns:
            常见现象列表
        """
        phenomena = []
        
        try:
            with self.driver.session() as session:
                for location in locations:
                    query = """
                    MATCH (b:GuzhangBuwei)-[:XB]-(x:Xianxiang)
                    WHERE b.title CONTAINS $location
                    RETURN x.title as phenomenon, 0.8 as confidence
                    """
                    
                    result = session.run(query, location=location)
                    
                    for record in result:
                        phenomena.append({
                            "phenomenon": record["phenomenon"],
                            "confidence": record["confidence"],
                            "related_location": location
                        })
                        
        except Exception as e:
            self.logger.error(f"查找部位现象失败: {e}")
        
        return phenomena
    
    def get_alarm_phenomena(self, alarms: List[str]) -> List[Dict]:
        """
        根据报警信息查找相关现象
        
        Args:
            alarms: 报警信息列表
            
        Returns:
            相关现象列表
        """
        phenomena = []
        
        try:
            with self.driver.session() as session:
                for alarm in alarms:
                    query = """
                    MATCH (e:Errorid)-[:XJ]-(x:Xianxiang)
                    WHERE e.title CONTAINS $alarm
                    RETURN x.title as phenomenon, 0.9 as confidence
                    """
                    
                    result = session.run(query, alarm=alarm)
                    
                    for record in result:
                        phenomena.append({
                            "phenomenon": record["phenomenon"],
                            "confidence": record["confidence"],
                            "related_alarm": alarm
                        })
                        
        except Exception as e:
            self.logger.error(f"查找报警现象失败: {e}")
        
        return phenomena
    
    def execute_reasoning_chain(self, fault_elements: List[FaultElement]) -> Dict[str, Any]:
        """
        执行推理链
        
        Args:
            fault_elements: 故障元素列表
            
        Returns:
            推理结果
        """
        reasoning_result = {
            "causes": [],
            "related_phenomena": [],
            "reasoning_paths": [],
            "confidence_scores": {}
        }
        
        # 按类型分组故障元素
        operations = [elem.content for elem in fault_elements if elem.element_type == FaultType.OPERATION]
        phenomena = [elem.content for elem in fault_elements if elem.element_type == FaultType.PHENOMENON]
        locations = [elem.content for elem in fault_elements if elem.element_type == FaultType.LOCATION]
        alarms = [elem.content for elem in fault_elements if elem.element_type == FaultType.ALARM]
        
        # 1. 根据现象直接查找原因
        if phenomena:
            direct_causes = self.get_fault_causes_by_phenomena(phenomena)
            reasoning_result["causes"].extend(direct_causes)
        
        # 2. 根据操作查找相关现象，再查找原因
        if operations:
            operation_phenomena = self.get_related_phenomena_by_operations(operations)
            reasoning_result["related_phenomena"].extend(operation_phenomena)
            
            # 基于操作相关现象查找原因
            op_phenomenon_titles = [p["phenomenon"] for p in operation_phenomena]
            indirect_causes = self.get_fault_causes_by_phenomena(op_phenomenon_titles)
            for cause in indirect_causes:
                cause["confidence"] *= 0.8  # 间接推理降低置信度
            reasoning_result["causes"].extend(indirect_causes)
        
        # 3. 根据部位查找常见现象和原因
        if locations:
            location_phenomena = self.get_location_phenomena(locations)
            reasoning_result["related_phenomena"].extend(location_phenomena)
            
            # 基于部位相关现象查找原因
            loc_phenomenon_titles = [p["phenomenon"] for p in location_phenomena]
            location_causes = self.get_fault_causes_by_phenomena(loc_phenomenon_titles)
            for cause in location_causes:
                cause["confidence"] *= 0.7  # 部位推理置信度更低
            reasoning_result["causes"].extend(location_causes)
        
        # 4. 根据报警查找相关现象和原因
        if alarms:
            alarm_phenomena = self.get_alarm_phenomena(alarms)
            reasoning_result["related_phenomena"].extend(alarm_phenomena)
            
            # 基于报警相关现象查找原因
            alarm_phenomenon_titles = [p["phenomenon"] for p in alarm_phenomena]
            alarm_causes = self.get_fault_causes_by_phenomena(alarm_phenomenon_titles)
            reasoning_result["causes"].extend(alarm_causes)
        
        return reasoning_result
    
    def add_new_knowledge(self, fault_elements: List[FaultElement], 
                         solution: str, user_feedback: str) -> bool:
        """
        添加新知识到图谱
        
        Args:
            fault_elements: 故障元素
            solution: 解决方案
            user_feedback: 用户反馈
            
        Returns:
            是否添加成功
        """
        try:
            with self.driver.session() as session:
                # 这里可以实现知识图谱的动态更新逻辑
                # 暂时返回True，实际实现需要根据具体需求设计
                return True
        except Exception as e:
            self.logger.error(f"添加新知识失败: {e}")
            return False