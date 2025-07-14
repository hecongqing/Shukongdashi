"""
Neo4j图数据库管理器

负责知识图谱的构建、查询和管理
"""

from py2neo import Graph, Node, Relationship
from typing import List, Dict, Any, Optional
import json
from loguru import logger


class GraphManager:
    """Neo4j图数据库管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.uri = config['uri']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        
        self.graph = None
        self.connect()
    
    def connect(self):
        """连接Neo4j数据库"""
        try:
            self.graph = Graph(
                self.uri,
                auth=(self.user, self.password),
                name=self.database
            )
            logger.info(f"成功连接到Neo4j数据库: {self.uri}")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise
    
    def create_constraints(self):
        """创建数据库约束"""
        try:
            # 为每种实体类型创建唯一性约束
            entity_types = ['Equipment', 'Component', 'Fault', 'Cause', 'Solution', 'Symptom', 'Material', 'Tool']
            
            for entity_type in entity_types:
                constraint_query = f"""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) 
                REQUIRE n.name IS UNIQUE
                """
                self.graph.run(constraint_query)
            
            logger.info("数据库约束创建完成")
        except Exception as e:
            logger.error(f"创建约束失败: {e}")
    
    def create_equipment_node(self, equipment_data: Dict[str, Any]) -> Node:
        """创建装备节点"""
        node = Node(
            "Equipment",
            name=equipment_data['name'],
            type=equipment_data.get('type', ''),
            model=equipment_data.get('model', ''),
            manufacturer=equipment_data.get('manufacturer', ''),
            description=equipment_data.get('description', '')
        )
        
        self.graph.create(node)
        return node
    
    def create_component_node(self, component_data: Dict[str, Any]) -> Node:
        """创建部件节点"""
        node = Node(
            "Component",
            name=component_data['name'],
            type=component_data.get('type', ''),
            description=component_data.get('description', '')
        )
        
        self.graph.create(node)
        return node
    
    def create_fault_node(self, fault_data: Dict[str, Any]) -> Node:
        """创建故障节点"""
        node = Node(
            "Fault",
            name=fault_data['name'],
            type=fault_data.get('type', ''),
            description=fault_data.get('description', ''),
            severity=fault_data.get('severity', 'medium')
        )
        
        self.graph.create(node)
        return node
    
    def create_cause_node(self, cause_data: Dict[str, Any]) -> Node:
        """创建原因节点"""
        node = Node(
            "Cause",
            name=cause_data['name'],
            description=cause_data.get('description', ''),
            probability=cause_data.get('probability', 0.5)
        )
        
        self.graph.create(node)
        return node
    
    def create_solution_node(self, solution_data: Dict[str, Any]) -> Node:
        """创建解决方案节点"""
        node = Node(
            "Solution",
            name=solution_data['name'],
            description=solution_data.get('description', ''),
            difficulty=solution_data.get('difficulty', 'medium'),
            cost=solution_data.get('cost', 'medium')
        )
        
        self.graph.create(node)
        return node
    
    def create_relationship(self, start_node: Node, end_node: Node, relationship_type: str, properties: Dict[str, Any] = None):
        """创建关系"""
        if properties is None:
            properties = {}
        
        relationship = Relationship(start_node, relationship_type, end_node, **properties)
        self.graph.create(relationship)
        return relationship
    
    def find_node_by_name(self, label: str, name: str) -> Optional[Node]:
        """根据名称查找节点"""
        query = f"MATCH (n:{label} {{name: $name}}) RETURN n"
        result = self.graph.run(query, name=name)
        record = result.data()
        
        if record:
            return record[0]['n']
        return None
    
    def get_or_create_node(self, label: str, node_data: Dict[str, Any]) -> Node:
        """获取或创建节点"""
        existing_node = self.find_node_by_name(label, node_data['name'])
        if existing_node:
            return existing_node
        
        # 创建新节点
        node = Node(label, **node_data)
        self.graph.create(node)
        return node
    
    def build_knowledge_graph(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """构建知识图谱"""
        logger.info("开始构建知识图谱")
        
        # 创建约束
        self.create_constraints()
        
        # 创建节点映射
        node_map = {}
        
        # 创建所有实体节点
        for entity in entities:
            entity_type = entity['type']
            entity_name = entity['text']
            
            # 根据实体类型创建节点
            if entity_type == 'Equipment':
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', '')
                }
                node = self.get_or_create_node('Equipment', node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
                
            elif entity_type == 'Component':
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', '')
                }
                node = self.get_or_create_node('Component', node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
                
            elif entity_type == 'Fault':
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', ''),
                    'severity': entity.get('severity', 'medium')
                }
                node = self.get_or_create_node('Fault', node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
                
            elif entity_type == 'Cause':
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', ''),
                    'probability': entity.get('probability', 0.5)
                }
                node = self.get_or_create_node('Cause', node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
                
            elif entity_type == 'Solution':
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', ''),
                    'difficulty': entity.get('difficulty', 'medium'),
                    'cost': entity.get('cost', 'medium')
                }
                node = self.get_or_create_node('Solution', node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
                
            elif entity_type in ['Symptom', 'Material', 'Tool']:
                node_data = {
                    'name': entity_name,
                    'description': entity.get('description', '')
                }
                node = self.get_or_create_node(entity_type, node_data)
                node_map[f"{entity_type}_{entity_name}"] = node
        
        # 创建关系
        for relation in relations:
            head_entity = relation['head']
            tail_entity = relation['tail']
            relation_type = relation['relation']
            
            # 查找头实体和尾实体节点
            head_node = None
            tail_node = None
            
            for key, node in node_map.items():
                if head_entity in key:
                    head_node = node
                if tail_entity in key:
                    tail_node = node
            
            if head_node and tail_node:
                self.create_relationship(head_node, tail_node, relation_type)
        
        logger.info(f"知识图谱构建完成，创建了 {len(node_map)} 个节点和 {len(relations)} 个关系")
    
    def query_equipment_faults(self, equipment_name: str) -> List[Dict[str, Any]]:
        """查询装备的故障信息"""
        query = """
        MATCH (e:Equipment)-[:HAS_FAULT]->(f:Fault)
        WHERE e.name CONTAINS $equipment_name
        RETURN e.name as equipment, f.name as fault, f.description as description
        """
        
        result = self.graph.run(query, equipment_name=equipment_name)
        return result.data()
    
    def query_fault_causes(self, fault_name: str) -> List[Dict[str, Any]]:
        """查询故障的原因"""
        query = """
        MATCH (f:Fault)-[:CAUSES]-(c:Cause)
        WHERE f.name CONTAINS $fault_name
        RETURN f.name as fault, c.name as cause, c.description as description
        """
        
        result = self.graph.run(query, fault_name=fault_name)
        return result.data()
    
    def query_fault_solutions(self, fault_name: str) -> List[Dict[str, Any]]:
        """查询故障的解决方案"""
        query = """
        MATCH (f:Fault)-[:SOLVES]-(s:Solution)
        WHERE f.name CONTAINS $fault_name
        RETURN f.name as fault, s.name as solution, s.description as description, s.difficulty as difficulty
        """
        
        result = self.graph.run(query, fault_name=fault_name)
        return result.data()
    
    def query_component_faults(self, component_name: str) -> List[Dict[str, Any]]:
        """查询部件的故障信息"""
        query = """
        MATCH (c:Component)-[:HAS_FAULT]->(f:Fault)
        WHERE c.name CONTAINS $component_name
        RETURN c.name as component, f.name as fault, f.description as description
        """
        
        result = self.graph.run(query, component_name=component_name)
        return result.data()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = {}
        
        # 统计节点数量
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as type, count(n) as count
        """
        node_result = self.graph.run(node_query)
        stats['nodes'] = {record['type']: record['count'] for record in node_result}
        
        # 统计关系数量
        relation_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        """
        relation_result = self.graph.run(relation_query)
        stats['relations'] = {record['type']: record['count'] for record in relation_result}
        
        return stats
    
    def clear_database(self):
        """清空数据库"""
        query = "MATCH (n) DETACH DELETE n"
        self.graph.run(query)
        logger.info("数据库已清空")