"""
Neo4j医疗知识图谱管理模块
"""
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm


@dataclass
class GraphConfig:
    """图数据库配置"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"


class MedicalGraphBuilder:
    """医疗知识图谱构建器"""
    
    # 节点标签定义
    NODE_LABELS = {
        'disease': 'Disease',
        'symptom': 'Symptom',
        'drug': 'Drug',
        'examination': 'Examination',
        'treatment': 'Treatment',
        'department': 'Department',
        'body_part': 'BodyPart',
        'pathogen': 'Pathogen'
    }
    
    # 关系类型定义
    RELATIONSHIP_TYPES = {
        'has_symptom': 'HAS_SYMPTOM',
        'treated_by': 'TREATED_BY',
        'examined_by': 'EXAMINED_BY',
        'caused_by': 'CAUSED_BY',
        'occurs_in': 'OCCURS_IN',
        'complication_of': 'COMPLICATION_OF',
        'contraindicated_for': 'CONTRAINDICATED_FOR',
        'indicated_for': 'INDICATED_FOR',
        'belongs_to': 'BELONGS_TO'
    }
    
    def __init__(self, config: GraphConfig):
        """初始化图谱构建器"""
        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri, 
            auth=(config.username, config.password)
        )
        logger.info(f"Connected to Neo4j at {config.uri}")
        
        # 创建索引
        self._create_indexes()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def _create_indexes(self):
        """创建索引以提高查询性能"""
        with self.driver.session(database=self.config.database) as session:
            # 为每种节点类型创建索引
            for node_type, label in self.NODE_LABELS.items():
                try:
                    session.run(f"CREATE INDEX {label}_name IF NOT EXISTS FOR (n:{label}) ON (n.name)")
                    session.run(f"CREATE INDEX {label}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)")
                    logger.info(f"Created indexes for {label}")
                except Exception as e:
                    logger.warning(f"Index creation for {label} failed: {e}")
    
    def create_node(self, node_type: str, properties: Dict[str, Any]) -> Optional[int]:
        """
        创建节点
        
        Args:
            node_type: 节点类型（如'disease', 'symptom'等）
            properties: 节点属性字典
            
        Returns:
            节点ID
        """
        if node_type not in self.NODE_LABELS:
            logger.error(f"Unknown node type: {node_type}")
            return None
            
        label = self.NODE_LABELS[node_type]
        
        # 确保有name属性
        if 'name' not in properties:
            logger.error("Node must have 'name' property")
            return None
        
        # 添加创建时间
        properties['created_at'] = datetime.now().isoformat()
        
        with self.driver.session(database=self.config.database) as session:
            # 检查节点是否已存在
            result = session.run(
                f"MATCH (n:{label} {{name: $name}}) RETURN n",
                name=properties['name']
            )
            
            if result.single():
                logger.info(f"{label} node '{properties['name']}' already exists")
                return None
            
            # 创建新节点
            result = session.run(
                f"CREATE (n:{label} $props) RETURN id(n) as node_id",
                props=properties
            )
            
            node_id = result.single()['node_id']
            logger.info(f"Created {label} node: {properties['name']} (ID: {node_id})")
            return node_id
    
    def create_relationship(self, 
                          source_type: str,
                          source_name: str,
                          target_type: str,
                          target_name: str,
                          relationship_type: str,
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        创建关系
        
        Args:
            source_type: 源节点类型
            source_name: 源节点名称
            target_type: 目标节点类型
            target_name: 目标节点名称
            relationship_type: 关系类型
            properties: 关系属性
            
        Returns:
            是否创建成功
        """
        if source_type not in self.NODE_LABELS or target_type not in self.NODE_LABELS:
            logger.error("Invalid node types")
            return False
            
        if relationship_type not in self.RELATIONSHIP_TYPES:
            logger.error(f"Unknown relationship type: {relationship_type}")
            return False
        
        source_label = self.NODE_LABELS[source_type]
        target_label = self.NODE_LABELS[target_type]
        rel_type = self.RELATIONSHIP_TYPES[relationship_type]
        
        if properties is None:
            properties = {}
            
        properties['created_at'] = datetime.now().isoformat()
        
        with self.driver.session(database=self.config.database) as session:
            # 创建关系
            result = session.run(f"""
                MATCH (a:{source_label} {{name: $source_name}})
                MATCH (b:{target_label} {{name: $target_name}})
                CREATE (a)-[r:{rel_type} $props]->(b)
                RETURN r
            """, source_name=source_name, target_name=target_name, props=properties)
            
            if result.single():
                logger.info(f"Created relationship: {source_name} -{rel_type}-> {target_name}")
                return True
            else:
                logger.warning(f"Failed to create relationship: nodes may not exist")
                return False
    
    def batch_create_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 1000):
        """
        批量创建节点
        
        Args:
            nodes: 节点列表，每个节点包含'type'和'properties'
            batch_size: 批处理大小
        """
        logger.info(f"Starting batch creation of {len(nodes)} nodes")
        
        # 按类型分组
        nodes_by_type = {}
        for node in nodes:
            node_type = node['type']
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node['properties'])
        
        # 批量创建
        with self.driver.session(database=self.config.database) as session:
            for node_type, node_list in nodes_by_type.items():
                if node_type not in self.NODE_LABELS:
                    logger.warning(f"Skipping unknown node type: {node_type}")
                    continue
                    
                label = self.NODE_LABELS[node_type]
                
                # 分批处理
                for i in tqdm(range(0, len(node_list), batch_size), 
                             desc=f"Creating {label} nodes"):
                    batch = node_list[i:i + batch_size]
                    
                    # 添加创建时间
                    for props in batch:
                        props['created_at'] = datetime.now().isoformat()
                    
                    # 使用UNWIND批量创建
                    session.run(f"""
                        UNWIND $batch as props
                        MERGE (n:{label} {{name: props.name}})
                        SET n += props
                    """, batch=batch)
        
        logger.info("Batch node creation completed")
    
    def batch_create_relationships(self, relationships: List[Dict[str, Any]], 
                                 batch_size: int = 1000):
        """
        批量创建关系
        
        Args:
            relationships: 关系列表，每个关系包含source, target, type等信息
            batch_size: 批处理大小
        """
        logger.info(f"Starting batch creation of {len(relationships)} relationships")
        
        # 按关系类型分组
        rels_by_type = {}
        for rel in relationships:
            rel_type = rel['type']
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)
        
        with self.driver.session(database=self.config.database) as session:
            for rel_type, rel_list in rels_by_type.items():
                if rel_type not in self.RELATIONSHIP_TYPES:
                    logger.warning(f"Skipping unknown relationship type: {rel_type}")
                    continue
                
                neo4j_rel_type = self.RELATIONSHIP_TYPES[rel_type]
                
                # 分批处理
                for i in tqdm(range(0, len(rel_list), batch_size),
                             desc=f"Creating {neo4j_rel_type} relationships"):
                    batch = rel_list[i:i + batch_size]
                    
                    # 准备批量数据
                    batch_data = []
                    for rel in batch:
                        props = rel.get('properties', {})
                        props['created_at'] = datetime.now().isoformat()
                        
                        batch_data.append({
                            'source_label': self.NODE_LABELS[rel['source_type']],
                            'source_name': rel['source_name'],
                            'target_label': self.NODE_LABELS[rel['target_type']],
                            'target_name': rel['target_name'],
                            'props': props
                        })
                    
                    # 批量创建关系
                    session.run(f"""
                        UNWIND $batch as rel
                        MATCH (a) WHERE a.name = rel.source_name 
                            AND labels(a)[0] = rel.source_label
                        MATCH (b) WHERE b.name = rel.target_name 
                            AND labels(b)[0] = rel.target_label
                        CREATE (a)-[r:{neo4j_rel_type}]->(b)
                        SET r += rel.props
                    """, batch=batch_data)
        
        logger.info("Batch relationship creation completed")
    
    def import_from_json(self, json_file: str):
        """
        从JSON文件导入数据
        
        JSON格式：
        {
            "nodes": [
                {"type": "disease", "properties": {"name": "糖尿病", ...}},
                ...
            ],
            "relationships": [
                {
                    "source_type": "disease",
                    "source_name": "糖尿病",
                    "target_type": "symptom",
                    "target_name": "多饮",
                    "type": "has_symptom",
                    "properties": {...}
                },
                ...
            ]
        }
        """
        logger.info(f"Importing data from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建节点
        if 'nodes' in data:
            self.batch_create_nodes(data['nodes'])
        
        # 创建关系
        if 'relationships' in data:
            self.batch_create_relationships(data['relationships'])
        
        logger.info("Import completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = {}
        
        with self.driver.session(database=self.config.database) as session:
            # 统计节点数量
            for node_type, label in self.NODE_LABELS.items():
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                stats[f'{node_type}_count'] = count
            
            # 统计关系数量
            for rel_type, neo4j_type in self.RELATIONSHIP_TYPES.items():
                result = session.run(
                    f"MATCH ()-[r:{neo4j_type}]->() RETURN count(r) as count"
                )
                count = result.single()['count']
                stats[f'{rel_type}_count'] = count
            
            # 总体统计
            result = session.run("MATCH (n) RETURN count(n) as count")
            stats['total_nodes'] = result.single()['count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['total_relationships'] = result.single()['count']
        
        return stats
    
    def query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行Cypher查询
        
        Args:
            cypher_query: Cypher查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        with self.driver.session(database=self.config.database) as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]
    
    def find_path(self, 
                  start_name: str, 
                  end_name: str, 
                  max_length: int = 5) -> List[Dict]:
        """
        查找两个节点之间的路径
        
        Args:
            start_name: 起始节点名称
            end_name: 结束节点名称
            max_length: 最大路径长度
            
        Returns:
            路径列表
        """
        query = """
        MATCH path = shortestPath(
            (start {name: $start_name})-[*..%d]-(end {name: $end_name})
        )
        RETURN path
        """ % max_length
        
        return self.query(query, {'start_name': start_name, 'end_name': end_name})
    
    def get_node_neighbors(self, 
                          node_name: str, 
                          depth: int = 1,
                          relationship_types: Optional[List[str]] = None) -> Dict:
        """
        获取节点的邻居
        
        Args:
            node_name: 节点名称
            depth: 搜索深度
            relationship_types: 限定的关系类型
            
        Returns:
            邻居信息
        """
        if relationship_types:
            rel_filter = '|'.join([
                self.RELATIONSHIP_TYPES.get(rt, rt) 
                for rt in relationship_types
            ])
            rel_pattern = f"[r:{rel_filter}*1..{depth}]"
        else:
            rel_pattern = f"[r*1..{depth}]"
        
        query = f"""
        MATCH (n {{name: $node_name}})-{rel_pattern}-(neighbor)
        RETURN DISTINCT neighbor, type(r[0]) as relationship
        """
        
        results = self.query(query, {'node_name': node_name})
        
        return {
            'center': node_name,
            'neighbors': results
        }


class MedicalGraphAnalyzer:
    """医疗知识图谱分析器"""
    
    def __init__(self, graph_builder: MedicalGraphBuilder):
        self.graph = graph_builder
    
    def find_similar_diseases(self, disease_name: str, limit: int = 10) -> List[Dict]:
        """
        查找相似疾病（基于共同症状）
        
        Args:
            disease_name: 疾病名称
            limit: 返回数量限制
            
        Returns:
            相似疾病列表
        """
        query = """
        MATCH (d1:Disease {name: $disease_name})-[:HAS_SYMPTOM]->(s:Symptom)
        MATCH (d2:Disease)-[:HAS_SYMPTOM]->(s)
        WHERE d1 <> d2
        WITH d2, COUNT(DISTINCT s) as common_symptoms
        ORDER BY common_symptoms DESC
        LIMIT $limit
        RETURN d2.name as disease, common_symptoms
        """
        
        return self.graph.query(query, {
            'disease_name': disease_name,
            'limit': limit
        })
    
    def get_treatment_options(self, disease_name: str) -> Dict[str, List]:
        """
        获取疾病的治疗方案
        
        Args:
            disease_name: 疾病名称
            
        Returns:
            治疗方案字典
        """
        # 获取药物治疗
        drugs_query = """
        MATCH (d:Disease {name: $disease_name})-[:TREATED_BY]->(drug:Drug)
        RETURN drug.name as name, drug.type as type, drug.usage as usage
        """
        drugs = self.graph.query(drugs_query, {'disease_name': disease_name})
        
        # 获取治疗方法
        treatments_query = """
        MATCH (d:Disease {name: $disease_name})-[:TREATED_BY]->(t:Treatment)
        RETURN t.name as name, t.description as description
        """
        treatments = self.graph.query(treatments_query, {'disease_name': disease_name})
        
        return {
            'drugs': drugs,
            'treatments': treatments
        }
    
    def get_diagnostic_info(self, symptoms: List[str]) -> List[Dict]:
        """
        根据症状推测可能的疾病
        
        Args:
            symptoms: 症状列表
            
        Returns:
            可能的疾病及概率
        """
        query = """
        UNWIND $symptoms as symptom_name
        MATCH (s:Symptom {name: symptom_name})<-[:HAS_SYMPTOM]-(d:Disease)
        WITH d, COUNT(DISTINCT s) as matched_symptoms
        MATCH (d)-[:HAS_SYMPTOM]->(total_s:Symptom)
        WITH d, matched_symptoms, COUNT(DISTINCT total_s) as total_symptoms
        RETURN d.name as disease, 
               matched_symptoms,
               total_symptoms,
               toFloat(matched_symptoms) / total_symptoms as probability
        ORDER BY probability DESC
        """
        
        return self.graph.query(query, {'symptoms': symptoms})


# 使用示例
if __name__ == "__main__":
    # 配置数据库连接
    config = GraphConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )
    
    # 创建图谱构建器
    with MedicalGraphBuilder(config) as builder:
        # 创建一些示例节点
        builder.create_node('disease', {
            'name': '糖尿病',
            'icd10': 'E11',
            'description': '一种慢性代谢性疾病'
        })
        
        builder.create_node('symptom', {
            'name': '多饮',
            'description': '饮水量明显增加'
        })
        
        builder.create_node('drug', {
            'name': '胰岛素',
            'type': '激素类药物',
            'usage': '皮下注射'
        })
        
        # 创建关系
        builder.create_relationship(
            'disease', '糖尿病',
            'symptom', '多饮',
            'has_symptom',
            {'probability': 0.8}
        )
        
        builder.create_relationship(
            'disease', '糖尿病',
            'drug', '胰岛素',
            'treated_by',
            {'effectiveness': '高'}
        )
        
        # 获取统计信息
        stats = builder.get_statistics()
        print("Graph statistics:", stats)
        
        # 分析器示例
        analyzer = MedicalGraphAnalyzer(builder)
        
        # 查找相似疾病
        similar = analyzer.find_similar_diseases('糖尿病')
        print("Similar diseases:", similar)
        
        # 获取治疗方案
        treatments = analyzer.get_treatment_options('糖尿病')
        print("Treatment options:", treatments)