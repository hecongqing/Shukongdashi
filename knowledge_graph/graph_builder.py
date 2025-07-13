"""
知识图谱构建模块
负责将抽取的实体和关系构建成知识图谱，并存储到Neo4j数据库
"""
import json
import re
from typing import List, Dict, Tuple, Any, Set, Optional
from pathlib import Path
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from py2neo.bulk import create_nodes, create_relationships
import numpy as np
from collections import defaultdict, Counter
from loguru import logger
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.settings import CONFIG
from models.ner_model import NERTrainer
from models.relation_extraction import RelationExtractionTrainer


class Triple:
    """三元组类"""
    
    def __init__(self, head: str, relation: str, tail: str, 
                 confidence: float = 1.0, source: str = ""):
        self.head = head.strip()
        self.relation = relation.strip()
        self.tail = tail.strip()
        self.confidence = confidence
        self.source = source
    
    def __str__(self):
        return f"({self.head}, {self.relation}, {self.tail})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return (self.head == other.head and 
                self.relation == other.relation and 
                self.tail == other.tail)
    
    def __hash__(self):
        return hash((self.head, self.relation, self.tail))
    
    def to_dict(self):
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "confidence": self.confidence,
            "source": self.source
        }


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["database"]["neo4j"]
        
        # 连接Neo4j数据库
        self.graph = Graph(
            self.config["uri"],
            auth=(self.config["user"], self.config["password"]),
            name=self.config.get("database", "neo4j")
        )
        
        # 图谱schema配置
        self.schema = CONFIG["graph_schema"]
        
        # 实体和关系统计
        self.entity_stats = defaultdict(int)
        self.relation_stats = defaultdict(int)
        
        # 初始化匹配器
        self.node_matcher = NodeMatcher(self.graph)
        self.rel_matcher = RelationshipMatcher(self.graph)
        
        logger.info("Knowledge Graph Builder initialized")
    
    def clear_graph(self):
        """清空图数据库"""
        logger.warning("Clearing all data in Neo4j database...")
        self.graph.delete_all()
        logger.info("Database cleared")
    
    def create_schema(self):
        """创建图谱schema约束和索引"""
        logger.info("Creating graph schema...")
        
        # 创建实体类型的唯一性约束
        for entity_type, properties in self.schema["entities"].items():
            constraints = properties.get("constraints", [])
            for constraint in constraints:
                try:
                    # 创建唯一性约束
                    cypher = f"CREATE CONSTRAINT ON (n:{entity_type}) ASSERT n.{constraint} IS UNIQUE"
                    self.graph.run(cypher)
                    logger.info(f"Created constraint for {entity_type}.{constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
        
        # 创建索引
        for entity_type in self.schema["entities"].keys():
            try:
                cypher = f"CREATE INDEX ON :{entity_type}(name)"
                self.graph.run(cypher)
                logger.info(f"Created index for {entity_type}")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")
        
        logger.info("Schema creation completed")
    
    def add_triple(self, triple: Triple) -> bool:
        """添加单个三元组到图数据库"""
        try:
            # 确定实体类型
            head_type = self._determine_entity_type(triple.head)
            tail_type = self._determine_entity_type(triple.tail)
            
            # 创建或获取头实体
            head_node = self._get_or_create_node(triple.head, head_type)
            
            # 创建或获取尾实体
            tail_node = self._get_or_create_node(triple.tail, tail_type)
            
            # 标准化关系名称
            relation_name = self._normalize_relation(triple.relation)
            
            # 创建关系
            relationship = Relationship(
                head_node, 
                relation_name, 
                tail_node,
                confidence=triple.confidence,
                source=triple.source
            )
            
            # 检查关系是否已存在
            existing_rel = self.rel_matcher.match(
                [head_node, tail_node], 
                r_type=relation_name
            ).first()
            
            if not existing_rel:
                self.graph.create(relationship)
                self.relation_stats[relation_name] += 1
                return True
            else:
                # 更新置信度（取最大值）
                if triple.confidence > existing_rel.get("confidence", 0):
                    existing_rel["confidence"] = triple.confidence
                    self.graph.push(existing_rel)
                return False
                
        except Exception as e:
            logger.error(f"Error adding triple {triple}: {e}")
            return False
    
    def batch_add_triples(self, triples: List[Triple], batch_size: int = 1000):
        """批量添加三元组"""
        logger.info(f"Adding {len(triples)} triples to knowledge graph...")
        
        # 分批处理
        for i in tqdm(range(0, len(triples), batch_size)):
            batch = triples[i:i + batch_size]
            
            # 收集节点和关系数据
            nodes_data = []
            relationships_data = []
            
            for triple in batch:
                # 准备节点数据
                head_type = self._determine_entity_type(triple.head)
                tail_type = self._determine_entity_type(triple.tail)
                
                nodes_data.append({
                    "name": triple.head,
                    "type": head_type,
                    "labels": [head_type]
                })
                nodes_data.append({
                    "name": triple.tail,
                    "type": tail_type,
                    "labels": [tail_type]
                })
                
                # 准备关系数据
                relation_name = self._normalize_relation(triple.relation)
                relationships_data.append({
                    "start_node": {"name": triple.head, "labels": [head_type]},
                    "end_node": {"name": triple.tail, "labels": [tail_type]},
                    "type": relation_name,
                    "properties": {
                        "confidence": triple.confidence,
                        "source": triple.source
                    }
                })
            
            # 去重节点
            unique_nodes = []
            seen_nodes = set()
            for node in nodes_data:
                node_key = (node["name"], tuple(node["labels"]))
                if node_key not in seen_nodes:
                    unique_nodes.append(node)
                    seen_nodes.add(node_key)
            
            # 批量创建节点
            self._batch_create_nodes(unique_nodes)
            
            # 批量创建关系
            self._batch_create_relationships(relationships_data)
        
        logger.info("Batch triple addition completed")
    
    def _batch_create_nodes(self, nodes_data: List[Dict]):
        """批量创建节点"""
        try:
            # 按标签分组
            nodes_by_label = defaultdict(list)
            for node in nodes_data:
                label = node["labels"][0]
                nodes_by_label[label].append({
                    "name": node["name"],
                    "type": node["type"]
                })
            
            # 为每个标签批量创建节点
            for label, nodes in nodes_by_label.items():
                if nodes:
                    # 使用MERGE确保不重复创建
                    cypher = f"""
                    UNWIND $nodes AS node
                    MERGE (n:{label} {{name: node.name}})
                    SET n.type = node.type
                    """
                    self.graph.run(cypher, nodes=nodes)
                    
        except Exception as e:
            logger.error(f"Error in batch node creation: {e}")
    
    def _batch_create_relationships(self, relationships_data: List[Dict]):
        """批量创建关系"""
        try:
            cypher = """
            UNWIND $rels AS rel
            MATCH (a {name: rel.start_node.name}), (b {name: rel.end_node.name})
            MERGE (a)-[r:RELATES]->(b)
            SET r.type = rel.type,
                r.confidence = rel.properties.confidence,
                r.source = rel.properties.source
            """
            self.graph.run(cypher, rels=relationships_data)
            
        except Exception as e:
            logger.error(f"Error in batch relationship creation: {e}")
    
    def _get_or_create_node(self, entity_name: str, entity_type: str) -> Node:
        """获取或创建节点"""
        # 查找现有节点
        existing_node = self.node_matcher.match(entity_type, name=entity_name).first()
        
        if existing_node:
            return existing_node
        
        # 创建新节点
        properties = {"name": entity_name, "type": entity_type}
        
        # 添加实体类型特定的属性
        if entity_type in self.schema["entities"]:
            schema_props = self.schema["entities"][entity_type]["properties"]
            for prop in schema_props:
                if prop not in properties:
                    properties[prop] = ""
        
        node = Node(entity_type, **properties)
        self.graph.create(node)
        
        # 更新统计
        self.entity_stats[entity_type] += 1
        
        return node
    
    def _determine_entity_type(self, entity_name: str) -> str:
        """确定实体类型"""
        # 简单的实体类型判断逻辑
        # 在实际应用中，可以使用更复杂的分类器
        
        # 检查是否是人名
        if self._is_person_name(entity_name):
            return "Person"
        
        # 检查是否是机构名
        if self._is_organization_name(entity_name):
            return "Organization"
        
        # 检查是否是地名
        if self._is_location_name(entity_name):
            return "Location"
        
        # 检查是否是事件
        if self._is_event_name(entity_name):
            return "Event"
        
        # 默认为概念
        return "Concept"
    
    def _is_person_name(self, name: str) -> bool:
        """判断是否为人名"""
        # 简单的人名判断规则
        person_indicators = ["先生", "女士", "教授", "博士", "院士", "主席", "总统", "总理"]
        return any(indicator in name for indicator in person_indicators) or len(name) <= 4
    
    def _is_organization_name(self, name: str) -> bool:
        """判断是否为机构名"""
        org_indicators = ["公司", "企业", "学校", "大学", "学院", "研究所", "医院", "银行", "政府", "部门"]
        return any(indicator in name for indicator in org_indicators)
    
    def _is_location_name(self, name: str) -> bool:
        """判断是否为地名"""
        location_indicators = ["市", "省", "县", "区", "国", "州", "路", "街", "村", "镇"]
        return any(indicator in name for indicator in location_indicators)
    
    def _is_event_name(self, name: str) -> bool:
        """判断是否为事件名"""
        event_indicators = ["会议", "活动", "比赛", "战争", "革命", "运动", "节日", "仪式"]
        return any(indicator in name for indicator in event_indicators)
    
    def _normalize_relation(self, relation: str) -> str:
        """标准化关系名称"""
        # 移除空格和特殊字符
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '_', relation)
        normalized = normalized.strip('_')
        
        # 关系名称映射
        relation_mapping = {
            "出生于": "BORN_IN",
            "毕业于": "GRADUATED_FROM",
            "工作于": "WORKS_FOR",
            "位于": "LOCATED_IN",
            "属于": "BELONGS_TO",
            "参与": "PARTICIPATED_IN",
            "发生在": "HAPPENED_IN"
        }
        
        return relation_mapping.get(normalized, normalized.upper())
    
    def load_triples_from_file(self, file_path: str) -> List[Triple]:
        """从文件加载三元组"""
        triples = []
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and all(k in item for k in ['head', 'relation', 'tail']):
                            triple = Triple(
                                head=item['head'],
                                relation=item['relation'],
                                tail=item['tail'],
                                confidence=item.get('confidence', 1.0),
                                source=item.get('source', 'file')
                            )
                            triples.append(triple)
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    triple = Triple(
                        head=str(row['head']),
                        relation=str(row['relation']),
                        tail=str(row['tail']),
                        confidence=row.get('confidence', 1.0),
                        source=row.get('source', 'file')
                    )
                    triples.append(triple)
            
            else:
                # 假设是文本格式：head \t relation \t tail
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            triple = Triple(
                                head=parts[0],
                                relation=parts[1],
                                tail=parts[2],
                                confidence=float(parts[3]) if len(parts) > 3 else 1.0,
                                source='file'
                            )
                            triples.append(triple)
            
            logger.info(f"Loaded {len(triples)} triples from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading triples from {file_path}: {e}")
        
        return triples
    
    def extract_triples_from_text(self, texts: List[str], 
                                 ner_model_path: str = None,
                                 re_model_path: str = None) -> List[Triple]:
        """从文本中抽取三元组"""
        logger.info(f"Extracting triples from {len(texts)} texts...")
        
        # 初始化模型
        ner_trainer = NERTrainer()
        re_trainer = RelationExtractionTrainer()
        
        # 加载模型
        ner_model = ner_trainer.load_model(ner_model_path)
        re_model = re_trainer.load_model(re_model_path)
        
        all_triples = []
        
        for text in tqdm(texts, desc="Extracting triples"):
            try:
                # 实体识别
                entities = ner_trainer.predict(text, ner_model)
                
                if len(entities) < 2:
                    continue
                
                # 生成实体对
                entity_pairs = []
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        entity1 = entities[i][0]
                        entity2 = entities[j][0]
                        entity_pairs.append((entity1, entity2))
                
                # 关系抽取
                if entity_pairs:
                    texts_for_re = [text] * len(entity_pairs)
                    relations = re_trainer.batch_predict(texts_for_re, entity_pairs, re_model)
                    
                    # 构建三元组
                    for (entity1, entity2), (relation, confidence) in zip(entity_pairs, relations):
                        if relation != "其他" and confidence > 0.5:  # 过滤低置信度关系
                            triple = Triple(
                                head=entity1,
                                relation=relation,
                                tail=entity2,
                                confidence=confidence,
                                source="extraction"
                            )
                            all_triples.append(triple)
                            
            except Exception as e:
                logger.error(f"Error extracting triples from text: {e}")
        
        logger.info(f"Extracted {len(all_triples)} triples")
        return all_triples
    
    def build_graph_from_triples(self, triples: List[Triple]):
        """从三元组构建知识图谱"""
        logger.info("Building knowledge graph from triples...")
        
        # 创建schema
        self.create_schema()
        
        # 批量添加三元组
        self.batch_add_triples(triples)
        
        # 打印统计信息
        self.print_statistics()
        
        logger.info("Knowledge graph construction completed")
    
    def build_graph_from_file(self, file_path: str):
        """从文件构建知识图谱"""
        triples = self.load_triples_from_file(file_path)
        if triples:
            self.build_graph_from_triples(triples)
    
    def build_graph_from_texts(self, texts: List[str], 
                              ner_model_path: str = None,
                              re_model_path: str = None):
        """从文本构建知识图谱"""
        triples = self.extract_triples_from_text(texts, ner_model_path, re_model_path)
        if triples:
            self.build_graph_from_triples(triples)
    
    def print_statistics(self):
        """打印图谱统计信息"""
        logger.info("Knowledge Graph Statistics:")
        
        # 节点统计
        node_counts = {}
        for entity_type in self.schema["entities"].keys():
            count = len(self.node_matcher.match(entity_type))
            node_counts[entity_type] = count
            logger.info(f"  {entity_type} nodes: {count}")
        
        total_nodes = sum(node_counts.values())
        logger.info(f"  Total nodes: {total_nodes}")
        
        # 关系统计
        cypher = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"
        result = self.graph.run(cypher)
        
        total_relations = 0
        for record in result:
            rel_type = record["rel_type"]
            count = record["count"]
            logger.info(f"  {rel_type} relations: {count}")
            total_relations += count
        
        logger.info(f"  Total relations: {total_relations}")
    
    def query_graph(self, cypher_query: str) -> List[Dict]:
        """执行Cypher查询"""
        try:
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def find_entity(self, entity_name: str) -> List[Dict]:
        """查找实体"""
        cypher = """
        MATCH (n)
        WHERE n.name CONTAINS $name
        RETURN n.name as name, labels(n) as types, n as entity
        LIMIT 10
        """
        return self.query_graph(cypher.replace("$name", f"'{entity_name}'"))
    
    def find_relations(self, entity_name: str) -> List[Dict]:
        """查找实体的所有关系"""
        cypher = """
        MATCH (n {name: $name})-[r]-(m)
        RETURN n.name as entity1, type(r) as relation, m.name as entity2,
               r.confidence as confidence
        """
        return self.query_graph(cypher.replace("$name", f"'{entity_name}'"))
    
    def visualize_subgraph(self, entity_name: str, depth: int = 2, save_path: str = None):
        """可视化子图"""
        # 使用NetworkX创建图
        G = nx.Graph()
        
        # 获取子图数据
        cypher = f"""
        MATCH path = (n {{name: '{entity_name}'}})-[*1..{depth}]-(m)
        RETURN path
        LIMIT 100
        """
        
        result = self.graph.run(cypher)
        
        for record in result:
            path = record["path"]
            for i in range(len(path.nodes) - 1):
                node1 = path.nodes[i]
                node2 = path.nodes[i + 1]
                rel = path.relationships[i]
                
                G.add_node(node1["name"], type=list(node1.labels)[0])
                G.add_node(node2["name"], type=list(node2.labels)[0])
                G.add_edge(node1["name"], node2["name"], relation=type(rel).__name__)
        
        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'Unknown')
            if node_type == 'Person':
                node_colors.append('lightblue')
            elif node_type == 'Organization':
                node_colors.append('lightgreen')
            elif node_type == 'Location':
                node_colors.append('orange')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Knowledge Subgraph for '{entity_name}'")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--clear", action="store_true", help="Clear the graph database")
    parser.add_argument("--build-from-file", help="Build graph from triples file")
    parser.add_argument("--build-from-text", help="Build graph from text file")
    parser.add_argument("--ner-model", help="NER model path")
    parser.add_argument("--re-model", help="Relation extraction model path")
    parser.add_argument("--query", help="Cypher query to execute")
    parser.add_argument("--find-entity", help="Find entity by name")
    parser.add_argument("--visualize", help="Visualize subgraph for entity")
    
    args = parser.parse_args()
    
    builder = KnowledgeGraphBuilder()
    
    if args.clear:
        builder.clear_graph()
    
    if args.build_from_file:
        builder.build_graph_from_file(args.build_from_file)
    
    if args.build_from_text:
        with open(args.build_from_text, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        builder.build_graph_from_texts(texts, args.ner_model, args.re_model)
    
    if args.query:
        results = builder.query_graph(args.query)
        for result in results:
            print(result)
    
    if args.find_entity:
        entities = builder.find_entity(args.find_entity)
        for entity in entities:
            print(entity)
    
    if args.visualize:
        builder.visualize_subgraph(args.visualize)


if __name__ == "__main__":
    main()