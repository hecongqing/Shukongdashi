#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j知识图谱构建模块
将抽取的实体和关系导入到Neo4j图数据库中
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
import networkx as nx
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class Neo4jGraphBuilder:
    """Neo4j知识图谱构建器"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.graph = None
        self.matcher = None
        
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.graph = Graph(self.uri, auth=(self.user, self.password))
            self.matcher = NodeMatcher(self.graph)
            logger.info("成功连接到Neo4j数据库")
            
            # 测试连接
            result = self.graph.run("RETURN 1 as test")
            logger.info("数据库连接测试成功")
            
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise
    
    def create_constraints(self):
        """创建数据库约束"""
        try:
            # 创建唯一性约束
            constraints = [
                "CREATE CONSTRAINT equipment_name IF NOT EXISTS FOR (e:Equipment) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT fault_name IF NOT EXISTS FOR (f:Fault) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
                "CREATE CONSTRAINT system_name IF NOT EXISTS FOR (s:System) REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT error_code_name IF NOT EXISTS FOR (ec:ErrorCode) REQUIRE ec.code IS UNIQUE",
                "CREATE CONSTRAINT solution_name IF NOT EXISTS FOR (sol:Solution) REQUIRE sol.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    self.graph.run(constraint)
                except Exception as e:
                    logger.warning(f"创建约束失败: {e}")
            
            logger.info("数据库约束创建完成")
            
        except Exception as e:
            logger.error(f"创建约束失败: {e}")
    
    def create_indexes(self):
        """创建数据库索引"""
        try:
            # 创建索引
            indexes = [
                "CREATE INDEX equipment_type IF NOT EXISTS FOR (e:Equipment) ON (e.type)",
                "CREATE INDEX component_type IF NOT EXISTS FOR (c:Component) ON (c.type)",
                "CREATE INDEX fault_type IF NOT EXISTS FOR (f:Fault) ON (f.type)",
                "CREATE INDEX brand_type IF NOT EXISTS FOR (b:Brand) ON (b.type)"
            ]
            
            for index in indexes:
                try:
                    self.graph.run(index)
                except Exception as e:
                    logger.warning(f"创建索引失败: {e}")
            
            logger.info("数据库索引创建完成")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def create_entity_node(self, entity: Dict[str, Any]) -> Node:
        """创建实体节点"""
        entity_type = entity.get('type', 'unknown')
        entity_text = entity.get('text', '')
        
        # 根据实体类型创建不同的节点标签
        if entity_type == 'equipment':
            node = Node("Equipment", name=entity_text, type=entity_type)
        elif entity_type == 'component':
            node = Node("Component", name=entity_text, type=entity_type)
        elif entity_type == 'fault':
            node = Node("Fault", name=entity_text, type=entity_type)
        elif entity_type == 'brand':
            node = Node("Brand", name=entity_text, type=entity_type)
        elif entity_type == 'system':
            node = Node("System", name=entity_text, type=entity_type)
        elif entity_type == 'error_code':
            node = Node("ErrorCode", code=entity_text, type=entity_type)
        elif entity_type == 'solution':
            node = Node("Solution", name=entity_text, type=entity_type)
        else:
            node = Node("Entity", name=entity_text, type=entity_type)
        
        # 添加额外属性
        for key, value in entity.items():
            if key not in ['type', 'text', 'name', 'code']:
                node[key] = value
        
        return node
    
    def create_relationship(self, head_node: Node, tail_node: Node, 
                          relation: Dict[str, Any]) -> Relationship:
        """创建关系"""
        relation_type = relation.get('relation', 'RELATED')
        confidence = relation.get('confidence', 0.5)
        
        # 创建关系
        rel = Relationship(head_node, relation_type, tail_node, confidence=confidence)
        
        # 添加额外属性
        for key, value in relation.items():
            if key not in ['relation', 'confidence']:
                rel[key] = value
        
        return rel
    
    def merge_node(self, node: Node) -> Node:
        """合并节点（如果已存在则更新，否则创建）"""
        try:
            # 根据节点类型查找现有节点
            if "Equipment" in node.labels:
                existing = self.matcher.match("Equipment", name=node["name"]).first()
            elif "Component" in node.labels:
                existing = self.matcher.match("Component", name=node["name"]).first()
            elif "Fault" in node.labels:
                existing = self.matcher.match("Fault", name=node["name"]).first()
            elif "Brand" in node.labels:
                existing = self.matcher.match("Brand", name=node["name"]).first()
            elif "System" in node.labels:
                existing = self.matcher.match("System", name=node["name"]).first()
            elif "ErrorCode" in node.labels:
                existing = self.matcher.match("ErrorCode", code=node["code"]).first()
            elif "Solution" in node.labels:
                existing = self.matcher.match("Solution", name=node["name"]).first()
            else:
                existing = None
            
            if existing:
                # 更新现有节点属性
                for key, value in node.items():
                    existing[key] = value
                return existing
            else:
                # 创建新节点
                self.graph.create(node)
                return node
                
        except Exception as e:
            logger.error(f"合并节点失败: {e}")
            # 直接创建节点
            self.graph.create(node)
            return node
    
    def build_from_extraction_results(self, extraction_file: str):
        """从抽取结果构建知识图谱"""
        logger.info(f"开始从文件构建知识图谱: {extraction_file}")
        
        # 读取抽取结果
        with open(extraction_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        entity_nodes = {}  # 存储实体节点
        relationships = []  # 存储关系
        
        for result in results:
            text = result.get('text', '')
            entities = result.get('entities', [])
            relations = result.get('relations', [])
            
            logger.info(f"处理文本: {text[:50]}...")
            
            # 创建实体节点
            for entity in entities:
                entity_key = f"{entity['type']}_{entity['text']}"
                if entity_key not in entity_nodes:
                    node = self.create_entity_node(entity)
                    merged_node = self.merge_node(node)
                    entity_nodes[entity_key] = merged_node
            
            # 创建关系
            for relation in relations:
                head_text = relation.get('head', '')
                tail_text = relation.get('tail', '')
                
                # 查找对应的实体节点
                head_node = None
                tail_node = None
                
                for entity_key, node in entity_nodes.items():
                    if head_text in entity_key:
                        head_node = node
                    if tail_text in entity_key:
                        tail_node = node
                
                if head_node and tail_node:
                    rel = self.create_relationship(head_node, tail_node, relation)
                    relationships.append(rel)
        
        # 批量创建关系
        if relationships:
            self.graph.create(*relationships)
            logger.info(f"创建了 {len(relationships)} 个关系")
        
        logger.info("知识图谱构建完成")
    
    def build_from_csv_files(self, csv_dir: str):
        """从CSV文件构建知识图谱"""
        logger.info(f"开始从CSV目录构建知识图谱: {csv_dir}")
        
        # 读取实体CSV文件
        entity_files = {
            'equipment': 'equipment_info.csv',
            'fault': 'fault_cases.csv',
            'solution': 'maintenance_procedures.csv'
        }
        
        entity_nodes = {}
        
        for entity_type, filename in entity_files.items():
            file_path = os.path.join(csv_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                for _, row in df.iterrows():
                    content = row.get('content', '')
                    if content:
                        entity = {
                            'type': entity_type,
                            'text': content,
                            'source_file': row.get('file_name', '')
                        }
                        
                        entity_key = f"{entity_type}_{content}"
                        if entity_key not in entity_nodes:
                            node = self.create_entity_node(entity)
                            merged_node = self.merge_node(node)
                            entity_nodes[entity_key] = merged_node
        
        logger.info(f"从CSV文件创建了 {len(entity_nodes)} 个实体节点")
    
    def query_graph(self, cypher_query: str) -> List[Dict]:
        """查询图数据库"""
        try:
            result = self.graph.run(cypher_query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = {}
        
        # 统计节点数量
        node_queries = [
            ("Equipment", "MATCH (e:Equipment) RETURN count(e) as count"),
            ("Component", "MATCH (c:Component) RETURN count(c) as count"),
            ("Fault", "MATCH (f:Fault) RETURN count(f) as count"),
            ("Brand", "MATCH (b:Brand) RETURN count(b) as count"),
            ("System", "MATCH (s:System) RETURN count(s) as count"),
            ("ErrorCode", "MATCH (ec:ErrorCode) RETURN count(ec) as count"),
            ("Solution", "MATCH (sol:Solution) RETURN count(sol) as count")
        ]
        
        for label, query in node_queries:
            result = self.query_graph(query)
            if result:
                stats[f"{label}_count"] = result[0]['count']
        
        # 统计关系数量
        rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self.query_graph(rel_query)
        if result:
            stats['relationship_count'] = result[0]['count']
        
        # 统计关系类型
        rel_type_query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        result = self.query_graph(rel_type_query)
        if result:
            stats['relationship_types'] = {r['type']: r['count'] for r in result}
        
        return stats
    
    def export_to_networkx(self) -> nx.DiGraph:
        """导出为NetworkX图"""
        logger.info("开始导出为NetworkX图")
        
        G = nx.DiGraph()
        
        # 查询所有节点
        nodes_query = """
        MATCH (n)
        RETURN labels(n) as labels, properties(n) as properties
        """
        nodes = self.query_graph(nodes_query)
        
        for node in nodes:
            labels = node['labels']
            props = node['properties']
            
            # 确定节点标签
            if 'Equipment' in labels:
                node_id = f"Equipment_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Equipment')
            elif 'Component' in labels:
                node_id = f"Component_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Component')
            elif 'Fault' in labels:
                node_id = f"Fault_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Fault')
            elif 'Brand' in labels:
                node_id = f"Brand_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Brand')
            elif 'System' in labels:
                node_id = f"System_{props.get('name', '')}"
                G.add_node(node_id, **props, type='System')
            elif 'ErrorCode' in labels:
                node_id = f"ErrorCode_{props.get('code', '')}"
                G.add_node(node_id, **props, type='ErrorCode')
            elif 'Solution' in labels:
                node_id = f"Solution_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Solution')
            else:
                node_id = f"Entity_{props.get('name', '')}"
                G.add_node(node_id, **props, type='Entity')
        
        # 查询所有关系
        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a) as a_labels, properties(a) as a_props,
               type(r) as relation_type, properties(r) as r_props,
               labels(b) as b_labels, properties(b) as b_props
        """
        relationships = self.query_graph(relationships_query)
        
        for rel in relationships:
            a_labels = rel['a_labels']
            a_props = rel['a_props']
            b_labels = rel['b_labels']
            b_props = rel['b_props']
            rel_type = rel['relation_type']
            r_props = rel['r_props']
            
            # 确定节点ID
            if 'Equipment' in a_labels:
                a_id = f"Equipment_{a_props.get('name', '')}"
            elif 'Component' in a_labels:
                a_id = f"Component_{a_props.get('name', '')}"
            elif 'Fault' in a_labels:
                a_id = f"Fault_{a_props.get('name', '')}"
            elif 'Brand' in a_labels:
                a_id = f"Brand_{a_props.get('name', '')}"
            elif 'System' in a_labels:
                a_id = f"System_{a_props.get('name', '')}"
            elif 'ErrorCode' in a_labels:
                a_id = f"ErrorCode_{a_props.get('code', '')}"
            elif 'Solution' in a_labels:
                a_id = f"Solution_{a_props.get('name', '')}"
            else:
                a_id = f"Entity_{a_props.get('name', '')}"
            
            if 'Equipment' in b_labels:
                b_id = f"Equipment_{b_props.get('name', '')}"
            elif 'Component' in b_labels:
                b_id = f"Component_{b_props.get('name', '')}"
            elif 'Fault' in b_labels:
                b_id = f"Fault_{b_props.get('name', '')}"
            elif 'Brand' in b_labels:
                b_id = f"Brand_{b_props.get('name', '')}"
            elif 'System' in b_labels:
                b_id = f"System_{b_props.get('name', '')}"
            elif 'ErrorCode' in b_labels:
                b_id = f"ErrorCode_{b_props.get('code', '')}"
            elif 'Solution' in b_labels:
                b_id = f"Solution_{b_props.get('name', '')}"
            else:
                b_id = f"Entity_{b_props.get('name', '')}"
            
            # 添加边
            G.add_edge(a_id, b_id, relation_type=rel_type, **r_props)
        
        logger.info(f"NetworkX图导出完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        return G
    
    def visualize_graph(self, output_file: str = "../data/graph_visualization.png"):
        """可视化知识图谱"""
        logger.info("开始可视化知识图谱")
        
        # 导出为NetworkX图
        G = self.export_to_networkx()
        
        if G.number_of_nodes() == 0:
            logger.warning("图谱为空，无法可视化")
            return
        
        # 创建可视化
        plt.figure(figsize=(20, 16))
        
        # 设置布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'Entity')
            if node_type == 'Equipment':
                node_colors.append('red')
            elif node_type == 'Component':
                node_colors.append('blue')
            elif node_type == 'Fault':
                node_colors.append('orange')
            elif node_type == 'Brand':
                node_colors.append('green')
            elif node_type == 'System':
                node_colors.append('purple')
            elif node_type == 'ErrorCode':
                node_colors.append('brown')
            elif node_type == 'Solution':
                node_colors.append('pink')
            else:
                node_colors.append('gray')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.7)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # 绘制标签
        labels = {node: G.nodes[node].get('name', G.nodes[node].get('code', node)) 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='SimHei')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Equipment'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Component'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Fault'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Brand'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=10, label='System'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', 
                      markersize=10, label='ErrorCode'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', 
                      markersize=10, label='Solution')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left')
        plt.title("装备制造故障知识图谱", fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        
        # 保存图片
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图谱可视化已保存到: {output_file}")

def main():
    """主函数"""
    # 创建图谱构建器
    builder = Neo4jGraphBuilder()
    
    try:
        # 连接数据库
        builder.connect()
        
        # 创建约束和索引
        builder.create_constraints()
        builder.create_indexes()
        
        # 从抽取结果构建图谱
        extraction_file = "../data/llm_extraction_results.json"
        if os.path.exists(extraction_file):
            builder.build_from_extraction_results(extraction_file)
        
        # 从CSV文件构建图谱
        csv_dir = "../data/csv"
        if os.path.exists(csv_dir):
            builder.build_from_csv_files(csv_dir)
        
        # 获取统计信息
        stats = builder.get_statistics()
        print("知识图谱统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 可视化图谱
        builder.visualize_graph()
        
        print("知识图谱构建完成")
        
    except Exception as e:
        logger.error(f"构建知识图谱失败: {e}")
        print(f"构建失败: {e}")

if __name__ == "__main__":
    main()