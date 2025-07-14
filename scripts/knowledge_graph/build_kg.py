#!/usr/bin/env python3
"""
知识图谱构建脚本
从处理后的数据构建Neo4j知识图谱
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
import pandas as pd
from loguru import logger
from neo4j import GraphDatabase

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.config.settings import get_settings
from backend.config.database import Neo4jConnection
from backend.services.entity_extraction_service import EntityExtractionService
from backend.services.relation_extraction_service import RelationExtractionService
from backend.utils.text_utils import TextPreprocessor

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.neo4j_conn = Neo4jConnection(
            self.settings.NEO4J_URI,
            self.settings.NEO4J_USER,
            self.settings.NEO4J_PASSWORD
        )
        self.entity_service = EntityExtractionService()
        self.relation_service = RelationExtractionService()
        self.text_preprocessor = TextPreprocessor()
        
        # 连接Neo4j
        self.driver = self.neo4j_conn.connect()
        
        # 实体和关系统计
        self.entity_stats = {}
        self.relation_stats = {}
        
    def __del__(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
    
    def clear_graph(self):
        """清空图数据库"""
        logger.info("清空现有图数据...")
        
        with self.driver.session() as session:
            # 删除所有节点和关系
            session.run("MATCH (n) DETACH DELETE n")
            
        logger.info("图数据清空完成")
    
    def create_constraints(self):
        """创建约束和索引"""
        logger.info("创建约束和索引...")
        
        constraints = [
            # 实体约束
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Equipment) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:FaultSymptom) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:FaultCause) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:RepairMethod) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Operation) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:AlarmCode) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (param:Parameter) REQUIRE param.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            
            # 索引
            "CREATE INDEX IF NOT EXISTS FOR (e:Equipment) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (f:FaultSymptom) ON (f.description)",
            "CREATE INDEX IF NOT EXISTS FOR (c:FaultCause) ON (c.description)",
            "CREATE INDEX IF NOT EXISTS FOR (r:RepairMethod) ON (r.description)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"创建约束失败: {constraint}, 错误: {e}")
        
        logger.info("约束和索引创建完成")
    
    async def load_data(self) -> List[Dict[str, Any]]:
        """加载处理后的数据"""
        data_path = f"{self.settings.PROCESSED_DATA_DIR}/all_fault_data.json"
        
        if not Path(data_path).exists():
            logger.error(f"数据文件不存在: {data_path}")
            return []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"加载了{len(data)}条数据")
        return data
    
    def _generate_entity_id(self, entity_text: str, entity_type: str) -> str:
        """生成实体ID"""
        import hashlib
        
        # 标准化文本
        normalized_text = self.text_preprocessor.normalize_text(entity_text)
        
        # 生成hash
        hash_obj = hashlib.md5(f"{entity_type}:{normalized_text}".encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    def create_entity_node(self, entity_text: str, entity_type: str, properties: Dict[str, Any] = None) -> str:
        """创建实体节点"""
        entity_id = self._generate_entity_id(entity_text, entity_type)
        
        # 标准化属性
        if properties is None:
            properties = {}
        
        properties.update({
            'id': entity_id,
            'name': entity_text,
            'type': entity_type,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
        
        # 根据实体类型确定标签
        label_mapping = {
            'EQUIPMENT': 'Equipment',
            'FAULT_SYMPTOM': 'FaultSymptom',
            'FAULT_CAUSE': 'FaultCause',
            'REPAIR_METHOD': 'RepairMethod',
            'PART': 'Part',
            'OPERATION': 'Operation',
            'ALARM_CODE': 'AlarmCode',
            'PARAMETER': 'Parameter',
            'LOCATION': 'Location'
        }
        
        label = label_mapping.get(entity_type, 'Entity')
        
        # 创建节点
        query = f"""
        MERGE (e:{label} {{id: $id}})
        SET e += $properties
        RETURN e.id as id
        """
        
        with self.driver.session() as session:
            result = session.run(query, id=entity_id, properties=properties)
            record = result.single()
            
            if record:
                # 更新统计
                if entity_type not in self.entity_stats:
                    self.entity_stats[entity_type] = 0
                self.entity_stats[entity_type] += 1
                
                return record['id']
        
        return entity_id
    
    def create_relation(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None):
        """创建关系"""
        if properties is None:
            properties = {}
        
        properties.update({
            'type': relation_type,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
        
        # 关系类型映射
        relation_mapping = {
            'CAUSES': 'CAUSES',
            'REPAIRS': 'REPAIRS',
            'CONTAINS': 'CONTAINS',
            'BELONGS_TO': 'BELONGS_TO',
            'RELATED_TO': 'RELATED_TO',
            'CONCURRENT': 'CONCURRENT',
            'PRECEDES': 'PRECEDES'
        }
        
        relation_label = relation_mapping.get(relation_type, 'RELATED_TO')
        
        # 创建关系
        query = f"""
        MATCH (source), (target)
        WHERE source.id = $source_id AND target.id = $target_id
        MERGE (source)-[r:{relation_label}]->(target)
        SET r += $properties
        RETURN r
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                                source_id=source_id, 
                                target_id=target_id, 
                                properties=properties)
            
            if result.single():
                # 更新统计
                if relation_type not in self.relation_stats:
                    self.relation_stats[relation_type] = 0
                self.relation_stats[relation_type] += 1
    
    async def extract_and_create_entities(self, data_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """提取并创建实体"""
        logger.info("开始提取和创建实体...")
        
        entity_mapping = {}  # 存储文本到实体ID的映射
        
        for i, item in enumerate(data_items):
            if i % 100 == 0:
                logger.info(f"处理进度: {i}/{len(data_items)}")
            
            text = item.get('processed_text', '')
            if not text:
                continue
            
            # 提取实体
            entities = await self.entity_service.extract_entities(text)
            
            # 创建实体节点
            for entity in entities:
                entity_id = self.create_entity_node(
                    entity.text, 
                    entity.entity_type,
                    {
                        'confidence': entity.confidence,
                        'source_data_id': item.get('id'),
                        'source_type': item.get('data_type')
                    }
                )
                
                # 存储映射关系
                if entity.text not in entity_mapping:
                    entity_mapping[entity.text] = []
                entity_mapping[entity.text].append(entity_id)
        
        logger.info(f"实体创建完成，共创建{sum(self.entity_stats.values())}个实体")
        return entity_mapping
    
    async def extract_and_create_relations(self, data_items: List[Dict[str, Any]], entity_mapping: Dict[str, List[str]]):
        """提取并创建关系"""
        logger.info("开始提取和创建关系...")
        
        for i, item in enumerate(data_items):
            if i % 100 == 0:
                logger.info(f"处理进度: {i}/{len(data_items)}")
            
            text = item.get('processed_text', '')
            if not text:
                continue
            
            # 提取关系
            relations = await self.relation_service.extract_relations(text)
            
            # 创建关系
            for relation in relations:
                source_entities = entity_mapping.get(relation.source_entity, [])
                target_entities = entity_mapping.get(relation.target_entity, [])
                
                # 创建所有可能的关系组合
                for source_id in source_entities:
                    for target_id in target_entities:
                        self.create_relation(
                            source_id,
                            target_id,
                            relation.relation_type,
                            {
                                'confidence': relation.confidence,
                                'source_data_id': item.get('id'),
                                'source_type': item.get('data_type')
                            }
                        )
        
        logger.info(f"关系创建完成，共创建{sum(self.relation_stats.values())}个关系")
    
    def create_structured_knowledge(self, data_items: List[Dict[str, Any]]):
        """创建结构化知识"""
        logger.info("创建结构化知识...")
        
        for item in data_items:
            # 提取结构化字段
            equipment_brand = item.get('equipment_brand')
            equipment_model = item.get('equipment_model')
            fault_phenomenon = item.get('fault_phenomenon')
            fault_cause = item.get('fault_cause')
            repair_method = item.get('repair_method')
            alarm_code = item.get('alarm_code')
            
            # 创建实体节点
            entity_ids = {}
            
            if equipment_brand:
                entity_ids['equipment'] = self.create_entity_node(
                    f"{equipment_brand} {equipment_model or ''}".strip(),
                    'EQUIPMENT',
                    {
                        'brand': equipment_brand,
                        'model': equipment_model,
                        'source_data_id': item.get('id')
                    }
                )
            
            if fault_phenomenon:
                entity_ids['symptom'] = self.create_entity_node(
                    fault_phenomenon,
                    'FAULT_SYMPTOM',
                    {'source_data_id': item.get('id')}
                )
            
            if fault_cause:
                entity_ids['cause'] = self.create_entity_node(
                    fault_cause,
                    'FAULT_CAUSE',
                    {'source_data_id': item.get('id')}
                )
            
            if repair_method:
                entity_ids['repair'] = self.create_entity_node(
                    repair_method,
                    'REPAIR_METHOD',
                    {'source_data_id': item.get('id')}
                )
            
            if alarm_code:
                entity_ids['alarm'] = self.create_entity_node(
                    alarm_code,
                    'ALARM_CODE',
                    {'source_data_id': item.get('id')}
                )
            
            # 创建关系
            if 'symptom' in entity_ids and 'cause' in entity_ids:
                self.create_relation(
                    entity_ids['symptom'],
                    entity_ids['cause'],
                    'CAUSES',
                    {'confidence': 0.9, 'source_data_id': item.get('id')}
                )
            
            if 'cause' in entity_ids and 'repair' in entity_ids:
                self.create_relation(
                    entity_ids['cause'],
                    entity_ids['repair'],
                    'REPAIRS',
                    {'confidence': 0.9, 'source_data_id': item.get('id')}
                )
            
            if 'equipment' in entity_ids and 'symptom' in entity_ids:
                self.create_relation(
                    entity_ids['equipment'],
                    entity_ids['symptom'],
                    'RELATED_TO',
                    {'confidence': 0.8, 'source_data_id': item.get('id')}
                )
            
            if 'alarm' in entity_ids and 'symptom' in entity_ids:
                self.create_relation(
                    entity_ids['alarm'],
                    entity_ids['symptom'],
                    'RELATED_TO',
                    {'confidence': 0.8, 'source_data_id': item.get('id')}
                )
        
        logger.info("结构化知识创建完成")
    
    def create_domain_knowledge(self):
        """创建领域知识"""
        logger.info("创建领域知识...")
        
        # 设备类型层次结构
        equipment_hierarchy = {
            '数控机床': ['车床', '铣床', '钻床', '磨床', '加工中心'],
            '车床': ['普通车床', '数控车床', '自动车床'],
            '铣床': ['立式铣床', '卧式铣床', '数控铣床'],
            '加工中心': ['立式加工中心', '卧式加工中心', '五轴加工中心']
        }
        
        # 创建设备层次结构
        for parent, children in equipment_hierarchy.items():
            parent_id = self.create_entity_node(parent, 'EQUIPMENT', {'is_category': True})
            
            for child in children:
                child_id = self.create_entity_node(child, 'EQUIPMENT', {'is_category': True})
                self.create_relation(child_id, parent_id, 'BELONGS_TO', {'confidence': 1.0})
        
        # 故障类型分类
        fault_categories = {
            '机械故障': ['轴承磨损', '齿轮损坏', '导轨磨损', '丝杠磨损'],
            '电气故障': ['电源故障', '伺服故障', '编码器故障', '继电器故障'],
            '液压故障': ['液压泵故障', '液压缸故障', '液压阀故障', '液压管路故障'],
            '控制故障': ['程序错误', '参数错误', '通讯故障', '传感器故障']
        }
        
        # 创建故障分类
        for category, faults in fault_categories.items():
            category_id = self.create_entity_node(category, 'FAULT_CAUSE', {'is_category': True})
            
            for fault in faults:
                fault_id = self.create_entity_node(fault, 'FAULT_CAUSE', {'is_category': True})
                self.create_relation(fault_id, category_id, 'BELONGS_TO', {'confidence': 1.0})
        
        logger.info("领域知识创建完成")
    
    def optimize_graph(self):
        """优化图结构"""
        logger.info("开始优化图结构...")
        
        # 合并相似实体
        self._merge_similar_entities()
        
        # 删除低置信度关系
        self._remove_low_confidence_relations()
        
        # 创建推理关系
        self._create_inferred_relations()
        
        logger.info("图结构优化完成")
    
    def _merge_similar_entities(self):
        """合并相似实体"""
        logger.info("合并相似实体...")
        
        # 查找名称相似的实体
        query = """
        MATCH (e1), (e2)
        WHERE e1.name CONTAINS e2.name OR e2.name CONTAINS e1.name
        AND e1.id <> e2.id
        AND labels(e1) = labels(e2)
        RETURN e1, e2
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            
            for record in result:
                e1 = record['e1']
                e2 = record['e2']
                
                # 简单的相似度判断
                if self._calculate_similarity(e1['name'], e2['name']) > 0.8:
                    # 合并实体（保留第一个，删除第二个）
                    merge_query = """
                    MATCH (e1 {id: $e1_id}), (e2 {id: $e2_id})
                    WITH e1, e2
                    MATCH (e2)-[r]-(other)
                    CREATE (e1)-[r2:REPLACED_BY {type: type(r)}]->(other)
                    SET r2 += properties(r)
                    DELETE r
                    WITH e1, e2
                    DELETE e2
                    """
                    
                    session.run(merge_query, e1_id=e1['id'], e2_id=e2['id'])
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的相似度计算
        if text1 == text2:
            return 1.0
        
        # 计算编辑距离
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = edit_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        
        return 1.0 - (distance / max_len)
    
    def _remove_low_confidence_relations(self):
        """删除低置信度关系"""
        logger.info("删除低置信度关系...")
        
        query = """
        MATCH ()-[r]->()
        WHERE r.confidence < 0.3
        DELETE r
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            logger.info(f"删除了低置信度关系")
    
    def _create_inferred_relations(self):
        """创建推理关系"""
        logger.info("创建推理关系...")
        
        # 传递性关系推理
        # 如果 A causes B 且 B causes C，则 A indirectly_causes C
        query = """
        MATCH (a)-[:CAUSES]->(b)-[:CAUSES]->(c)
        WHERE a <> c
        MERGE (a)-[r:INDIRECTLY_CAUSES]->(c)
        SET r.confidence = 0.7, r.inferred = true
        """
        
        with self.driver.session() as session:
            session.run(query)
    
    def generate_statistics(self):
        """生成统计信息"""
        logger.info("生成统计信息...")
        
        # 统计查询
        queries = {
            'nodes': 'MATCH (n) RETURN count(n) as count',
            'relationships': 'MATCH ()-[r]->() RETURN count(r) as count',
            'node_types': 'MATCH (n) RETURN labels(n) as labels, count(n) as count',
            'relationship_types': 'MATCH ()-[r]->() RETURN type(r) as type, count(r) as count'
        }
        
        stats = {}
        
        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                if name in ['nodes', 'relationships']:
                    stats[name] = result.single()['count']
                else:
                    stats[name] = [dict(record) for record in result]
        
        # 保存统计信息
        stats_path = f"{self.settings.KNOWLEDGE_DATA_DIR}/kg_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_path}")
        logger.info(f"图统计: {stats['nodes']} 个节点, {stats['relationships']} 个关系")
        
        return stats
    
    async def build_knowledge_graph(self):
        """构建知识图谱"""
        logger.info("开始构建知识图谱...")
        
        # 1. 清空现有图数据
        self.clear_graph()
        
        # 2. 创建约束和索引
        self.create_constraints()
        
        # 3. 加载数据
        data_items = await self.load_data()
        if not data_items:
            logger.error("没有数据可处理")
            return
        
        # 4. 创建结构化知识
        self.create_structured_knowledge(data_items)
        
        # 5. 提取并创建实体
        entity_mapping = await self.extract_and_create_entities(data_items)
        
        # 6. 提取并创建关系
        await self.extract_and_create_relations(data_items, entity_mapping)
        
        # 7. 创建领域知识
        self.create_domain_knowledge()
        
        # 8. 优化图结构
        self.optimize_graph()
        
        # 9. 生成统计信息
        stats = self.generate_statistics()
        
        logger.info("知识图谱构建完成！")
        return stats

async def main():
    """主函数"""
    builder = KnowledgeGraphBuilder()
    
    try:
        stats = await builder.build_knowledge_graph()
        print(f"知识图谱构建成功！")
        print(f"节点数: {stats['nodes']}")
        print(f"关系数: {stats['relationships']}")
        
    except Exception as e:
        logger.error(f"构建知识图谱失败: {e}")
        raise
    
    finally:
        if hasattr(builder, 'driver') and builder.driver:
            builder.driver.close()

if __name__ == "__main__":
    asyncio.run(main())