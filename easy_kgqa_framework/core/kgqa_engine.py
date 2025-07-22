"""
简洁版KGQA引擎 - 使用Neo4j和实体识别
Easy Knowledge Graph Question Answering Engine with Neo4j and NER
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import re
from ..utils.text_utils import TextUtils
from ..utils.entity_recognizer import SimpleEntityRecognizer


class EasyKGQA:
    """简洁版知识图谱问答引擎 - 使用Neo4j"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        """
        初始化KGQA引擎
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: 用户名
            neo4j_password: 密码
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.text_utils = TextUtils()
        self.entity_recognizer = SimpleEntityRecognizer()
        self._init_database()
        
    def __del__(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver'):
            self.driver.close()
            
    def _init_database(self):
        """初始化数据库，创建基础约束"""
        with self.driver.session() as session:
            # 创建唯一性约束
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT equipment_name IF NOT EXISTS FOR (eq:Equipment) REQUIRE eq.name IS UNIQUE")
            session.run("CREATE CONSTRAINT component_name IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE")
    
    def add_entity(self, name: str, entity_type: str, description: str = "") -> bool:
        """添加实体到Neo4j"""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MERGE (e:Entity {name: $name}) "
                    "SET e.type = $type, e.description = $description "
                    "RETURN e",
                    name=name, type=entity_type, description=description
                )
                return result.single() is not None
            except Exception as e:
                print(f"添加实体失败: {e}")
                return False
    
    def add_relation(self, subject: str, predicate: str, obj: str) -> bool:
        """添加关系到Neo4j"""
        with self.driver.session() as session:
            try:
                session.run(
                    "MATCH (s:Entity {name: $subject}) "
                    "MATCH (o:Entity {name: $object}) "
                    "MERGE (s)-[r:RELATES {type: $predicate}]->(o)",
                    subject=subject, predicate=predicate, object=obj
                )
                return True
            except Exception as e:
                print(f"添加关系失败: {e}")
                return False
    
    def add_fault_case(self, equipment: str, symptom: str, cause: str, solution: str) -> bool:
        """添加故障案例"""
        with self.driver.session() as session:
            try:
                session.run(
                    "MERGE (eq:Equipment {name: $equipment}) "
                    "MERGE (s:Symptom {name: $symptom}) "
                    "MERGE (c:Cause {name: $cause}) "
                    "MERGE (sol:Solution {name: $solution}) "
                    "MERGE (eq)-[:HAS_SYMPTOM]->(s) "
                    "MERGE (s)-[:CAUSED_BY]->(c) "
                    "MERGE (c)-[:SOLVED_BY]->(sol)",
                    equipment=equipment, symptom=symptom, cause=cause, solution=solution
                )
                return True
            except Exception as e:
                print(f"添加故障案例失败: {e}")
                return False
    
    def query_entity(self, name: str) -> Optional[Dict]:
        """查询实体信息"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.name CONTAINS $name "
                "RETURN e.name as name, e.type as type, e.description as description "
                "LIMIT 1",
                name=name
            )
            record = result.single()
            if record:
                return {
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"]
                }
        return None
    
    def find_relations(self, entity: str) -> List[Dict]:
        """查找实体相关的关系"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r:RELATES]->(o:Entity) "
                "WHERE s.name CONTAINS $entity OR o.name CONTAINS $entity "
                "RETURN s.name as subject, r.type as predicate, o.name as object "
                "LIMIT 10",
                entity=entity
            )
            
            relations = []
            for record in result:
                relations.append({
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"]
                })
            return relations
    
    def search_fault_cases(self, query: str) -> List[Dict]:
        """搜索故障案例"""
        # 使用实体识别提取关键实体
        entities = self.entity_recognizer.recognize(query)
        
        with self.driver.session() as session:
            cases = []
            
            # 如果识别到实体，精确搜索
            if entities:
                for entity in entities:
                    result = session.run(
                        "MATCH (eq:Equipment)-[:HAS_SYMPTOM]->(s:Symptom)-[:CAUSED_BY]->(c:Cause)-[:SOLVED_BY]->(sol:Solution) "
                        "WHERE eq.name CONTAINS $entity OR s.name CONTAINS $entity "
                        "RETURN eq.name as equipment, s.name as symptom, c.name as cause, sol.name as solution "
                        "LIMIT 5",
                        entity=entity
                    )
                    
                    for record in result:
                        cases.append({
                            "equipment": record["equipment"],
                            "symptom": record["symptom"],
                            "cause": record["cause"],
                            "solution": record["solution"]
                        })
            
            # 如果没有识别到实体，使用关键词搜索
            if not cases:
                keywords = self.text_utils.extract_keywords(query)
                for keyword in keywords[:3]:  # 限制关键词数量
                    result = session.run(
                        "MATCH (eq:Equipment)-[:HAS_SYMPTOM]->(s:Symptom)-[:CAUSED_BY]->(c:Cause)-[:SOLVED_BY]->(sol:Solution) "
                        "WHERE s.name CONTAINS $keyword OR c.name CONTAINS $keyword "
                        "RETURN eq.name as equipment, s.name as symptom, c.name as cause, sol.name as solution "
                        "LIMIT 3",
                        keyword=keyword
                    )
                    
                    for record in result:
                        cases.append({
                            "equipment": record["equipment"],
                            "symptom": record["symptom"],
                            "cause": record["cause"],
                            "solution": record["solution"]
                        })
            
            return cases
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题的主要接口"""
        result = {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "entities_found": [],
            "sources": []
        }
        
        # 1. 实体识别
        entities = self.entity_recognizer.recognize(question)
        result["entities_found"] = entities
        
        # 2. 判断问题类型
        is_fault_question = any(word in question for word in ['怎么办', '故障', '问题', '不转', '不工作', '异常', '坏了'])
        is_definition_question = any(word in question for word in ['什么是', '是什么', '定义'])
        is_relation_question = any(word in question for word in ['包含', '有什么', '含有', '组成'])
        
        # 3. 根据问题类型和识别的实体进行查询
        if is_fault_question:
            # 故障类问题 - 搜索故障案例
            fault_cases = self.search_fault_cases(question)
            if fault_cases:
                best_case = fault_cases[0]
                result["answer"] = f"根据故障诊断，{best_case['equipment']}出现'{best_case['symptom']}'的原因可能是：{best_case['cause']}。建议解决方案：{best_case['solution']}"
                result["confidence"] = 0.8
                result["sources"] = [{"type": "fault_case", "data": best_case}]
            else:
                result["answer"] = "抱歉，没有找到相关的故障案例。"
                
        elif is_definition_question and entities:
            # 定义类问题 - 查询实体信息
            entity_info = self.query_entity(entities[0])
            if entity_info:
                result["answer"] = f"{entity_info['name']}是{entity_info['type']}，{entity_info['description']}"
                result["confidence"] = 0.8
                result["sources"] = [{"type": "entity", "data": entity_info}]
            else:
                result["answer"] = f"抱歉，没有找到关于'{entities[0]}'的定义信息。"
                
        elif is_relation_question and entities:
            # 关系类问题 - 查询关系信息
            relations = self.find_relations(entities[0])
            if relations:
                objects = [r['object'] for r in relations[:3]]
                result["answer"] = f"{entities[0]}包含或相关的部件有：{', '.join(objects)}"
                result["confidence"] = 0.7
                result["sources"] = [{"type": "relation", "data": relations[0]}]
            else:
                result["answer"] = f"抱歉，没有找到关于'{entities[0]}'的关系信息。"
                
        else:
            # 通用查询 - 先尝试实体，再尝试故障案例
            if entities:
                entity_info = self.query_entity(entities[0])
                if entity_info:
                    result["answer"] = f"找到相关信息：{entity_info['name']}是{entity_info['type']}，{entity_info['description']}"
                    result["confidence"] = 0.6
                    result["sources"] = [{"type": "entity", "data": entity_info}]
                else:
                    # 尝试故障案例
                    fault_cases = self.search_fault_cases(question)
                    if fault_cases:
                        best_case = fault_cases[0]
                        result["answer"] = f"找到相关故障信息：{best_case['equipment']}的{best_case['symptom']}问题"
                        result["confidence"] = 0.5
                        result["sources"] = [{"type": "fault_case", "data": best_case}]
            
            if not result["answer"]:
                result["answer"] = "抱歉，没有找到相关信息。请尝试更具体的描述。"
                result["confidence"] = 0.0
        
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """获取知识库统计信息"""
        with self.driver.session() as session:
            stats = {}
            
            # 实体数量
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats["entities"] = result.single()["count"]
            
            # 关系数量
            result = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as count")
            stats["relations"] = result.single()["count"]
            
            # 设备数量
            result = session.run("MATCH (eq:Equipment) RETURN count(eq) as count")
            stats["equipment"] = result.single()["count"]
            
            # 故障案例数量
            result = session.run("MATCH (s:Symptom) RETURN count(s) as count")
            stats["fault_cases"] = result.single()["count"]
            
            return stats
    
    def clear_database(self):
        """清空数据库（用于测试）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")