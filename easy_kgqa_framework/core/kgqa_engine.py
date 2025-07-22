"""
简洁版KGQA引擎
Easy Knowledge Graph Question Answering Engine
"""

import sqlite3
import re
from typing import List, Dict, Any, Optional
from ..utils.text_utils import TextUtils


class EasyKGQA:
    """简洁版知识图谱问答引擎"""
    
    def __init__(self, db_path: str = "easy_kg.db"):
        """
        初始化KGQA引擎
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self.text_utils = TextUtils()
        self._init_database()
        
    def _init_database(self):
        """初始化数据库，创建基础表结构"""
        with sqlite3.connect(self.db_path) as conn:
            # 实体表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE,
                    type TEXT,
                    description TEXT
                )
            ''')
            
            # 关系表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY,
                    subject_id INTEGER,
                    predicate TEXT,
                    object_id INTEGER,
                    FOREIGN KEY (subject_id) REFERENCES entities (id),
                    FOREIGN KEY (object_id) REFERENCES entities (id)
                )
            ''')
            
            # 故障案例表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fault_cases (
                    id INTEGER PRIMARY KEY,
                    equipment TEXT,
                    symptom TEXT,
                    cause TEXT,
                    solution TEXT
                )
            ''')
            
            conn.commit()
    
    def add_entity(self, name: str, entity_type: str, description: str = ""):
        """添加实体"""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO entities (name, type, description) VALUES (?, ?, ?)",
                    (name, entity_type, description)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # 实体已存在
    
    def add_relation(self, subject: str, predicate: str, obj: str):
        """添加关系"""
        with sqlite3.connect(self.db_path) as conn:
            # 获取实体ID
            subject_id = self._get_entity_id(conn, subject)
            object_id = self._get_entity_id(conn, obj)
            
            if subject_id and object_id:
                conn.execute(
                    "INSERT INTO relations (subject_id, predicate, object_id) VALUES (?, ?, ?)",
                    (subject_id, predicate, object_id)
                )
                conn.commit()
                return True
            return False
    
    def add_fault_case(self, equipment: str, symptom: str, cause: str, solution: str):
        """添加故障案例"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO fault_cases (equipment, symptom, cause, solution) VALUES (?, ?, ?, ?)",
                (equipment, symptom, cause, solution)
            )
            conn.commit()
    
    def _get_entity_id(self, conn, name: str) -> Optional[int]:
        """获取实体ID"""
        cursor = conn.execute("SELECT id FROM entities WHERE name = ?", (name,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def query_entity(self, name: str) -> Optional[Dict]:
        """查询实体信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name, type, description FROM entities WHERE name LIKE ?",
                (f"%{name}%",)
            )
            result = cursor.fetchone()
            if result:
                return {
                    "name": result[0],
                    "type": result[1],
                    "description": result[2]
                }
        return None
    
    def find_relations(self, entity: str) -> List[Dict]:
        """查找实体相关的关系"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT e1.name as subject, r.predicate, e2.name as object
                FROM relations r
                JOIN entities e1 ON r.subject_id = e1.id
                JOIN entities e2 ON r.object_id = e2.id
                WHERE e1.name LIKE ? OR e2.name LIKE ?
            ''', (f"%{entity}%", f"%{entity}%"))
            
            relations = []
            for row in cursor.fetchall():
                relations.append({
                    "subject": row[0],
                    "predicate": row[1],
                    "object": row[2]
                })
            return relations
    
    def search_fault_cases(self, query: str) -> List[Dict]:
        """搜索故障案例"""
        # 简单的关键词匹配
        keywords = self.text_utils.extract_keywords(query)
        
        with sqlite3.connect(self.db_path) as conn:
            # 构建搜索条件
            conditions = []
            params = []
            
            for keyword in keywords:
                conditions.append("(equipment LIKE ? OR symptom LIKE ? OR cause LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            
            if not conditions:
                return []
            
            sql = f"SELECT * FROM fault_cases WHERE {' OR '.join(conditions)}"
            cursor = conn.execute(sql, params)
            
            cases = []
            for row in cursor.fetchall():
                cases.append({
                    "id": row[0],
                    "equipment": row[1],
                    "symptom": row[2],
                    "cause": row[3],
                    "solution": row[4]
                })
            return cases
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题的主要接口"""
        # 简化的问答逻辑
        result = {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "sources": []
        }
        
        # 判断问题类型
        is_fault_question = any(word in question for word in ['怎么办', '故障', '问题', '不转', '不工作', '异常'])
        is_definition_question = any(word in question for word in ['什么是', '是什么', '定义'])
        is_relation_question = any(word in question for word in ['包含', '有什么', '含有', '组成'])
        
        # 1. 尝试实体查询
        keywords = self.text_utils.extract_keywords(question)
        entities_found = []
        
        for keyword in keywords:
            entity = self.query_entity(keyword)
            if entity:
                entities_found.append(entity)
        
        # 2. 查找关系
        relations_found = []
        for keyword in keywords:
            relations = self.find_relations(keyword)
            relations_found.extend(relations)
        
        # 3. 搜索故障案例
        fault_cases = self.search_fault_cases(question)
        
        # 4. 根据问题类型生成答案
        if is_fault_question and fault_cases:
            # 故障类问题优先返回故障案例
            best_case = fault_cases[0]
            result["answer"] = f"根据故障案例，{best_case['equipment']}出现{best_case['symptom']}的原因可能是：{best_case['cause']}。建议解决方案：{best_case['solution']}"
            result["confidence"] = 0.8
            result["sources"] = [{"type": "fault_case", "data": best_case}]
            
        elif is_definition_question and entities_found:
            # 定义类问题优先返回实体信息
            entity = entities_found[0]
            result["answer"] = f"{entity['name']}是{entity['type']}，{entity['description']}"
            result["confidence"] = 0.8
            result["sources"] = [{"type": "entity", "data": entity}]
            
        elif is_relation_question and relations_found:
            # 关系类问题优先返回关系信息
            relations_text = "、".join([f"{r['object']}" for r in relations_found[:3]])
            subject = relations_found[0]['subject']
            result["answer"] = f"{subject}包含或相关的部件有：{relations_text}"
            result["confidence"] = 0.7
            result["sources"] = [{"type": "relation", "data": relations_found[0]}]
            
        elif entities_found:
            # 其他情况，优先返回实体信息
            entity = entities_found[0]
            result["answer"] = f"{entity['name']}是{entity['type']}，{entity['description']}"
            result["confidence"] = 0.6
            result["sources"] = [{"type": "entity", "data": entity}]
            
        elif relations_found:
            # 返回关系信息
            relation = relations_found[0]
            result["answer"] = f"{relation['subject']} {relation['predicate']} {relation['object']}"
            result["confidence"] = 0.5
            result["sources"] = [{"type": "relation", "data": relation}]
            
        elif fault_cases:
            # 最后考虑故障案例
            best_case = fault_cases[0]
            result["answer"] = f"找到相关故障案例：{best_case['equipment']}出现{best_case['symptom']}，原因：{best_case['cause']}，解决方案：{best_case['solution']}"
            result["confidence"] = 0.4
            result["sources"] = [{"type": "fault_case", "data": best_case}]
            
        else:
            result["answer"] = "抱歉，没有找到相关信息。"
            result["confidence"] = 0.0
        
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """获取知识库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # 实体数量
            cursor = conn.execute("SELECT COUNT(*) FROM entities")
            stats["entities"] = cursor.fetchone()[0]
            
            # 关系数量
            cursor = conn.execute("SELECT COUNT(*) FROM relations")
            stats["relations"] = cursor.fetchone()[0]
            
            # 故障案例数量
            cursor = conn.execute("SELECT COUNT(*) FROM fault_cases")
            stats["fault_cases"] = cursor.fetchone()[0]
            
            return stats