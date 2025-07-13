"""
知识图谱问答系统
结合Neo4j知识图谱和大语言模型实现智能问答
"""
import re
import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import jieba
import jieba.posseg as pseg
from py2neo import Graph
from loguru import logger
import opencc
from datetime import datetime
import redis
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import CONFIG
from knowledge_graph.graph_builder import KnowledgeGraphBuilder


class QuestionParser:
    """问题解析器"""
    
    def __init__(self):
        # 问题类型模式
        self.question_patterns = {
            "definition": [
                r"什么是(.+)",
                r"(.+)是什么",
                r"(.+)的定义",
                r"(.+)的含义"
            ],
            "attribute": [
                r"(.+)的(.+)是什么",
                r"(.+)有什么(.+)",
                r"(.+)的(.+)"
            ],
            "relation": [
                r"(.+)和(.+)的关系",
                r"(.+)与(.+)有什么关系",
                r"(.+)属于(.+)",
                r"(.+)位于(.+)"
            ],
            "count": [
                r"有多少(.+)",
                r"(.+)的数量",
                r"一共有几个(.+)"
            ],
            "comparison": [
                r"(.+)和(.+)的区别",
                r"(.+)与(.+)相比",
                r"比较(.+)和(.+)"
            ],
            "listing": [
                r"列出(.+)",
                r"(.+)包括哪些",
                r"(.+)有哪些"
            ]
        }
        
        # 实体识别模式
        self.entity_patterns = [
            r"《(.+?)》",  # 书名、电影名等
            r""(.+?)"",   # 引号内容
            r"'(.+?)'",   # 单引号内容
        ]
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """解析问题"""
        # 清理问题
        question = self._clean_question(question)
        
        # 识别问题类型
        question_type = self._identify_question_type(question)
        
        # 提取实体
        entities = self._extract_entities(question)
        
        # 提取关键词
        keywords = self._extract_keywords(question)
        
        # 分析语法结构
        syntax_info = self._analyze_syntax(question)
        
        return {
            "original_question": question,
            "question_type": question_type,
            "entities": entities,
            "keywords": keywords,
            "syntax_info": syntax_info
        }
    
    def _clean_question(self, question: str) -> str:
        """清理问题文本"""
        # 移除多余空格
        question = re.sub(r'\s+', ' ', question.strip())
        # 繁简转换
        cc = opencc.OpenCC('t2s')
        question = cc.convert(question)
        return question
    
    def _identify_question_type(self, question: str) -> str:
        """识别问题类型"""
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return qtype
        return "general"
    
    def _extract_entities(self, question: str) -> List[str]:
        """提取实体"""
        entities = []
        
        # 使用正则模式提取
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, question)
            entities.extend(matches)
        
        # 使用词性标注提取专有名词
        words = pseg.cut(question)
        for word, flag in words:
            if flag in ['nr', 'ns', 'nt', 'nz']:  # 人名、地名、机构名、其他专名
                entities.append(word)
        
        return list(set(entities))
    
    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词"""
        # 分词
        words = jieba.cut(question)
        
        # 过滤停用词和标点
        stopwords = {'的', '是', '在', '有', '和', '与', '了', '吗', '呢', '什么', '哪些', '如何', '怎么'}
        keywords = [word for word in words if word not in stopwords and len(word) > 1]
        
        return keywords
    
    def _analyze_syntax(self, question: str) -> Dict[str, Any]:
        """分析语法结构"""
        words = list(pseg.cut(question))
        
        # 提取主语、谓语、宾语
        subjects = []
        predicates = []
        objects = []
        
        for word, flag in words:
            if flag in ['n', 'nr', 'ns', 'nt', 'nz']:
                subjects.append(word)
            elif flag in ['v', 'vd', 'vn']:
                predicates.append(word)
            elif flag in ['n', 'nr', 'ns', 'nt', 'nz'] and predicates:
                objects.append(word)
        
        return {
            "subjects": subjects,
            "predicates": predicates,
            "objects": objects
        }


class CypherGenerator:
    """Cypher查询生成器"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        
        # 模板映射
        self.query_templates = {
            "definition": self._generate_definition_query,
            "attribute": self._generate_attribute_query,
            "relation": self._generate_relation_query,
            "count": self._generate_count_query,
            "listing": self._generate_listing_query,
            "general": self._generate_general_query
        }
    
    def generate_cypher(self, parsed_question: Dict[str, Any]) -> List[str]:
        """生成Cypher查询"""
        question_type = parsed_question["question_type"]
        generator = self.query_templates.get(question_type, self._generate_general_query)
        
        queries = generator(parsed_question)
        return queries if isinstance(queries, list) else [queries]
    
    def _generate_definition_query(self, parsed_question: Dict[str, Any]) -> str:
        """生成定义查询"""
        entities = parsed_question["entities"]
        
        if entities:
            entity = entities[0]
            return f"""
            MATCH (n)
            WHERE n.name = '{entity}' OR n.name CONTAINS '{entity}'
            RETURN n.name as entity, labels(n) as types, 
                   n.description as description, n as properties
            LIMIT 5
            """
        else:
            keywords = parsed_question["keywords"]
            if keywords:
                keyword = keywords[0]
                return f"""
                MATCH (n)
                WHERE n.name CONTAINS '{keyword}'
                RETURN n.name as entity, labels(n) as types,
                       n.description as description
                LIMIT 5
                """
        
        return "MATCH (n) RETURN n LIMIT 1"
    
    def _generate_attribute_query(self, parsed_question: Dict[str, Any]) -> str:
        """生成属性查询"""
        entities = parsed_question["entities"]
        keywords = parsed_question["keywords"]
        
        if entities and len(entities) >= 1:
            entity = entities[0]
            if len(keywords) >= 2:
                attribute = keywords[-1]  # 通常属性是最后一个关键词
                return f"""
                MATCH (n)
                WHERE n.name = '{entity}' OR n.name CONTAINS '{entity}'
                RETURN n.name as entity, n.{attribute} as {attribute},
                       n as properties
                LIMIT 5
                """
            else:
                return f"""
                MATCH (n)
                WHERE n.name = '{entity}' OR n.name CONTAINS '{entity}'
                RETURN n.name as entity, n as properties
                LIMIT 5
                """
        
        return "MATCH (n) RETURN n LIMIT 1"
    
    def _generate_relation_query(self, parsed_question: Dict[str, Any]) -> str:
        """生成关系查询"""
        entities = parsed_question["entities"]
        
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            return f"""
            MATCH (a)-[r]-(b)
            WHERE (a.name CONTAINS '{entity1}' AND b.name CONTAINS '{entity2}')
               OR (a.name CONTAINS '{entity2}' AND b.name CONTAINS '{entity1}')
            RETURN a.name as entity1, type(r) as relation, b.name as entity2,
                   r.confidence as confidence
            LIMIT 10
            """
        elif len(entities) == 1:
            entity = entities[0]
            return f"""
            MATCH (n)-[r]-(m)
            WHERE n.name CONTAINS '{entity}'
            RETURN n.name as entity1, type(r) as relation, m.name as entity2,
                   r.confidence as confidence
            LIMIT 10
            """
        
        return "MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 5"
    
    def _generate_count_query(self, parsed_question: Dict[str, Any]) -> str:
        """生成计数查询"""
        keywords = parsed_question["keywords"]
        
        if keywords:
            keyword = keywords[0]
            return f"""
            MATCH (n)
            WHERE n.name CONTAINS '{keyword}' OR '{keyword}' IN labels(n)
            RETURN count(n) as count, labels(n) as types
            """
        
        return "MATCH (n) RETURN count(n) as total_count"
    
    def _generate_listing_query(self, parsed_question: Dict[str, Any]) -> str:
        """生成列表查询"""
        keywords = parsed_question["keywords"]
        entities = parsed_question["entities"]
        
        if entities:
            entity = entities[0]
            return f"""
            MATCH (n)-[r]-(m)
            WHERE n.name CONTAINS '{entity}'
            RETURN DISTINCT m.name as related_entities, type(r) as relation
            LIMIT 20
            """
        elif keywords:
            keyword = keywords[0]
            return f"""
            MATCH (n)
            WHERE n.name CONTAINS '{keyword}' OR '{keyword}' IN labels(n)
            RETURN n.name as entity, labels(n) as types
            LIMIT 20
            """
        
        return "MATCH (n) RETURN n.name as entity LIMIT 20"
    
    def _generate_general_query(self, parsed_question: Dict[str, Any]) -> List[str]:
        """生成通用查询"""
        entities = parsed_question["entities"]
        keywords = parsed_question["keywords"]
        
        queries = []
        
        # 基于实体的查询
        for entity in entities:
            queries.append(f"""
            MATCH (n)
            WHERE n.name CONTAINS '{entity}'
            RETURN n.name as entity, labels(n) as types, n as properties
            LIMIT 5
            """)
        
        # 基于关键词的查询
        for keyword in keywords:
            queries.append(f"""
            MATCH (n)
            WHERE n.name CONTAINS '{keyword}'
            RETURN n.name as entity, labels(n) as types
            LIMIT 5
            """)
        
        # 如果没有特定实体或关键词，返回通用查询
        if not queries:
            queries.append("MATCH (n) RETURN n.name as entity LIMIT 10")
        
        return queries


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, llm_config: Dict[str, Any] = None):
        self.llm_config = llm_config or CONFIG["model"]["llm"]
        
        # 初始化本地大模型
        self._init_llm()
        
        # 答案模板
        self.answer_templates = {
            "definition": "根据知识图谱，{entity}是{description}。",
            "attribute": "{entity}的{attribute}是{value}。",
            "relation": "{entity1}和{entity2}的关系是：{relation}。",
            "count": "根据知识图谱统计，{type}共有{count}个。",
            "listing": "相关的实体包括：{entities}。",
            "no_result": "抱歉，我在知识图谱中没有找到相关信息。",
            "general": "根据知识图谱，以下是相关信息：{information}。"
        }
    
    def _init_llm(self):
        """初始化大语言模型"""
        try:
            # 这里可以根据配置加载不同的模型
            model_name = self.llm_config.get("model_name", "chatglm3-6b")
            model_path = self.llm_config.get("model_path")
            
            if model_path and Path(model_path).exists():
                logger.info(f"Loading local LLM from {model_path}")
                # 加载本地模型
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                self.model.eval()
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.has_llm = True
                logger.info("Local LLM loaded successfully")
            else:
                logger.warning("Local LLM not available, using template-based generation")
                self.has_llm = False
                
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            self.has_llm = False
    
    def generate_answer(self, parsed_question: Dict[str, Any], 
                       query_results: List[Dict[str, Any]]) -> str:
        """生成答案"""
        if not query_results:
            return self.answer_templates["no_result"]
        
        question_type = parsed_question["question_type"]
        
        # 使用大模型生成答案
        if self.has_llm:
            return self._generate_with_llm(parsed_question, query_results)
        else:
            # 使用模板生成答案
            return self._generate_with_template(question_type, query_results)
    
    def _generate_with_llm(self, parsed_question: Dict[str, Any], 
                          query_results: List[Dict[str, Any]]) -> str:
        """使用大模型生成答案"""
        try:
            # 构建提示词
            question = parsed_question["original_question"]
            context = self._format_context(query_results)
            
            prompt = f"""
            基于以下知识图谱信息，回答用户的问题。请确保答案准确、简洁、有用。

            知识图谱信息：
            {context}

            用户问题：{question}

            回答：
            """
            
            # 使用模型生成答案
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            # 解码答案
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分
            if "回答：" in answer:
                answer = answer.split("回答：")[-1].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            # 回退到模板方法
            return self._generate_with_template(parsed_question["question_type"], query_results)
    
    def _generate_with_template(self, question_type: str, 
                               query_results: List[Dict[str, Any]]) -> str:
        """使用模板生成答案"""
        if question_type == "definition" and query_results:
            result = query_results[0]
            entity = result.get("entity", "")
            description = result.get("description", "")
            
            if description:
                return self.answer_templates["definition"].format(
                    entity=entity, description=description
                )
            else:
                # 基于属性构建描述
                properties = result.get("properties", {})
                if isinstance(properties, dict):
                    props_text = ", ".join([f"{k}: {v}" for k, v in properties.items() if v])
                    return f"{entity}的相关信息：{props_text}。"
        
        elif question_type == "attribute" and query_results:
            result = query_results[0]
            entity = result.get("entity", "")
            
            # 查找非空属性
            for key, value in result.items():
                if key not in ["entity", "types", "properties"] and value:
                    return self.answer_templates["attribute"].format(
                        entity=entity, attribute=key, value=value
                    )
        
        elif question_type == "relation" and query_results:
            relations = []
            for result in query_results[:5]:  # 限制结果数量
                entity1 = result.get("entity1", "")
                entity2 = result.get("entity2", "")
                relation = result.get("relation", "")
                if entity1 and entity2 and relation:
                    relations.append(f"{entity1} {relation} {entity2}")
            
            if relations:
                return f"找到以下关系：{'; '.join(relations)}。"
        
        elif question_type == "count" and query_results:
            result = query_results[0]
            count = result.get("count", 0)
            types = result.get("types", [])
            
            if types:
                type_name = types[0] if isinstance(types, list) else str(types)
                return self.answer_templates["count"].format(type=type_name, count=count)
            else:
                return f"总计找到{count}个相关实体。"
        
        elif question_type == "listing" and query_results:
            entities = []
            for result in query_results[:10]:  # 限制数量
                entity = result.get("related_entities") or result.get("entity", "")
                if entity:
                    entities.append(entity)
            
            if entities:
                return self.answer_templates["listing"].format(entities="、".join(entities))
        
        # 通用情况
        info_parts = []
        for result in query_results[:5]:
            entity = result.get("entity", "")
            if entity:
                info_parts.append(entity)
        
        if info_parts:
            return self.answer_templates["general"].format(information="、".join(info_parts))
        
        return self.answer_templates["no_result"]
    
    def _format_context(self, query_results: List[Dict[str, Any]]) -> str:
        """格式化知识图谱查询结果为上下文"""
        context_parts = []
        
        for i, result in enumerate(query_results[:10]):  # 限制上下文长度
            if isinstance(result, dict):
                parts = []
                for key, value in result.items():
                    if value and key not in ["properties"]:
                        if isinstance(value, (list, tuple)):
                            value = ", ".join(str(v) for v in value)
                        parts.append(f"{key}: {value}")
                
                if parts:
                    context_parts.append(f"{i+1}. {'; '.join(parts)}")
        
        return "\n".join(context_parts)


class KnowledgeGraphQA:
    """知识图谱问答系统主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG
        
        # 初始化组件
        self.graph_builder = KnowledgeGraphBuilder()
        self.question_parser = QuestionParser()
        self.cypher_generator = CypherGenerator(self.graph_builder.graph)
        self.answer_generator = AnswerGenerator()
        
        # 初始化缓存
        self._init_cache()
        
        logger.info("Knowledge Graph QA System initialized")
    
    def _init_cache(self):
        """初始化缓存系统"""
        try:
            cache_config = self.config["database"]["redis"]
            self.cache = redis.from_url(
                cache_config["url"],
                decode_responses=cache_config["decode_responses"]
            )
            self.use_cache = True
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Cache initialization failed: {e}")
            self.use_cache = False
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题"""
        start_time = datetime.now()
        
        # 检查缓存
        if self.use_cache:
            cached_answer = self._get_cached_answer(question)
            if cached_answer:
                return {
                    "question": question,
                    "answer": cached_answer,
                    "source": "cache",
                    "response_time": (datetime.now() - start_time).total_seconds()
                }
        
        try:
            # 解析问题
            parsed_question = self.question_parser.parse_question(question)
            logger.info(f"Parsed question: {parsed_question}")
            
            # 生成Cypher查询
            cypher_queries = self.cypher_generator.generate_cypher(parsed_question)
            logger.info(f"Generated {len(cypher_queries)} Cypher queries")
            
            # 执行查询
            all_results = []
            for query in cypher_queries:
                try:
                    results = self.graph_builder.query_graph(query)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error executing query: {e}")
            
            logger.info(f"Query results: {len(all_results)} records found")
            
            # 生成答案
            answer = self.answer_generator.generate_answer(parsed_question, all_results)
            
            # 缓存答案
            if self.use_cache and answer:
                self._cache_answer(question, answer)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "question": question,
                "answer": answer,
                "parsed_question": parsed_question,
                "query_results": all_results,
                "cypher_queries": cypher_queries,
                "source": "knowledge_graph",
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": "抱歉，处理您的问题时出现了错误。",
                "error": str(e),
                "response_time": (datetime.now() - start_time).total_seconds()
            }
    
    def _get_cached_answer(self, question: str) -> Optional[str]:
        """获取缓存的答案"""
        try:
            cache_key = f"qa:{hash(question)}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("Answer retrieved from cache")
                return cached
        except Exception as e:
            logger.error(f"Error getting cached answer: {e}")
        return None
    
    def _cache_answer(self, question: str, answer: str):
        """缓存答案"""
        try:
            cache_key = f"qa:{hash(question)}"
            cache_ttl = self.config["qa"]["cache_ttl"]
            self.cache.setex(cache_key, cache_ttl, answer)
            logger.info("Answer cached successfully")
        except Exception as e:
            logger.error(f"Error caching answer: {e}")
    
    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量回答问题"""
        answers = []
        for question in questions:
            answer = self.answer_question(question)
            answers.append(answer)
        return answers
    
    def get_similar_questions(self, question: str, threshold: float = 0.8) -> List[str]:
        """获取相似问题"""
        # 这里可以实现基于向量相似度的问题推荐
        # 暂时返回空列表
        return []


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph QA System")
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--batch-file", help="File containing questions to answer")
    
    args = parser.parse_args()
    
    # 初始化QA系统
    qa_system = KnowledgeGraphQA()
    
    if args.question:
        # 单个问题
        result = qa_system.answer_question(args.question)
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"响应时间: {result['response_time']:.2f}秒")
    
    elif args.batch_file:
        # 批量问题
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = qa_system.batch_answer(questions)
        
        for result in results:
            print(f"问题: {result['question']}")
            print(f"答案: {result['answer']}")
            print("-" * 50)
    
    elif args.interactive:
        # 交互模式
        print("知识图谱问答系统 (输入 'exit' 退出)")
        print("-" * 50)
        
        while True:
            question = input("请输入您的问题: ").strip()
            
            if question.lower() in ['exit', '退出', 'quit']:
                break
            
            if not question:
                continue
            
            try:
                result = qa_system.answer_question(question)
                print(f"答案: {result['answer']}")
                print(f"响应时间: {result['response_time']:.2f}秒")
                print("-" * 50)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()