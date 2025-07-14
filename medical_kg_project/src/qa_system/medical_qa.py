"""
基于知识图谱的医疗问答系统
"""
import re
import jieba
import jieba.posseg as pseg
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json

from ..kg_builder.neo4j_manager import MedicalGraphBuilder, GraphConfig
from ..models.llm_extractor import MedicalLLMExtractor


class IntentType(Enum):
    """问题意图类型"""
    DISEASE_SYMPTOM = "disease_symptom"  # 疾病有什么症状
    SYMPTOM_DISEASE = "symptom_disease"  # 症状对应什么疾病
    DISEASE_TREATMENT = "disease_treatment"  # 疾病如何治疗
    DISEASE_DRUG = "disease_drug"  # 疾病用什么药
    DISEASE_EXAMINATION = "disease_examination"  # 疾病需要做什么检查
    DISEASE_PREVENTION = "disease_prevention"  # 疾病如何预防
    DISEASE_CAUSE = "disease_cause"  # 疾病的病因
    DISEASE_COMPLICATION = "disease_complication"  # 疾病的并发症
    DRUG_INFO = "drug_info"  # 药物信息
    GENERAL_CONSULTATION = "general_consultation"  # 一般咨询


@dataclass
class QueryIntent:
    """查询意图"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, List[str]]  # 实体类型 -> 实体列表


@dataclass
class Answer:
    """答案"""
    text: str
    data: Optional[Dict] = None
    confidence: float = 1.0
    source: str = "knowledge_graph"


class MedicalIntentClassifier:
    """医疗意图分类器"""
    
    def __init__(self):
        # 意图关键词模式
        self.intent_patterns = {
            IntentType.DISEASE_SYMPTOM: [
                r"(.+)有什么症状", r"(.+)的症状", r"(.+)表现",
                r"(.+)有哪些表现", r"(.+)会出现什么"
            ],
            IntentType.SYMPTOM_DISEASE: [
                r"(.+)是什么病", r"(.+)可能是什么病",
                r"出现(.+)是怎么回事", r"(.+)是什么原因"
            ],
            IntentType.DISEASE_TREATMENT: [
                r"(.+)怎么治疗", r"(.+)如何治疗", r"(.+)治疗方法",
                r"得了(.+)怎么办", r"(.+)怎么处理"
            ],
            IntentType.DISEASE_DRUG: [
                r"(.+)吃什么药", r"(.+)用什么药", r"(.+)的药物",
                r"(.+)推荐用药"
            ],
            IntentType.DISEASE_EXAMINATION: [
                r"(.+)做什么检查", r"(.+)需要检查什么", r"(.+)检查项目",
                r"怀疑(.+)要做什么检查"
            ],
            IntentType.DISEASE_PREVENTION: [
                r"(.+)如何预防", r"(.+)怎么预防", r"预防(.+)",
                r"怎样避免(.+)"
            ],
            IntentType.DISEASE_CAUSE: [
                r"(.+)的病因", r"(.+)是什么引起的", r"为什么会得(.+)",
                r"(.+)的原因"
            ],
            IntentType.DISEASE_COMPLICATION: [
                r"(.+)的并发症", r"(.+)会引起什么", r"(.+)会导致什么"
            ],
            IntentType.DRUG_INFO: [
                r"(.+)是什么药", r"(.+)的作用", r"(.+)的副作用",
                r"(.+)怎么用"
            ]
        }
        
        # 加载自定义词典
        self._load_medical_dict()
    
    def _load_medical_dict(self):
        """加载医学词典"""
        # 这里可以加载自定义的医学词典
        medical_words = [
            '糖尿病', '高血压', '冠心病', '肺炎', '胃炎',
            '发热', '咳嗽', '头痛', '腹痛', '胸闷',
            '阿司匹林', '布洛芬', '青霉素', '胰岛素'
        ]
        for word in medical_words:
            jieba.add_word(word)
    
    def classify(self, question: str) -> QueryIntent:
        """
        分类用户问题意图
        
        Args:
            question: 用户问题
            
        Returns:
            查询意图
        """
        # 分词和词性标注
        words = list(pseg.cut(question))
        
        # 尝试匹配意图模式
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    # 提取实体
                    entities = self._extract_entities(question, words)
                    
                    return QueryIntent(
                        intent_type=intent_type,
                        confidence=0.9,
                        entities=entities
                    )
        
        # 默认为一般咨询
        entities = self._extract_entities(question, words)
        return QueryIntent(
            intent_type=IntentType.GENERAL_CONSULTATION,
            confidence=0.5,
            entities=entities
        )
    
    def _extract_entities(self, question: str, words: List) -> Dict[str, List[str]]:
        """提取实体"""
        entities = {
            'disease': [],
            'symptom': [],
            'drug': [],
            'examination': [],
            'body_part': []
        }
        
        # 基于词性和规则的实体识别
        for word, pos in words:
            # 这里可以根据词性或词典进行更精确的识别
            if pos in ['n', 'nr', 'nz', 'nt']:
                # 简单规则：包含"病"的可能是疾病
                if '病' in word or '症' in word:
                    entities['disease'].append(word)
                elif '药' in word or '素' in word:
                    entities['drug'].append(word)
                # 可以添加更多规则
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities


class CypherQueryBuilder:
    """Cypher查询构建器"""
    
    def __init__(self):
        # 查询模板
        self.query_templates = {
            IntentType.DISEASE_SYMPTOM: """
                MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
                RETURN d.name as disease, collect(s.name) as symptoms
            """,
            IntentType.SYMPTOM_DISEASE: """
                MATCH (s:Symptom {name: $symptom})<-[:HAS_SYMPTOM]-(d:Disease)
                RETURN s.name as symptom, collect(d.name) as diseases
            """,
            IntentType.DISEASE_TREATMENT: """
                MATCH (d:Disease {name: $disease})-[:TREATED_BY]->(t)
                WHERE labels(t)[0] IN ['Drug', 'Treatment']
                RETURN d.name as disease, 
                       collect(DISTINCT {name: t.name, type: labels(t)[0]}) as treatments
            """,
            IntentType.DISEASE_DRUG: """
                MATCH (d:Disease {name: $disease})-[:TREATED_BY]->(drug:Drug)
                RETURN d.name as disease, 
                       collect({name: drug.name, usage: drug.usage, type: drug.type}) as drugs
            """,
            IntentType.DISEASE_EXAMINATION: """
                MATCH (d:Disease {name: $disease})-[:EXAMINED_BY]->(e:Examination)
                RETURN d.name as disease, collect(e.name) as examinations
            """,
            IntentType.DISEASE_CAUSE: """
                MATCH (d:Disease {name: $disease})-[:CAUSED_BY]->(c)
                RETURN d.name as disease, collect(c.name) as causes
            """,
            IntentType.DISEASE_COMPLICATION: """
                MATCH (d:Disease {name: $disease})-[:COMPLICATION_OF]->(c:Disease)
                RETURN d.name as disease, collect(c.name) as complications
            """,
            IntentType.DRUG_INFO: """
                MATCH (drug:Drug {name: $drug})
                OPTIONAL MATCH (drug)<-[:TREATED_BY]-(d:Disease)
                RETURN drug, collect(d.name) as diseases
            """
        }
    
    def build(self, intent: QueryIntent) -> Tuple[str, Dict]:
        """
        构建Cypher查询
        
        Args:
            intent: 查询意图
            
        Returns:
            (Cypher查询语句, 参数字典)
        """
        query_template = self.query_templates.get(intent.intent_type)
        if not query_template:
            raise ValueError(f"No template for intent: {intent.intent_type}")
        
        # 构建参数
        params = {}
        if intent.intent_type == IntentType.DISEASE_SYMPTOM:
            if intent.entities.get('disease'):
                params['disease'] = intent.entities['disease'][0]
        elif intent.intent_type == IntentType.SYMPTOM_DISEASE:
            if intent.entities.get('symptom'):
                params['symptom'] = intent.entities['symptom'][0]
        elif intent.intent_type in [
            IntentType.DISEASE_TREATMENT,
            IntentType.DISEASE_DRUG,
            IntentType.DISEASE_EXAMINATION,
            IntentType.DISEASE_CAUSE,
            IntentType.DISEASE_COMPLICATION
        ]:
            if intent.entities.get('disease'):
                params['disease'] = intent.entities['disease'][0]
        elif intent.intent_type == IntentType.DRUG_INFO:
            if intent.entities.get('drug'):
                params['drug'] = intent.entities['drug'][0]
        
        return query_template, params


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.llm = MedicalLLMExtractor(model_name='chatglm3-6b')
        
        # 答案模板
        self.answer_templates = {
            IntentType.DISEASE_SYMPTOM: "{disease}的常见症状包括：{symptoms}。",
            IntentType.SYMPTOM_DISEASE: "出现{symptom}可能与以下疾病有关：{diseases}。建议及时就医进行详细检查。",
            IntentType.DISEASE_TREATMENT: "{disease}的治疗方法包括：{treatments}。具体治疗方案需要医生根据病情制定。",
            IntentType.DISEASE_DRUG: "{disease}的常用药物有：{drugs}。请在医生指导下用药。",
            IntentType.DISEASE_EXAMINATION: "{disease}需要进行的检查包括：{examinations}。",
            IntentType.DISEASE_CAUSE: "{disease}的病因包括：{causes}。",
            IntentType.DISEASE_COMPLICATION: "{disease}可能出现的并发症有：{complications}。",
            IntentType.DRUG_INFO: "{drug_info}"
        }
    
    def generate(self, 
                intent: QueryIntent, 
                query_result: List[Dict],
                original_question: str) -> Answer:
        """
        生成答案
        
        Args:
            intent: 查询意图
            query_result: 知识图谱查询结果
            original_question: 原始问题
            
        Returns:
            答案
        """
        if not query_result:
            return Answer(
                text="抱歉，我暂时无法找到相关信息。建议您咨询专业医生。",
                confidence=0.3
            )
        
        # 使用模板生成基础答案
        template_answer = self._generate_template_answer(intent, query_result)
        
        # 如果启用LLM，使用LLM优化答案
        if self.use_llm:
            final_answer = self._enhance_with_llm(
                original_question, 
                template_answer, 
                query_result
            )
        else:
            final_answer = template_answer
        
        # 添加免责声明
        final_answer += "\n\n提示：以上信息仅供参考，具体诊疗请咨询专业医生。"
        
        return Answer(
            text=final_answer,
            data=query_result,
            confidence=0.8
        )
    
    def _generate_template_answer(self, 
                                 intent: QueryIntent, 
                                 query_result: List[Dict]) -> str:
        """使用模板生成答案"""
        template = self.answer_templates.get(intent.intent_type)
        if not template:
            return str(query_result)
        
        # 处理查询结果
        result = query_result[0] if query_result else {}
        
        # 格式化数据
        if intent.intent_type == IntentType.DISEASE_SYMPTOM:
            symptoms = result.get('symptoms', [])
            return template.format(
                disease=result.get('disease', ''),
                symptoms='、'.join(symptoms[:5])  # 限制显示数量
            )
        elif intent.intent_type == IntentType.SYMPTOM_DISEASE:
            diseases = result.get('diseases', [])
            return template.format(
                symptom=result.get('symptom', ''),
                diseases='、'.join(diseases[:5])
            )
        elif intent.intent_type == IntentType.DISEASE_TREATMENT:
            treatments = result.get('treatments', [])
            treatment_list = []
            for t in treatments[:5]:
                if isinstance(t, dict):
                    treatment_list.append(t.get('name', ''))
                else:
                    treatment_list.append(str(t))
            return template.format(
                disease=result.get('disease', ''),
                treatments='、'.join(treatment_list)
            )
        elif intent.intent_type == IntentType.DISEASE_DRUG:
            drugs = result.get('drugs', [])
            drug_list = []
            for d in drugs[:5]:
                if isinstance(d, dict):
                    drug_info = d.get('name', '')
                    if d.get('usage'):
                        drug_info += f"（{d['usage']}）"
                    drug_list.append(drug_info)
                else:
                    drug_list.append(str(d))
            return template.format(
                disease=result.get('disease', ''),
                drugs='、'.join(drug_list)
            )
        else:
            # 其他类型的简单处理
            return str(result)
    
    def _enhance_with_llm(self, 
                         question: str, 
                         template_answer: str, 
                         query_result: List[Dict]) -> str:
        """使用LLM增强答案"""
        prompt = f"""
请基于以下信息，为用户问题提供一个专业、准确、易懂的医疗回答。

用户问题：{question}
知识图谱数据：{json.dumps(query_result, ensure_ascii=False)}
初步答案：{template_answer}

要求：
1. 回答要准确、专业
2. 语言要通俗易懂
3. 适当补充相关建议
4. 不要编造不存在的信息

回答：
"""
        
        try:
            response = self.llm._generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return template_answer


class MedicalQASystem:
    """医疗问答系统主类"""
    
    def __init__(self, graph_config: GraphConfig, use_llm: bool = True):
        """
        初始化问答系统
        
        Args:
            graph_config: 图数据库配置
            use_llm: 是否使用大模型增强
        """
        self.graph_builder = MedicalGraphBuilder(graph_config)
        self.intent_classifier = MedicalIntentClassifier()
        self.query_builder = CypherQueryBuilder()
        self.answer_generator = AnswerGenerator(use_llm=use_llm)
        
        logger.info("Medical QA System initialized")
    
    def answer(self, question: str) -> Answer:
        """
        回答用户问题
        
        Args:
            question: 用户问题
            
        Returns:
            答案
        """
        logger.info(f"Received question: {question}")
        
        try:
            # 1. 意图识别
            intent = self.intent_classifier.classify(question)
            logger.info(f"Identified intent: {intent.intent_type.value}")
            
            # 2. 构建查询
            if intent.intent_type == IntentType.GENERAL_CONSULTATION:
                # 对于一般咨询，直接使用LLM回答
                return self._handle_general_consultation(question)
            
            cypher_query, params = self.query_builder.build(intent)
            logger.info(f"Built query with params: {params}")
            
            # 3. 执行查询
            query_result = self.graph_builder.query(cypher_query, params)
            logger.info(f"Query returned {len(query_result)} results")
            
            # 4. 生成答案
            answer = self.answer_generator.generate(intent, query_result, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return Answer(
                text="抱歉，处理您的问题时出现了错误。请稍后再试或咨询专业医生。",
                confidence=0.1
            )
    
    def _handle_general_consultation(self, question: str) -> Answer:
        """处理一般咨询"""
        # 这里可以使用LLM直接回答或返回默认响应
        return Answer(
            text="这是一个一般性的医疗咨询问题。建议您提供更具体的症状或疾病信息，以便我给出更准确的建议。如有紧急情况，请立即就医。",
            confidence=0.5,
            source="general"
        )
    
    def batch_answer(self, questions: List[str]) -> List[Answer]:
        """批量回答问题"""
        answers = []
        for question in questions:
            answer = self.answer(question)
            answers.append(answer)
        return answers
    
    def close(self):
        """关闭资源"""
        self.graph_builder.close()
        if hasattr(self.answer_generator, 'llm'):
            # 清理LLM资源
            del self.answer_generator.llm


# Web API接口
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="医疗问答系统API")

# 全局问答系统实例
qa_system = None


class QuestionRequest(BaseModel):
    """问题请求"""
    question: str
    user_id: Optional[str] = None


class AnswerResponse(BaseModel):
    """答案响应"""
    answer: str
    confidence: float
    source: str
    data: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global qa_system
    config = GraphConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )
    qa_system = MedicalQASystem(config, use_llm=True)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    global qa_system
    if qa_system:
        qa_system.close()


@app.post("/qa", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """问答接口"""
    if not qa_system:
        raise HTTPException(status_code=503, detail="QA system not initialized")
    
    answer = qa_system.answer(request.question)
    
    return AnswerResponse(
        answer=answer.text,
        confidence=answer.confidence,
        source=answer.source,
        data=answer.data
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗问答系统")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli",
                       help="运行模式：cli(命令行) 或 api(Web服务)")
    parser.add_argument("--host", default="0.0.0.0", help="API服务地址")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        # 命令行交互模式
        config = GraphConfig(
            uri="bolt://localhost:7687",
            username="neo4j", 
            password="your_password"
        )
        
        qa_system = MedicalQASystem(config, use_llm=False)
        
        print("医疗问答系统已启动！输入'退出'或'quit'结束对话。")
        print("-" * 50)
        
        while True:
            question = input("\n请输入您的问题：").strip()
            
            if question in ['退出', 'quit', 'exit']:
                print("感谢使用，再见！")
                break
            
            if not question:
                continue
            
            answer = qa_system.answer(question)
            print(f"\n回答：{answer.text}")
            print(f"置信度：{answer.confidence:.2f}")
        
        qa_system.close()