"""
故障分析器
整合文本处理、知识图谱推理、相似案例匹配和解决方案推荐的核心组件
"""

import logging
from typing import Optional, Dict, Any
from ..models.entities import UserQuery, DiagnosisResult, EquipmentInfo
from ..utils.text_processor import TextProcessor
from .kg_engine import KnowledgeGraphEngine
from .similarity_matcher import SimilarityMatcher
from .solution_recommender import SolutionRecommender


class FaultAnalyzer:
    """故障分析器 - 系统的核心组件"""
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str,
                 case_database_path: str = None,
                 vectorizer_path: str = None,
                 stopwords_path: str = None,
                 custom_dict_path: str = None,
                 enable_web_search: bool = True,
                 entity_service_url: str = "http://127.0.0.1:50003/extract_entities",
                 enable_entity_recognition: bool = True):
        """
        初始化故障分析器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_username: Neo4j用户名
            neo4j_password: Neo4j密码
            case_database_path: 案例数据库路径
            vectorizer_path: 向量化器路径
            stopwords_path: 停用词文件路径
            custom_dict_path: 自定义词典路径
            enable_web_search: 是否启用网络搜索
            entity_service_url: 实体识别服务URL
            enable_entity_recognition: 是否启用实体识别
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        try:
            # 文本处理器
            self.text_processor = TextProcessor(
                stopwords_path=stopwords_path,
                custom_dict_path=custom_dict_path,
                entity_service_url=entity_service_url,
                enable_entity_recognition=enable_entity_recognition
            )
            
            # 知识图谱引擎
            self.kg_engine = KnowledgeGraphEngine(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password
            )
            
            # 相似度匹配器
            self.similarity_matcher = SimilarityMatcher(
                case_database_path=case_database_path,
                vectorizer_path=vectorizer_path,
                text_processor=self.text_processor
            )
            
            # 解决方案推荐器
            self.solution_recommender = SolutionRecommender(
                enable_web_search=enable_web_search
            )
            
            self.logger.info("故障分析器初始化成功")
            
        except Exception as e:
            self.logger.error(f"故障分析器初始化失败: {e}")
            raise
    
    def analyze_fault(self, 
                     fault_description: str,
                     brand: str = None,
                     model: str = None,
                     error_code: str = None,
                     related_phenomena: list = None,
                     user_feedback: str = None) -> DiagnosisResult:
        """
        分析故障并返回诊断结果
        
        Args:
            fault_description: 故障描述
            brand: 设备品牌
            model: 设备型号
            error_code: 故障代码
            related_phenomena: 相关现象列表
            user_feedback: 用户反馈
            
        Returns:
            诊断结果
        """
        try:
            # 1. 构建用户查询对象
            equipment_info = EquipmentInfo(
                brand=brand,
                model=model,
                error_code=error_code
            )
            
            user_query = UserQuery(
                equipment_info=equipment_info,
                fault_description=fault_description,
                related_phenomena=related_phenomena or [],
                user_feedback=user_feedback
            )
            
            # 2. 文本预处理和故障元素提取
            self.logger.info("开始文本分析...")
            cleaned_description = self.text_processor.clean_text(fault_description)
            sentences = self.text_processor.split_sentences(cleaned_description)
            
            # 提取故障元素
            fault_elements = []
            for sentence in sentences:
                elements = self.text_processor.extract_fault_elements(sentence)
                fault_elements.extend(elements)
            
            # 添加相关现象的故障元素
            for phenomenon in related_phenomena or []:
                phenomenon_elements = self.text_processor.extract_fault_elements(phenomenon)
                fault_elements.extend(phenomenon_elements)
            
            self.logger.info(f"提取到 {len(fault_elements)} 个故障元素")
            
            # 3. 知识图谱推理
            self.logger.info("开始知识图谱推理...")
            kg_reasoning_result = self.kg_engine.execute_reasoning_chain(fault_elements)
            
            # 4. 相似案例匹配
            self.logger.info("开始相似案例匹配...")
            similar_cases = self.similarity_matcher.find_similar_cases(
                query=user_query,
                top_k=5,
                min_similarity=0.1
            )
            
            # 5. 生成综合推荐结果
            self.logger.info("生成解决方案推荐...")
            diagnosis_result = self.solution_recommender.generate_recommendations(
                kg_reasoning_result=kg_reasoning_result,
                similar_cases=similar_cases,
                user_query=user_query,
                fault_elements=fault_elements
            )
            
            self.logger.info("故障分析完成")
            return diagnosis_result
            
        except Exception as e:
            self.logger.error(f"故障分析失败: {e}")
            return DiagnosisResult(
                causes=["分析过程出现错误"],
                solutions=["请检查输入信息或联系技术支持"],
                confidence=0.0,
                reasoning_path=[],
                similar_cases=[],
                recommendations=["建议重新描述故障现象"]
            )
    
    def analyze_fault_from_query(self, user_query: UserQuery) -> DiagnosisResult:
        """
        从用户查询对象分析故障
        
        Args:
            user_query: 用户查询对象
            
        Returns:
            诊断结果
        """
        return self.analyze_fault(
            fault_description=user_query.fault_description,
            brand=user_query.equipment_info.brand,
            model=user_query.equipment_info.model,
            error_code=user_query.equipment_info.error_code,
            related_phenomena=user_query.related_phenomena,
            user_feedback=user_query.user_feedback
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        status = {
            "kg_engine": {
                "connected": False,
                "error": None
            },
            "similarity_matcher": {
                "cases_loaded": 0,
                "vectorizer_ready": False
            },
            "text_processor": {
                "ready": True
            },
            "entity_recognition": self.text_processor.get_entity_recognition_status()
        }
        
        try:
            # 检查知识图谱引擎状态
            status["kg_engine"]["connected"] = self.kg_engine.test_connection()
        except Exception as e:
            status["kg_engine"]["error"] = str(e)
        
        try:
            # 检查相似度匹配器状态
            stats = self.similarity_matcher.get_case_statistics()
            status["similarity_matcher"]["cases_loaded"] = stats.get("total_cases", 0)
            status["similarity_matcher"]["vectorizer_ready"] = (
                self.similarity_matcher.vectorizer is not None
            )
        except Exception as e:
            status["similarity_matcher"]["error"] = str(e)
        
        return status
    
    def add_user_feedback(self, 
                         user_query: UserQuery, 
                         chosen_solution: str, 
                         effectiveness_score: float):
        """
        添加用户反馈
        
        Args:
            user_query: 原始用户查询
            chosen_solution: 用户选择的解决方案
            effectiveness_score: 有效性评分 (0-1)
        """
        try:
            # 更新解决方案推荐器
            self.solution_recommender.add_user_feedback(
                user_query, chosen_solution, effectiveness_score
            )
            
            # 如果反馈积极，可以考虑将其添加为新的案例
            if effectiveness_score >= 0.8:
                self._add_successful_case(user_query, chosen_solution)
            
            self.logger.info(f"用户反馈已记录：评分 {effectiveness_score}")
            
        except Exception as e:
            self.logger.error(f"记录用户反馈失败: {e}")
    
    def _add_successful_case(self, user_query: UserQuery, solution: str):
        """添加成功案例到案例库"""
        try:
            from ..models.entities import SimilarCase, FaultElement, FaultType
            import uuid
            
            # 提取故障元素
            fault_elements = self.text_processor.extract_fault_elements(
                user_query.fault_description
            )
            
            # 创建新案例
            new_case = SimilarCase(
                case_id=str(uuid.uuid4()),
                description=user_query.fault_description,
                similarity=1.0,  # 新案例默认相似度
                elements=fault_elements,
                solution=solution
            )
            
            # 添加到相似度匹配器
            self.similarity_matcher.add_case(new_case)
            
            self.logger.info("成功案例已添加到案例库")
            
        except Exception as e:
            self.logger.error(f"添加成功案例失败: {e}")
    
    def export_knowledge(self, export_path: str, format: str = "json"):
        """
        导出知识库
        
        Args:
            export_path: 导出路径
            format: 导出格式
        """
        try:
            self.similarity_matcher.export_cases(export_path, format)
            self.logger.info(f"知识库已导出到 {export_path}")
        except Exception as e:
            self.logger.error(f"导出知识库失败: {e}")
    
    def update_solution_database(self, new_solutions: Dict[str, list]):
        """
        更新解决方案数据库
        
        Args:
            new_solutions: 新解决方案字典
        """
        try:
            self.solution_recommender.update_solution_database(new_solutions)
            self.logger.info("解决方案数据库已更新")
        except Exception as e:
            self.logger.error(f"更新解决方案数据库失败: {e}")
    
    def save_state(self):
        """保存系统状态"""
        try:
            self.similarity_matcher.save()
            self.logger.info("系统状态已保存")
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
    
    def close(self):
        """关闭系统并清理资源"""
        try:
            # 保存状态
            self.save_state()
            
            # 关闭数据库连接
            self.kg_engine.close()
            
            self.logger.info("故障分析器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭故障分析器失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()