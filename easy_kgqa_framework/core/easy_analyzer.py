"""
简化版KGQA分析器
整合文本处理、实体识别、知识图谱查询等功能
"""

from typing import List, Dict, Any, Optional
from ..models.entities import FaultElement, AnalysisResult, UserQuery
from ..core.kg_engine import KnowledgeGraphEngine
from ..utils.text_processor import SimpleTextProcessor
from ..utils.entity_service import EntityService
from ..config import config


class EasyAnalyzer:
    """简化版KGQA分析器"""
    
    def __init__(self, 
                 neo4j_uri: str = None,
                 neo4j_username: str = None,
                 neo4j_password: str = None,
                 entity_service_url: str = None):
        """
        初始化分析器
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_username: Neo4j用户名
            neo4j_password: Neo4j密码
            entity_service_url: 实体识别服务URL
        """
        # 使用配置或参数
        self.neo4j_uri = neo4j_uri or config.NEO4J_URI
        self.neo4j_username = neo4j_username or config.NEO4J_USERNAME
        self.neo4j_password = neo4j_password or config.NEO4J_PASSWORD
        
        # 初始化组件
        self.kg_engine = KnowledgeGraphEngine(
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password
        )
        self.text_processor = SimpleTextProcessor()
        self.entity_service = EntityService(entity_service_url)
        
        # 测试连接
        self._test_connections()
    
    def _test_connections(self):
        """测试各服务连接"""
        # 测试Neo4j连接
        if self.kg_engine.test_connection():
            print("✓ Neo4j连接成功")
        else:
            print("✗ Neo4j连接失败")
        
        # 测试实体识别服务
        if self.entity_service.test_service():
            print("✓ 实体识别服务连接成功")
        else:
            print("✗ 实体识别服务连接失败，将使用基础文本处理")
    
    def analyze_question(self, question: str, use_entity_service: bool = True) -> AnalysisResult:
        """
        分析用户问题
        
        Args:
            question: 用户问题
            use_entity_service: 是否使用外部实体识别服务
            
        Returns:
            分析结果
        """
        # 1. 文本清理
        cleaned_text = self.text_processor.clean_text(question)
        
        # 2. 实体提取
        if use_entity_service:
            # 尝试使用外部实体识别服务
            elements = self.entity_service.extract_entities(cleaned_text)
            if not elements:
                # 如果外部服务失败，使用内部处理
                elements = self.text_processor.extract_fault_elements(cleaned_text)
        else:
            # 使用内部文本处理
            elements = self.text_processor.extract_fault_elements(cleaned_text)
        
        # 3. 知识图谱查询
        kg_results = self.kg_engine.query_by_fault_elements(elements)
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(elements, kg_results)
        
        # 5. 构建结果
        result = AnalysisResult(
            question=question,
            elements=elements,
            kg_results=kg_results["relations"],
            reasoning_path=kg_results["reasoning_paths"],
            confidence=confidence
        )
        
        return result
    
    def simple_qa(self, question: str) -> List[Dict]:
        """
        简单问答
        
        Args:
            question: 用户问题
            
        Returns:
            回答结果列表
        """
        return self.kg_engine.simple_qa(question)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        # Neo4j状态
        neo4j_status = self.kg_engine.test_connection()
        kg_stats = self.kg_engine.get_statistics() if neo4j_status else {}
        
        # 实体识别服务状态
        entity_service_status = self.entity_service.test_service()
        
        return {
            "neo4j": {
                "status": "connected" if neo4j_status else "disconnected",
                "uri": self.neo4j_uri,
                "statistics": kg_stats
            },
            "entity_service": {
                "status": "connected" if entity_service_status else "disconnected",
                "url": self.entity_service.service_url
            },
            "system": {
                "version": "1.0.0",
                "mode": "easy_kgqa"
            }
        }
    
    def _calculate_confidence(self, elements: List[FaultElement], kg_results: Dict) -> float:
        """
        计算整体置信度
        
        Args:
            elements: 提取的元素
            kg_results: 知识图谱查询结果
            
        Returns:
            置信度分数 (0-1)
        """
        if not elements:
            return 0.0
        
        # 基于元素置信度
        element_confidence = sum(elem.confidence for elem in elements) / len(elements)
        
        # 基于知识图谱匹配度
        kg_confidence = 0.5  # 基础分数
        if kg_results.get("relations"):
            kg_confidence = min(0.9, 0.5 + len(kg_results["relations"]) * 0.1)
        
        # 综合置信度
        final_confidence = (element_confidence * 0.4 + kg_confidence * 0.6)
        return round(final_confidence, 2)
    
    def close(self):
        """关闭资源"""
        if self.kg_engine:
            self.kg_engine.close()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()