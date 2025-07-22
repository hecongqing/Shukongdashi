"""
相似度匹配器
实现基于向量相似度的故障案例匹配
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import logging
from ..models.entities import SimilarCase, FaultElement, UserQuery
from ..utils.text_processor import TextProcessor


class SimilarityMatcher:
    """相似度匹配器"""
    
    def __init__(self, 
                 case_database_path: str = None,
                 vectorizer_path: str = None,
                 text_processor: TextProcessor = None):
        """
        初始化相似度匹配器
        
        Args:
            case_database_path: 案例数据库路径
            vectorizer_path: 向量化器保存路径
            text_processor: 文本处理器
        """
        self.case_database_path = case_database_path
        self.vectorizer_path = vectorizer_path
        self.text_processor = text_processor or TextProcessor()
        self.logger = logging.getLogger(__name__)
        
        # 案例数据库
        self.cases = []
        self.case_vectors = None
        self.vectorizer = None
        
        # 加载已有数据
        self._load_data()
    
    def _load_data(self):
        """加载案例数据和向量化器"""
        try:
            # 加载案例数据库
            if self.case_database_path and os.path.exists(self.case_database_path):
                with open(self.case_database_path, 'rb') as f:
                    self.cases = pickle.load(f)
                self.logger.info(f"加载了 {len(self.cases)} 个案例")
            
            # 加载向量化器和案例向量
            if self.vectorizer_path and os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data.get('vectorizer')
                    self.case_vectors = data.get('case_vectors')
                self.logger.info("成功加载向量化器和案例向量")
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
    
    def _save_data(self):
        """保存案例数据和向量化器"""
        try:
            # 保存案例数据库
            if self.case_database_path:
                os.makedirs(os.path.dirname(self.case_database_path), exist_ok=True)
                with open(self.case_database_path, 'wb') as f:
                    pickle.dump(self.cases, f)
            
            # 保存向量化器和案例向量
            if self.vectorizer_path and self.vectorizer is not None:
                os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
                data = {
                    'vectorizer': self.vectorizer,
                    'case_vectors': self.case_vectors
                }
                with open(self.vectorizer_path, 'wb') as f:
                    pickle.dump(data, f)
                    
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
    
    def add_case(self, case: SimilarCase):
        """
        添加新案例
        
        Args:
            case: 新案例
        """
        self.cases.append(case)
        # 重新构建向量化器和向量矩阵
        self._build_vectors()
    
    def add_cases_batch(self, cases: List[SimilarCase]):
        """
        批量添加案例
        
        Args:
            cases: 案例列表
        """
        self.cases.extend(cases)
        # 重新构建向量化器和向量矩阵
        self._build_vectors()
    
    def _build_vectors(self):
        """构建案例向量表示"""
        if not self.cases:
            return
        
        try:
            # 准备文本数据
            texts = []
            for case in self.cases:
                # 组合案例描述和故障元素内容
                text_parts = [case.description]
                for element in case.elements:
                    text_parts.append(element.content)
                
                # 清理和预处理文本
                combined_text = " ".join(text_parts)
                cleaned_text = self.text_processor.clean_text(combined_text)
                texts.append(cleaned_text)
            
            # 创建或更新TF-IDF向量化器
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=None,  # 我们使用自己的停用词处理
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                self.case_vectors = self.vectorizer.fit_transform(texts)
            else:
                # 更新现有向量化器
                self.case_vectors = self.vectorizer.transform(texts)
            
            self.logger.info(f"构建了 {len(self.cases)} 个案例的向量表示")
            
        except Exception as e:
            self.logger.error(f"构建向量失败: {e}")
    
    def find_similar_cases(self, 
                          query: UserQuery, 
                          top_k: int = 5, 
                          min_similarity: float = 0.1) -> List[SimilarCase]:
        """
        查找相似案例
        
        Args:
            query: 用户查询
            top_k: 返回前k个最相似的案例
            min_similarity: 最小相似度阈值
            
        Returns:
            相似案例列表
        """
        if not self.cases or self.vectorizer is None or self.case_vectors is None:
            return []
        
        try:
            # 准备查询文本
            query_text_parts = [query.fault_description]
            query_text_parts.extend(query.related_phenomena)
            
            # 添加设备信息
            equipment_info = query.equipment_info
            if equipment_info.brand:
                query_text_parts.append(equipment_info.brand)
            if equipment_info.model:
                query_text_parts.append(equipment_info.model)
            if equipment_info.error_code:
                query_text_parts.append(equipment_info.error_code)
            
            # 清理和预处理查询文本
            combined_query = " ".join(query_text_parts)
            cleaned_query = self.text_processor.clean_text(combined_query)
            
            # 将查询转换为向量
            query_vector = self.vectorizer.transform([cleaned_query])
            
            # 计算相似度
            similarities = cosine_similarity(query_vector, self.case_vectors).flatten()
            
            # 获取相似度排序的索引
            similar_indices = np.argsort(similarities)[::-1]
            
            # 构建结果
            similar_cases = []
            for idx in similar_indices[:top_k]:
                similarity = similarities[idx]
                if similarity >= min_similarity:
                    case = self.cases[idx]
                    # 创建新的SimilarCase对象，更新相似度
                    similar_case = SimilarCase(
                        case_id=case.case_id,
                        description=case.description,
                        similarity=float(similarity),
                        elements=case.elements.copy(),
                        solution=case.solution
                    )
                    similar_cases.append(similar_case)
            
            return similar_cases
            
        except Exception as e:
            self.logger.error(f"查找相似案例失败: {e}")
            return []
    
    def calculate_element_similarity(self, 
                                   elements1: List[FaultElement], 
                                   elements2: List[FaultElement]) -> float:
        """
        计算故障元素之间的相似度
        
        Args:
            elements1: 故障元素列表1
            elements2: 故障元素列表2
            
        Returns:
            相似度分数
        """
        if not elements1 or not elements2:
            return 0.0
        
        # 按类型分组元素
        def group_by_type(elements):
            groups = {}
            for element in elements:
                element_type = element.element_type
                if element_type not in groups:
                    groups[element_type] = []
                groups[element_type].append(element.content)
            return groups
        
        groups1 = group_by_type(elements1)
        groups2 = group_by_type(elements2)
        
        # 计算各类型的相似度
        type_similarities = []
        all_types = set(groups1.keys()) | set(groups2.keys())
        
        for element_type in all_types:
            contents1 = groups1.get(element_type, [])
            contents2 = groups2.get(element_type, [])
            
            if not contents1 or not contents2:
                type_similarities.append(0.0)
                continue
            
            # 计算该类型下的最大相似度
            max_sim = 0.0
            for content1 in contents1:
                for content2 in contents2:
                    sim = self.text_processor.calculate_text_similarity(content1, content2)
                    max_sim = max(max_sim, sim)
            
            type_similarities.append(max_sim)
        
        # 返回平均相似度
        return sum(type_similarities) / len(type_similarities) if type_similarities else 0.0
    
    def update_case_feedback(self, case_id: str, feedback_score: float):
        """
        根据用户反馈更新案例质量评分
        
        Args:
            case_id: 案例ID
            feedback_score: 反馈评分 (0-1)
        """
        for case in self.cases:
            if case.case_id == case_id:
                # 这里可以实现更复杂的评分更新逻辑
                # 例如：case.quality_score = (case.quality_score + feedback_score) / 2
                break
    
    def get_case_statistics(self) -> Dict[str, Any]:
        """
        获取案例库统计信息
        
        Returns:
            统计信息字典
        """
        if not self.cases:
            return {"total_cases": 0}
        
        # 统计各类型故障元素的分布
        element_type_counts = {}
        for case in self.cases:
            for element in case.elements:
                element_type = element.element_type.value
                element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1
        
        # 计算平均相似度分布
        similarities = [case.similarity for case in self.cases if hasattr(case, 'similarity')]
        
        stats = {
            "total_cases": len(self.cases),
            "element_type_distribution": element_type_counts,
            "average_similarity": np.mean(similarities) if similarities else 0.0,
            "similarity_std": np.std(similarities) if similarities else 0.0
        }
        
        return stats
    
    def export_cases(self, export_path: str, format: str = "json"):
        """
        导出案例数据
        
        Args:
            export_path: 导出路径
            format: 导出格式 ("json", "csv")
        """
        try:
            import json
            import pandas as pd
            
            if format == "json":
                case_data = [case.to_dict() for case in self.cases]
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, ensure_ascii=False, indent=2)
            
            elif format == "csv":
                # 扁平化案例数据用于CSV导出
                flat_data = []
                for case in self.cases:
                    flat_case = {
                        "case_id": case.case_id,
                        "description": case.description,
                        "solution": case.solution,
                        "similarity": getattr(case, 'similarity', 0.0)
                    }
                    
                    # 添加故障元素信息
                    for i, element in enumerate(case.elements):
                        flat_case[f"element_{i}_content"] = element.content
                        flat_case[f"element_{i}_type"] = element.element_type.value
                        flat_case[f"element_{i}_confidence"] = element.confidence
                    
                    flat_data.append(flat_case)
                
                df = pd.DataFrame(flat_data)
                df.to_csv(export_path, index=False, encoding='utf-8')
            
            self.logger.info(f"成功导出 {len(self.cases)} 个案例到 {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出案例失败: {e}")
    
    def save(self):
        """保存匹配器状态"""
        self._save_data()