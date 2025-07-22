"""
解决方案推荐器
综合知识图谱推理、相似案例匹配等信息，推荐最适合的解决方案
"""

from typing import List, Dict, Any, Tuple
import logging
from collections import Counter, defaultdict
import requests
from bs4 import BeautifulSoup
import time
import random

from ..models.entities import (
    DiagnosisResult, SimilarCase, FaultElement, 
    UserQuery, EquipmentInfo
)


class SolutionRecommender:
    """解决方案推荐器"""
    
    def __init__(self, 
                 enable_web_search: bool = True,
                 web_search_timeout: int = 10):
        """
        初始化解决方案推荐器
        
        Args:
            enable_web_search: 是否启用网络搜索
            web_search_timeout: 网络搜索超时时间
        """
        self.enable_web_search = enable_web_search
        self.web_search_timeout = web_search_timeout
        self.logger = logging.getLogger(__name__)
        
        # 解决方案数据库（可以从外部加载）
        self.solution_database = {
            "电机故障": [
                "检查电机电源连接",
                "检查电机轴承是否损坏",
                "检查电机线圈绝缘",
                "更换损坏的电机组件"
            ],
            "液压系统故障": [
                "检查液压油位和质量",
                "检查液压泵工作状态",
                "检查液压管路是否有泄漏",
                "清洗或更换液压滤芯"
            ],
            "数控系统故障": [
                "重启数控系统",
                "检查系统参数设置",
                "更新系统软件",
                "检查硬件连接"
            ],
            "机械故障": [
                "检查机械传动部件",
                "调整机械间隙",
                "润滑机械部件",
                "更换磨损零件"
            ]
        }
        
        # 常见故障代码对应的解决方案
        self.alarm_solutions = {
            "ALM401": [
                "检查刀库液压系统压力",
                "检查刀链传动机构",
                "调整刀库定位参数",
                "检查刀库电机工作状态"
            ],
            "ALM402": [
                "检查主轴定向功能",
                "检查主轴编码器",
                "调整主轴参数"
            ]
        }
    
    def generate_recommendations(self, 
                               kg_reasoning_result: Dict[str, Any],
                               similar_cases: List[SimilarCase],
                               user_query: UserQuery,
                               fault_elements: List[FaultElement]) -> DiagnosisResult:
        """
        生成综合推荐结果
        
        Args:
            kg_reasoning_result: 知识图谱推理结果
            similar_cases: 相似案例列表
            user_query: 用户查询
            fault_elements: 故障元素列表
            
        Returns:
            诊断结果
        """
        try:
            # 1. 整合故障原因
            causes = self._integrate_causes(kg_reasoning_result, similar_cases)
            
            # 2. 生成解决方案
            solutions = self._generate_solutions(causes, user_query, fault_elements)
            
            # 3. 计算置信度
            confidence = self._calculate_confidence(kg_reasoning_result, similar_cases)
            
            # 4. 生成推理路径
            reasoning_path = self._generate_reasoning_path(kg_reasoning_result, similar_cases)
            
            # 5. 生成进一步检查建议
            recommendations = self._generate_recommendations(causes, fault_elements)
            
            # 6. 如果需要，进行在线搜索补充
            if self.enable_web_search and confidence < 0.7:
                web_solutions = self._search_online_solutions(user_query)
                solutions.extend(web_solutions)
            
            result = DiagnosisResult(
                causes=causes,
                solutions=solutions,
                confidence=confidence,
                reasoning_path=reasoning_path,
                similar_cases=similar_cases,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成推荐结果失败: {e}")
            return DiagnosisResult(
                causes=["分析失败"],
                solutions=["请联系技术支持"],
                confidence=0.0,
                reasoning_path=[],
                similar_cases=[],
                recommendations=[]
            )
    
    def _integrate_causes(self, 
                         kg_result: Dict[str, Any], 
                         similar_cases: List[SimilarCase]) -> List[str]:
        """整合故障原因"""
        cause_scores = defaultdict(float)
        
        # 从知识图谱结果中提取原因
        for cause_info in kg_result.get("causes", []):
            cause = cause_info.get("cause", "")
            confidence = cause_info.get("confidence", 0.0)
            cause_scores[cause] += confidence
        
        # 从相似案例中提取原因（通过分析解决方案推断）
        for case in similar_cases:
            # 简单地将相似度作为权重
            weight = case.similarity * 0.5  # 相似案例权重稍低
            
            # 从解决方案中推断可能的原因
            inferred_causes = self._infer_causes_from_solution(case.solution)
            for cause in inferred_causes:
                cause_scores[cause] += weight
        
        # 按得分排序并返回
        sorted_causes = sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)
        return [cause for cause, score in sorted_causes[:5]]  # 返回前5个原因
    
    def _infer_causes_from_solution(self, solution: str) -> List[str]:
        """从解决方案推断可能的故障原因"""
        inferred_causes = []
        
        # 简单的关键词匹配推断
        if "电机" in solution:
            inferred_causes.append("电机故障")
        if "液压" in solution:
            inferred_causes.append("液压系统故障")
        if "轴承" in solution:
            inferred_causes.append("轴承磨损")
        if "润滑" in solution:
            inferred_causes.append("润滑不良")
        if "温度" in solution:
            inferred_causes.append("温度异常")
        if "参数" in solution:
            inferred_causes.append("参数设置错误")
        
        return inferred_causes
    
    def _generate_solutions(self, 
                          causes: List[str], 
                          user_query: UserQuery, 
                          fault_elements: List[FaultElement]) -> List[str]:
        """生成解决方案"""
        solutions = []
        
        # 1. 基于故障原因生成解决方案
        for cause in causes:
            cause_solutions = self._get_solutions_by_cause(cause)
            solutions.extend(cause_solutions)
        
        # 2. 基于设备信息生成特定解决方案
        equipment_solutions = self._get_solutions_by_equipment(user_query.equipment_info)
        solutions.extend(equipment_solutions)
        
        # 3. 基于故障代码生成解决方案
        if user_query.equipment_info.error_code:
            alarm_solutions = self.alarm_solutions.get(
                user_query.equipment_info.error_code, []
            )
            solutions.extend(alarm_solutions)
        
        # 4. 基于故障元素生成通用解决方案
        element_solutions = self._get_solutions_by_elements(fault_elements)
        solutions.extend(element_solutions)
        
        # 去重并按重要性排序
        unique_solutions = list(dict.fromkeys(solutions))  # 保持顺序的去重
        
        return unique_solutions[:10]  # 返回前10个解决方案
    
    def _get_solutions_by_cause(self, cause: str) -> List[str]:
        """根据故障原因获取解决方案"""
        solutions = []
        
        # 在解决方案数据库中查找匹配的解决方案
        for category, category_solutions in self.solution_database.items():
            if any(keyword in cause for keyword in category.split()):
                solutions.extend(category_solutions)
        
        return solutions
    
    def _get_solutions_by_equipment(self, equipment_info: EquipmentInfo) -> List[str]:
        """根据设备信息生成特定解决方案"""
        solutions = []
        
        # 根据品牌和型号提供特定建议
        if equipment_info.brand:
            brand_lower = equipment_info.brand.lower()
            if "fanuc" in brand_lower or "发那科" in brand_lower:
                solutions.extend([
                    "检查FANUC系统参数设置",
                    "查看FANUC报警历史记录",
                    "重置FANUC系统"
                ])
            elif "siemens" in brand_lower or "西门子" in brand_lower:
                solutions.extend([
                    "检查SIEMENS系统诊断信息",
                    "执行SIEMENS系统自检程序"
                ])
        
        return solutions
    
    def _get_solutions_by_elements(self, fault_elements: List[FaultElement]) -> List[str]:
        """根据故障元素生成解决方案"""
        solutions = []
        
        for element in fault_elements:
            content = element.content.lower()
            
            if "温度" in content or "过热" in content:
                solutions.extend([
                    "检查冷却系统工作状态",
                    "清理散热通道",
                    "检查温度传感器"
                ])
            
            if "振动" in content or "异响" in content:
                solutions.extend([
                    "检查机械连接紧固情况",
                    "检查轴承状态",
                    "调整机械平衡"
                ])
            
            if "停止" in content or "不运行" in content:
                solutions.extend([
                    "检查电源供应",
                    "检查控制信号",
                    "重启设备"
                ])
        
        return solutions
    
    def _calculate_confidence(self, 
                            kg_result: Dict[str, Any], 
                            similar_cases: List[SimilarCase]) -> float:
        """计算整体置信度"""
        confidence_factors = []
        
        # 知识图谱推理置信度
        kg_causes = kg_result.get("causes", [])
        if kg_causes:
            kg_confidence = sum(c.get("confidence", 0.0) for c in kg_causes) / len(kg_causes)
            confidence_factors.append(kg_confidence * 0.6)  # 知识图谱权重0.6
        
        # 相似案例置信度
        if similar_cases:
            similarity_confidence = sum(case.similarity for case in similar_cases) / len(similar_cases)
            confidence_factors.append(similarity_confidence * 0.4)  # 相似案例权重0.4
        
        # 综合置信度
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.1  # 默认最低置信度
    
    def _generate_reasoning_path(self, 
                               kg_result: Dict[str, Any], 
                               similar_cases: List[SimilarCase]) -> List[Dict]:
        """生成推理路径"""
        reasoning_path = []
        
        # 知识图谱推理路径
        reasoning_path.append({
            "step": "知识图谱推理",
            "description": "基于知识图谱进行故障原因分析",
            "results": kg_result.get("causes", [])
        })
        
        # 相似案例匹配路径
        if similar_cases:
            reasoning_path.append({
                "step": "相似案例匹配",
                "description": f"找到 {len(similar_cases)} 个相似案例",
                "results": [case.to_dict() for case in similar_cases[:3]]  # 只显示前3个
            })
        
        return reasoning_path
    
    def _generate_recommendations(self, 
                                causes: List[str], 
                                fault_elements: List[FaultElement]) -> List[str]:
        """生成进一步检查建议"""
        recommendations = []
        
        # 基于故障原因的建议
        for cause in causes[:3]:  # 只考虑前3个原因
            if "电机" in cause:
                recommendations.append("建议检查相关电机的运行状态和参数")
            elif "液压" in cause:
                recommendations.append("建议检查液压系统的压力和油液状态")
            elif "轴承" in cause:
                recommendations.append("建议检查轴承的润滑和磨损情况")
        
        # 通用建议
        recommendations.extend([
            "建议记录故障发生时的具体操作步骤",
            "建议检查设备的日常维护记录",
            "如问题持续，建议联系设备制造商技术支持"
        ])
        
        return list(set(recommendations))  # 去重
    
    def _search_online_solutions(self, user_query: UserQuery) -> List[str]:
        """在线搜索解决方案"""
        solutions = []
        
        if not self.enable_web_search:
            return solutions
        
        try:
            # 构建搜索关键词
            search_keywords = []
            
            if user_query.equipment_info.brand:
                search_keywords.append(user_query.equipment_info.brand)
            if user_query.equipment_info.model:
                search_keywords.append(user_query.equipment_info.model)
            if user_query.equipment_info.error_code:
                search_keywords.append(user_query.equipment_info.error_code)
            
            # 添加故障描述的关键词
            description_keywords = user_query.fault_description[:50]  # 限制长度
            search_keywords.append(description_keywords)
            search_keywords.append("解决方法")
            
            search_query = " ".join(search_keywords)
            
            # 模拟搜索（实际实现中可以使用真实的搜索API）
            mock_solutions = self._mock_web_search(search_query)
            solutions.extend(mock_solutions)
            
        except Exception as e:
            self.logger.error(f"在线搜索失败: {e}")
        
        return solutions
    
    def _mock_web_search(self, query: str) -> List[str]:
        """模拟网络搜索（实际实现中应该使用真实的搜索API）"""
        # 这里返回一些模拟的搜索结果
        mock_results = [
            "根据网络资源：建议检查设备电源系统",
            "根据网络资源：建议更新设备驱动程序",
            "根据网络资源：建议联系设备制造商获取最新技术文档",
            "根据网络资源：建议检查设备的环境条件是否符合要求"
        ]
        
        # 随机返回1-3个结果
        num_results = random.randint(1, min(3, len(mock_results)))
        return random.sample(mock_results, num_results)
    
    def update_solution_database(self, new_solutions: Dict[str, List[str]]):
        """更新解决方案数据库"""
        for category, solutions in new_solutions.items():
            if category in self.solution_database:
                # 合并现有解决方案
                existing_solutions = set(self.solution_database[category])
                new_solution_set = set(solutions)
                self.solution_database[category] = list(existing_solutions | new_solution_set)
            else:
                # 添加新类别
                self.solution_database[category] = solutions
        
        self.logger.info("解决方案数据库已更新")
    
    def add_user_feedback(self, 
                         user_query: UserQuery, 
                         chosen_solution: str, 
                         effectiveness_score: float):
        """添加用户反馈以改进推荐算法"""
        # 这里可以实现基于用户反馈的学习机制
        # 例如：调整解决方案的权重、更新推荐策略等
        
        self.logger.info(f"收到用户反馈：解决方案 '{chosen_solution}' 的有效性评分为 {effectiveness_score}")
        
        # 实际实现中可以将反馈存储到数据库中，用于后续的机器学习优化