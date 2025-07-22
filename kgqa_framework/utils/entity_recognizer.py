"""
实体识别器
集成外部实体识别服务，提供更精准的实体提取能力
"""

import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..models.entities import FaultElement, FaultType


class EntityRecognizer:
    """实体识别器"""
    
    def __init__(self, 
                 service_url: str = "http://127.0.0.1:50003/extract_entities",
                 timeout: int = 10,
                 fallback_enabled: bool = True):
        """
        初始化实体识别器
        
        Args:
            service_url: 实体识别服务的URL
            timeout: 请求超时时间（秒）
            fallback_enabled: 是否启用回退模式（使用规则匹配）
        """
        self.service_url = service_url
        self.timeout = timeout
        self.fallback_enabled = fallback_enabled
        self.logger = logging.getLogger(__name__)
        
        # 测试服务连接
        self.service_available = self._test_service()
        
        # 实体类型映射（将NER服务的实体类型映射到故障类型）
        self.entity_type_mapping = {
            # 部件单元 -> 部位
            '部件单元': FaultType.LOCATION,
            'COMPONENT': FaultType.LOCATION,
            
            # 性能表征 -> 现象
            '性能表征': FaultType.PHENOMENON,
            'PERFORMANCE': FaultType.PHENOMENON,
            
            # 故障状态 -> 现象
            '故障状态': FaultType.PHENOMENON,
            'FAULT_STATE': FaultType.PHENOMENON,
            
            # 检测工具 -> 部位（工具也是一种设备部位）
            '检测工具': FaultType.LOCATION,
            'DETECTION_TOOL': FaultType.LOCATION
        }
        
        # 规则匹配的回退模式（与原有逻辑保持一致）
        self.fallback_patterns = {
            FaultType.OPERATION: {
                'keywords': ['启动', '停止', '运行', '操作', '按下', '开启', '关闭', '切换', '调整', '自动换刀'],
                'patterns': [r'(.{0,5})(启动|停止|运行|操作|按下|开启|关闭|切换|调整|自动换刀)(.{0,5})']
            },
            FaultType.PHENOMENON: {
                'keywords': ['报警', '异响', '振动', '温度高', '不工作', '故障', '错误', '停止', '卡住', '不到位'],
                'patterns': [r'(.{0,5})(报警|异响|振动|温度高|不工作|故障|错误|停止|卡住|不到位)(.{0,5})']
            },
            FaultType.LOCATION: {
                'keywords': ['主轴', '刀库', '伺服', '液压', '电机', '轴承', '丝杠', '导轨', '控制器', '刀链'],
                'patterns': [r'(.{0,3})(主轴|刀库|伺服|液压|电机|轴承|丝杠|导轨|控制器|刀链)(.{0,3})']
            },
            FaultType.ALARM: {
                'keywords': ['ALM', 'ALARM', '警报', '报警码'],
                'patterns': [r'(ALM\d+|ALARM\d+|警报\d+|报警码\d+)']
            }
        }
    
    def _test_service(self) -> bool:
        """测试实体识别服务是否可用"""
        try:
            test_data = {"text": "测试"}
            response = requests.post(
                self.service_url, 
                json=test_data, 
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"实体识别服务不可用: {e}")
            return False
    
    def extract_entities(self, text: str) -> List[FaultElement]:
        """
        提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            故障元素列表
        """
        elements = []
        
        # 首先尝试使用NER服务
        if self.service_available:
            try:
                ner_elements = self._extract_with_ner_service(text)
                elements.extend(ner_elements)
                self.logger.info(f"NER服务提取到 {len(ner_elements)} 个实体")
            except Exception as e:
                self.logger.error(f"NER服务调用失败: {e}")
                self.service_available = False
        
        # 如果NER服务不可用或提取结果较少，使用规则匹配作为补充
        if not self.service_available or len(elements) < 3:
            fallback_elements = self._extract_with_rules(text)
            # 去重合并
            elements = self._merge_elements(elements, fallback_elements)
            self.logger.info(f"规则匹配补充提取到 {len(fallback_elements)} 个实体")
        
        # 后处理：去重、排序、过滤
        elements = self._post_process_elements(elements, text)
        
        return elements
    
    def _extract_with_ner_service(self, text: str) -> List[FaultElement]:
        """使用NER服务提取实体"""
        try:
            data = {"text": text}
            response = requests.post(
                self.service_url, 
                json=data, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            elements = []
            
            # 解析NER服务返回的结果
            # 格式: {'entities': [{'end_pos': 7, 'name': '刀链', 'start_pos': 5, 'type': '部件单元'},...]}
            if 'entities' in result:
                for entity in result['entities']:
                    entity_text = entity.get('name', '')  # 实体名称
                    entity_type = entity.get('type', '')   # 实体类型（中文）
                    start_pos = entity.get('start_pos', 0) # 开始位置
                    confidence = 0.9  # NER服务的置信度较高
                    
                    # 映射实体类型到故障类型
                    fault_type = self.entity_type_mapping.get(entity_type, FaultType.PHENOMENON)
                    
                    element = FaultElement(
                        content=entity_text,
                        element_type=fault_type,
                        confidence=confidence,
                        position=start_pos
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            self.logger.error(f"NER服务调用异常: {e}")
            raise
    
    def _extract_with_rules(self, text: str) -> List[FaultElement]:
        """使用规则匹配提取实体（回退模式）"""
        import re
        elements = []
        
        for fault_type, config in self.fallback_patterns.items():
            # 关键词匹配
            for keyword in config['keywords']:
                if keyword in text:
                    # 获取关键词周围的上下文
                    context = self._extract_context(text, keyword)
                    if context:
                        element = FaultElement(
                            content=context,
                            element_type=fault_type,
                            confidence=0.7,  # 规则匹配的置信度稍低
                            position=text.find(keyword)
                        )
                        elements.append(element)
            
            # 正则模式匹配
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text)
                for match in matches:
                    element = FaultElement(
                        content=match.group().strip(),
                        element_type=fault_type,
                        confidence=0.8,
                        position=match.start()
                    )
                    elements.append(element)
        
        return elements
    
    def _extract_context(self, text: str, keyword: str, window: int = 10) -> str:
        """提取关键词周围的上下文"""
        pos = text.find(keyword)
        if pos == -1:
            return ""
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(keyword) + window)
        context = text[start:end].strip()
        
        return context
    
    def _merge_elements(self, ner_elements: List[FaultElement], rule_elements: List[FaultElement]) -> List[FaultElement]:
        """合并NER和规则提取的元素，去重"""
        all_elements = ner_elements.copy()
        
        # 创建已有实体的集合（用于去重）
        existing_contents = {elem.content.lower() for elem in ner_elements}
        
        # 添加规则匹配的元素（如果不重复）
        for rule_elem in rule_elements:
            if rule_elem.content.lower() not in existing_contents:
                all_elements.append(rule_elem)
                existing_contents.add(rule_elem.content.lower())
        
        return all_elements
    
    def _post_process_elements(self, elements: List[FaultElement], original_text: str) -> List[FaultElement]:
        """后处理实体列表"""
        if not elements:
            return elements
        
        # 1. 去重（基于内容和类型）
        seen = set()
        unique_elements = []
        for element in elements:
            key = (element.content.lower(), element.element_type)
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)
        
        # 2. 过滤过短的实体
        filtered_elements = [
            elem for elem in unique_elements 
            if len(elem.content.strip()) >= 2
        ]
        
        # 3. 按位置排序
        filtered_elements.sort(key=lambda x: x.position if x.position is not None else 0)
        
        # 4. 限制数量（避免过多实体）
        if len(filtered_elements) > 15:
            # 按置信度排序，取前15个
            filtered_elements.sort(key=lambda x: x.confidence, reverse=True)
            filtered_elements = filtered_elements[:15]
            # 重新按位置排序
            filtered_elements.sort(key=lambda x: x.position if x.position is not None else 0)
        
        return filtered_elements
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "service_url": self.service_url,
            "service_available": self.service_available,
            "fallback_enabled": self.fallback_enabled,
            "entity_types_supported": list(self.entity_type_mapping.keys())
        }
    
    def update_entity_mapping(self, new_mapping: Dict[str, FaultType]):
        """更新实体类型映射"""
        self.entity_type_mapping.update(new_mapping)
        self.logger.info("实体类型映射已更新")
    
    def refresh_service_status(self):
        """刷新服务状态"""
        self.service_available = self._test_service()
        return self.service_available