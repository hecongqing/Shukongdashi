"""
实体识别服务接口
调用外部实体识别服务
"""

import requests
from typing import List, Dict, Any
from ..models.entities import FaultElement, FaultType
from ..config import config


class EntityService:
    """实体识别服务接口"""
    
    def __init__(self, service_url: str = None):
        """
        初始化实体识别服务
        
        Args:
            service_url: 服务URL，默认从配置读取
        """
        self.service_url = service_url or config.ENTITY_SERVICE_URL
        self.timeout = 10
        
        # 实体类型映射
        self.entity_type_mapping = {
            '部件单元': FaultType.LOCATION,
            '性能表征': FaultType.PHENOMENON,
            '故障状态': FaultType.PHENOMENON,
            '检测工具': FaultType.LOCATION,
            '故障现象': FaultType.PHENOMENON,
            '操作行为': FaultType.OPERATION,
            '报警信息': FaultType.ALARM
        }
    
    def test_service(self) -> bool:
        """测试服务是否可用"""
        try:
            response = requests.get(
                self.service_url.replace('/extract_entities', '/health'),
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def extract_entities(self, text: str) -> List[FaultElement]:
        """
        调用外部服务提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的故障元素列表
        """
        try:
            # 调用外部实体识别服务
            response = requests.post(
                self.service_url,
                json={"text": text},
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_entities(result)
            else:
                print(f"实体识别服务错误: {response.status_code}")
                return []
                
        except requests.exceptions.Timeout:
            print("实体识别服务超时")
            return []
        except Exception as e:
            print(f"调用实体识别服务时出错: {e}")
            return []
    
    def _parse_entities(self, result: Dict[str, Any]) -> List[FaultElement]:
        """
        解析服务返回的实体
        
        Args:
            result: 服务返回的结果
            
        Returns:
            故障元素列表
        """
        elements = []
        
        # 假设服务返回格式为：
        # {
        #   "entities": [
        #     {"text": "主轴", "label": "部件单元", "confidence": 0.9},
        #     {"text": "不转", "label": "故障状态", "confidence": 0.8}
        #   ]
        # }
        
        entities = result.get("entities", [])
        for entity in entities:
            text = entity.get("text", "")
            label = entity.get("label", "")
            confidence = entity.get("confidence", 0.8)
            
            # 映射实体类型
            fault_type = self.entity_type_mapping.get(label, FaultType.PHENOMENON)
            
            element = FaultElement(
                content=text,
                element_type=fault_type,
                confidence=confidence
            )
            elements.append(element)
        
        return elements