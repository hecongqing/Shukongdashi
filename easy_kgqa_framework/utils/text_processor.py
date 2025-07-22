"""
简化版文本处理器
提供基本的中文文本处理功能
"""

import re
import jieba
from typing import List, Dict, Any
from ..models.entities import FaultElement, FaultType


class SimpleTextProcessor:
    """简化版文本处理器"""
    
    def __init__(self):
        """初始化文本处理器"""
        # 故障关键词映射
        self.fault_keywords = {
            FaultType.OPERATION: [
                "开机", "关机", "启动", "停止", "运行", "操作", "执行", "设置",
                "调节", "更换", "安装", "拆卸", "检查", "测试", "校准"
            ],
            FaultType.PHENOMENON: [
                "不转", "不动", "停止", "异响", "振动", "发热", "冒烟", "漏油",
                "报警", "显示", "闪烁", "无反应", "卡死", "松动", "变形"
            ],
            FaultType.LOCATION: [
                "主轴", "刀库", "刀架", "导轨", "丝杠", "电机", "编码器", "传感器",
                "液压", "气动", "控制器", "显示器", "键盘", "电源", "风扇"
            ],
            FaultType.ALARM: [
                "ALM", "ALARM", "报警", "错误", "故障", "异常", "警告", "提示"
            ]
        }
        
        # 停用词
        self.stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", 
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
            "着", "没有", "看", "好", "自己", "这"
        }
    
    def segment_text(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词和标点符号
        filtered_words = []
        for word in words:
            if (len(word.strip()) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[^\w\s]+$', word)):
                filtered_words.append(word.strip())
        
        return filtered_words
    
    def extract_fault_elements(self, text: str) -> List[FaultElement]:
        """
        提取故障元素
        
        Args:
            text: 输入文本
            
        Returns:
            提取的故障元素列表
        """
        elements = []
        words = self.segment_text(text)
        
        # 基于关键词匹配提取元素
        for word in words:
            for fault_type, keywords in self.fault_keywords.items():
                for keyword in keywords:
                    if keyword in word or word in keyword:
                        element = FaultElement(
                            content=word,
                            element_type=fault_type,
                            confidence=0.8  # 基础置信度
                        )
                        elements.append(element)
                        break
        
        # 去重
        unique_elements = []
        seen = set()
        for element in elements:
            key = (element.content, element.element_type)
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)
        
        return unique_elements
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 去除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 去除特殊字符（保留中文、英文、数字）
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        return text
    
    def extract_alarm_codes(self, text: str) -> List[str]:
        """
        提取报警代码
        
        Args:
            text: 输入文本
            
        Returns:
            报警代码列表
        """
        # 匹配常见的报警代码格式
        patterns = [
            r'ALM\d+',           # ALM123
            r'ALARM\d+',         # ALARM123
            r'ERR\d+',           # ERR123
            r'ERROR\d+',         # ERROR123
            r'故障代码[\s:：]*(\d+)',  # 故障代码123
            r'报警[\s:：]*(\d+)'      # 报警123
        ]
        
        alarm_codes = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    alarm_codes.append(match.group(1))
                else:
                    alarm_codes.append(match.group(0))
        
        return list(set(alarm_codes))  # 去重