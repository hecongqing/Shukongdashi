"""
文本处理工具
实现文本预处理、分句、分词、关键词提取等功能
"""

import re
import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Dict, Set, Optional
import json
import os
from ..models.entities import FaultElement, FaultType
from .entity_recognizer import EntityRecognizer


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, 
                 stopwords_path: str = None, 
                 custom_dict_path: str = None,
                 entity_service_url: str = "http://127.0.0.1:50003/extract_entities",
                 enable_entity_recognition: bool = True):
        """
        初始化文本处理器
        
        Args:
            stopwords_path: 停用词文件路径
            custom_dict_path: 自定义词典文件路径
            entity_service_url: 实体识别服务URL
            enable_entity_recognition: 是否启用实体识别
        """
        self.stopwords = self._load_stopwords(stopwords_path)
        
        if custom_dict_path and os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)
        
        # 实体识别器
        self.enable_entity_recognition = enable_entity_recognition
        if enable_entity_recognition:
            try:
                self.entity_recognizer = EntityRecognizer(
                    service_url=entity_service_url,
                    timeout=10,
                    fallback_enabled=True
                )
            except Exception as e:
                print(f"实体识别器初始化失败，使用规则匹配: {e}")
                self.entity_recognizer = None
                self.enable_entity_recognition = False
        else:
            self.entity_recognizer = None
        
        # 故障相关关键词模式（作为回退方案）
        self.fault_patterns = {
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
    
    def _load_stopwords(self, stopwords_path: str) -> Set[str]:
        """加载停用词"""
        stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f.readlines()}
        
        # 默认停用词
        default_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '当', '下', '能', '过', '时', '得', '对', '可以', '但是', '后',
            '那', '来', '用', '她', '们', '到了', '大', '里', '以', '都是', '可', '这个'
        }
        stopwords.update(default_stopwords)
        return stopwords
    
    def split_sentences(self, text: str) -> List[str]:
        """
        分句处理
        
        Args:
            text: 输入文本
            
        Returns:
            分句结果列表
        """
        # 清理文本
        text = text.strip()
        
        # 按标点符号分句
        sentence_delimiters = r'[。！？；\n]+'
        sentences = re.split(sentence_delimiters, text)
        
        # 过滤空句子并清理
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def segment_words(self, text: str, remove_stopwords: bool = True) -> List[Tuple[str, str]]:
        """
        分词处理
        
        Args:
            text: 输入文本
            remove_stopwords: 是否移除停用词
            
        Returns:
            (词, 词性) 元组列表
        """
        words = []
        
        # 使用jieba进行分词和词性标注
        for word, flag in pseg.cut(text):
            word = word.strip()
            if not word:
                continue
                
            if remove_stopwords and word in self.stopwords:
                continue
                
            words.append((word, flag))
        
        return words
    
    def extract_fault_elements(self, text: str) -> List[FaultElement]:
        """
        提取故障元素（增强版本）
        
        Args:
            text: 故障描述文本
            
        Returns:
            故障元素列表
        """
        elements = []
        
        # 首先尝试使用实体识别服务
        if self.enable_entity_recognition and self.entity_recognizer:
            try:
                ner_elements = self.entity_recognizer.extract_entities(text)
                elements.extend(ner_elements)
                print(f"实体识别提取到 {len(ner_elements)} 个元素")
            except Exception as e:
                print(f"实体识别失败，使用规则匹配: {e}")
                # 如果实体识别失败，使用规则匹配
                elements = self._extract_with_rules(text)
        else:
            # 使用原有的规则匹配方法
            elements = self._extract_with_rules(text)
        
        return elements
    
    def _extract_with_rules(self, text: str) -> List[FaultElement]:
        """使用规则匹配提取故障元素（原有方法）"""
        elements = []
        
        for fault_type, config in self.fault_patterns.items():
            # 关键词匹配
            for keyword in config['keywords']:
                if keyword in text:
                    # 获取关键词周围的上下文
                    context = self._extract_context(text, keyword)
                    if context:
                        element = FaultElement(
                            content=context,
                            element_type=fault_type,
                            confidence=0.8,
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
                        confidence=0.9,
                        position=match.start()
                    )
                    elements.append(element)
        
        # 去重和排序
        elements = self._deduplicate_elements(elements)
        elements.sort(key=lambda x: x.position if x.position else 0)
        
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
    
    def _deduplicate_elements(self, elements: List[FaultElement]) -> List[FaultElement]:
        """去重故障元素"""
        seen = set()
        unique_elements = []
        
        for element in elements:
            key = (element.content, element.element_type)
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)
        
        return unique_elements
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            (关键词, 权重) 元组列表
        """
        # 分词
        words = self.segment_words(text, remove_stopwords=True)
        
        # 简单的TF统计
        word_freq = {}
        for word, flag in words:
            # 过滤单字和标点符号
            if len(word) < 2 or flag in ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'q']:
                continue
            
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 计算权重（这里使用简单的频率，实际可以使用TF-IDF等）
        if not word_freq:
            return []
        
        max_freq = max(word_freq.values())
        keywords = [(word, freq / max_freq) for word, freq in word_freq.items()]
        
        # 排序并返回top-k
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:top_k]
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，！？；：""''（）【】]', '', text)
        
        # 清理首尾空格
        text = text.strip()
        
        return text
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（基于词汇重叠）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        words1 = set(word for word, _ in self.segment_words(text1))
        words2 = set(word for word, _ in self.segment_words(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_entity_recognition_status(self) -> Dict[str, any]:
        """获取实体识别状态"""
        if not self.entity_recognizer:
            return {
                "enabled": False,
                "service_available": False,
                "fallback_mode": True
            }
        
        status = self.entity_recognizer.get_service_status()
        status["enabled"] = self.enable_entity_recognition
        status["fallback_mode"] = not status.get("service_available", False)
        
        return status
    
    def refresh_entity_service(self):
        """刷新实体识别服务状态"""
        if self.entity_recognizer:
            return self.entity_recognizer.refresh_service_status()
        return False