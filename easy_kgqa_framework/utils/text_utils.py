"""
简洁的文本处理工具
Simple Text Processing Utilities
"""

import re
from typing import List


class TextUtils:
    """简洁的文本处理工具类"""
    
    def __init__(self):
        # 简单的停用词列表
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '它', '他', '她', '什么', '怎么', '为什么', '哪里', '吗',
            '呢', '啊', '吧', '嘛', '么', '呀'
        }
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词 - 简化版本"""
        # 清理文本
        text = self.clean_text(text)
        
        # 简单分词
        words = re.split(r'[\s，。！？；：、]+', text)
        
        # 提取关键词
        keywords = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word not in self.stopwords:
                keywords.append(word)
                
                # 对于长词，也提取子词
                if len(word) > 2:
                    for i in range(len(word) - 1):
                        for length in [2, 3, 4]:
                            if i + length <= len(word):
                                subword = word[i:i+length]
                                if subword not in self.stopwords and subword not in keywords:
                                    keywords.append(subword)
        
        return keywords
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单的Jaccard相似度）"""
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 and not keywords2:
            return 1.0
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """高亮关键词（用于显示）"""
        for keyword in keywords:
            text = text.replace(keyword, f"**{keyword}**")
        return text