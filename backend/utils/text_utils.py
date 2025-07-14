import re
from typing import List

class TextPreprocessor:
    """文本预处理工具

    这里只实现了一些基础清洗及规范化功能，后续可根据实际需求扩展。
    """

    def __init__(self):
        # 保留可扩展的正则模式或停用词等
        self.whitespace_pattern = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """基础清洗：去除多余空白符、特殊控制字符等"""
        if text is None:
            return ""
        # 替换制表符/换行符为空格
        text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        # 合并多余空格
        text = self.whitespace_pattern.sub(" ", text)
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """统一文本表示用于生成 ID 或比对文本相似度"""
        text = self.clean_text(text)
        # 全角转半角、大小写归一等（简单实现）
        text = text.lower()
        # 去除中文空白字符
        text = text.replace("\u3000", " ")
        return text

    def tokenize(self, text: str) -> List[str]:
        """简单分词（可根据项目使用 jieba 或其他库）"""
        return list(text)