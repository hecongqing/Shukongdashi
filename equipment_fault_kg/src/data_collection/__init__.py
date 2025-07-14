"""
装备制造故障知识图谱 - 数据采集模块

该模块负责从各种数据源采集装备故障相关的数据，包括：
- 故障案例数据
- 技术手册数据
- 专家知识库数据
"""

from .crawler import WebCrawler, PDFExtractor
from .processor import DataProcessor
from .collector import DataCollector

__all__ = ['WebCrawler', 'PDFExtractor', 'DataProcessor', 'DataCollector']