"""
数据采集爬虫模块

包含网络爬虫和PDF文档提取功能
"""

import requests
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from loguru import logger


class WebCrawler:
    """网络爬虫类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config['crawler']['user_agent']
        })
        self.delay = config['crawler']['delay']
        self.timeout = config['crawler']['timeout']
        self.max_retries = config['crawler']['max_retries']
        
    def get_page(self, url: str) -> Optional[str]:
        """获取网页内容"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"正在获取页面: {url}")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                time.sleep(self.delay)
                return response.text
            except Exception as e:
                logger.warning(f"第{attempt + 1}次尝试失败: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"获取页面失败: {url}")
                    return None
                time.sleep(self.delay * 2)
        return None
    
    def parse_html(self, html: str) -> Dict[str, Any]:
        """解析HTML内容"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 提取标题
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # 提取正文内容
        content = ""
        # 常见的正文容器
        content_selectors = [
            'article', '.content', '.main-content', '.post-content',
            '.article-content', '#content', '.text'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text().strip()
                break
        
        if not content:
            # 如果没有找到特定容器，提取所有段落
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text().strip() for p in paragraphs])
        
        return {
            'title': title_text,
            'content': content,
            'url': getattr(self, 'current_url', '')
        }
    
    def crawl_fault_cases(self, base_url: str, max_pages: int = 100) -> List[Dict[str, Any]]:
        """爬取故障案例数据"""
        cases = []
        page = 1
        
        while page <= max_pages:
            url = f"{base_url}?page={page}"
            html = self.get_page(url)
            
            if not html:
                break
                
            self.current_url = url
            parsed_data = self.parse_html(html)
            
            if parsed_data['content']:
                cases.append(parsed_data)
                logger.info(f"成功获取第{page}页数据")
            
            page += 1
        
        logger.info(f"共获取{len(cases)}个故障案例")
        return cases
    
    def crawl_with_selenium(self, url: str) -> Optional[str]:
        """使用Selenium处理动态页面"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # 等待页面加载
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # 获取页面内容
            page_source = driver.page_source
            driver.quit()
            
            return page_source
        except Exception as e:
            logger.error(f"Selenium爬取失败: {e}")
            return None


class PDFExtractor:
    """PDF文档提取器"""
    
    def __init__(self):
        pass
    
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取PDF文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PyPDF2提取失败: {e}")
        return text
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """使用PyMuPDF提取PDF文本"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF提取失败: {e}")
        return text
    
    def extract_text(self, pdf_path: str) -> str:
        """提取PDF文本（优先使用PyMuPDF）"""
        text = self.extract_text_pymupdf(pdf_path)
        if not text.strip():
            text = self.extract_text_pypdf2(pdf_path)
        return text
    
    def extract_fault_manual(self, pdf_path: str) -> List[Dict[str, Any]]:
        """提取故障手册数据"""
        text = self.extract_text(pdf_path)
        
        # 简单的文本分割（实际项目中需要更复杂的解析逻辑）
        sections = text.split('\n\n')
        
        fault_data = []
        current_fault = {}
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # 检测故障标题（简单的启发式规则）
            if any(keyword in section for keyword in ['故障', '问题', '异常', '错误']):
                if current_fault:
                    fault_data.append(current_fault)
                current_fault = {
                    'title': section[:100],
                    'content': section,
                    'source': pdf_path
                }
            elif current_fault:
                current_fault['content'] += '\n' + section
        
        if current_fault:
            fault_data.append(current_fault)
        
        logger.info(f"从PDF提取了{len(fault_data)}个故障信息")
        return fault_data


class APICollector:
    """API数据收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
    
    def collect_expert_knowledge(self, api_url: str, api_key: str = None) -> List[Dict[str, Any]]:
        """从专家知识库API收集数据"""
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        try:
            response = self.session.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"从API获取了{len(data)}条专家知识")
            return data
            
        except Exception as e:
            logger.error(f"API数据收集失败: {e}")
            return []