"""
数据采集模块
支持多种数据源的数据采集，包括网页爬取、API接口、文件解析等
"""
import asyncio
import aiohttp
import requests
import json
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from loguru import logger
import jieba
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from config.settings import CONFIG

class DataCrawler:
    """数据采集器基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG["data"]["crawler"]
        self.session = requests.Session()
        self.session.headers.update(self.config["headers"])
        self.data_dir = Path(CONFIG["data"]["raw_data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """爬取单个URL的数据"""
        raise NotImplementedError
    
    def batch_crawl(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量爬取多个URL的数据"""
        results = []
        for url in urls:
            try:
                result = self.crawl(url, **kwargs)
                if result:
                    results.append(result)
                time.sleep(self.config["delay"])
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
        return results
    
    def save_data(self, data: List[Dict[str, Any]], filename: str, format: str = "json"):
        """保存数据到文件"""
        filepath = self.data_dir / filename
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "csv":
            if data:
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False, encoding='utf-8')
        elif format == "txt":
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(str(item) + '\n')
        
        logger.info(f"Saved {len(data)} items to {filepath}")


class WikipediaCrawler(DataCrawler):
    """Wikipedia数据爬取器"""
    
    def __init__(self, language: str = "zh"):
        super().__init__()
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org"
        self.api_url = f"{self.base_url}/w/api.php"
    
    def crawl_page(self, title: str) -> Dict[str, Any]:
        """爬取Wikipedia页面信息"""
        try:
            # 获取页面内容
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts|pageimages|categories|links",
                "exintro": True,
                "explaintext": True,
                "piprop": "original",
                "cllimit": 50,
                "pllimit": 100
            }
            
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return {}
            
            page_id = list(pages.keys())[0]
            page_data = pages[page_id]
            
            # 获取infobox信息
            infobox = self._extract_infobox(title)
            
            result = {
                "title": title,
                "pageid": page_data.get("pageid"),
                "extract": page_data.get("extract", ""),
                "categories": [cat.get("title", "").replace("Category:", "") 
                             for cat in page_data.get("categories", [])],
                "links": [link.get("title", "") for link in page_data.get("links", [])],
                "infobox": infobox,
                "image": page_data.get("original", {}).get("source", ""),
                "source": "wikipedia",
                "language": self.language,
                "crawl_time": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error crawling Wikipedia page {title}: {e}")
            return {}
    
    def _extract_infobox(self, title: str) -> Dict[str, str]:
        """提取infobox信息"""
        try:
            params = {
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "wikitext"
            }
            
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
            
            # 简单的infobox解析（可以进一步优化）
            infobox = {}
            if "{{Infobox" in wikitext:
                lines = wikitext.split('\n')
                in_infobox = False
                for line in lines:
                    if line.startswith("{{Infobox"):
                        in_infobox = True
                        continue
                    if in_infobox and line.strip() == "}}":
                        break
                    if in_infobox and "|" in line:
                        parts = line.split("|", 1)
                        if len(parts) == 2:
                            key = parts[1].split("=")[0].strip()
                            value = parts[1].split("=", 1)[1].strip() if "=" in parts[1] else ""
                            if key and value:
                                infobox[key] = value
            
            return infobox
            
        except Exception as e:
            logger.error(f"Error extracting infobox for {title}: {e}")
            return {}
    
    def search_pages(self, query: str, limit: int = 10) -> List[str]:
        """搜索Wikipedia页面"""
        try:
            params = {
                "action": "opensearch",
                "format": "json",
                "search": query,
                "limit": limit
            }
            
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            return data[1] if len(data) > 1 else []
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia for {query}: {e}")
            return []


class BaiduBaikeCrawler(DataCrawler):
    """百度百科爬取器"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://baike.baidu.com"
    
    def crawl_page(self, keyword: str) -> Dict[str, Any]:
        """爬取百度百科页面"""
        try:
            url = f"{self.base_url}/item/{keyword}"
            response = self.session.get(url, timeout=self.config["timeout"])
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取基本信息
            title = soup.find('dd', class_='lemmaWgt-lemmaTitle-title')
            title = title.get_text().strip() if title else keyword
            
            # 提取摘要
            summary = soup.find('div', class_='lemma-summary')
            summary_text = summary.get_text().strip() if summary else ""
            
            # 提取基本信息卡片
            basic_info = {}
            info_box = soup.find('div', class_='basic-info')
            if info_box:
                items = info_box.find_all('dl', class_='basicInfo-item')
                for item in items:
                    name_tag = item.find('dt', class_='basicInfo-item name')
                    value_tag = item.find('dd', class_='basicInfo-item value')
                    if name_tag and value_tag:
                        name = name_tag.get_text().strip()
                        value = value_tag.get_text().strip()
                        basic_info[name] = value
            
            # 提取正文内容
            content_div = soup.find('div', class_='lemmaWgt-lemmaContent')
            content = content_div.get_text().strip() if content_div else ""
            
            # 提取图片
            images = []
            img_tags = soup.find_all('img', class_='lemmaWgt-lemmaContent-pictureWrap-pictrue')
            for img in img_tags:
                src = img.get('src')
                if src:
                    images.append(src)
            
            # 提取标签
            tags = []
            tag_list = soup.find('span', class_='taglist')
            if tag_list:
                tag_links = tag_list.find_all('a')
                tags = [tag.get_text().strip() for tag in tag_links]
            
            result = {
                "title": title,
                "keyword": keyword,
                "url": url,
                "summary": summary_text,
                "basic_info": basic_info,
                "content": content,
                "images": images,
                "tags": tags,
                "source": "baidu_baike",
                "crawl_time": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error crawling Baidu Baike page {keyword}: {e}")
            return {}


class NewsCrawler(DataCrawler):
    """新闻数据爬取器"""
    
    def __init__(self, news_sites: List[str] = None):
        super().__init__()
        self.news_sites = news_sites or [
            "https://news.sina.com.cn",
            "https://news.163.com", 
            "https://www.thepaper.cn"
        ]
    
    def crawl_news_list(self, site_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """爬取新闻列表"""
        news_list = []
        
        try:
            response = self.session.get(site_url, timeout=self.config["timeout"])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 根据不同新闻网站的结构提取链接
            if "sina.com" in site_url:
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href and '/news/' in href and link.get_text().strip():
                        full_url = urljoin(site_url, href)
                        news_list.append({
                            "url": full_url,
                            "title": link.get_text().strip(),
                            "source": "sina"
                        })
            
            elif "163.com" in site_url:
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link.get('href')
                    if href and 'news.163.com' in href and link.get_text().strip():
                        news_list.append({
                            "url": href,
                            "title": link.get_text().strip(),
                            "source": "163"
                        })
            
        except Exception as e:
            logger.error(f"Error crawling news list from {site_url}: {e}")
        
        return news_list[:max_pages * 20]  # 限制数量
    
    def crawl_news_content(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """爬取新闻详细内容"""
        try:
            response = self.session.get(news_item["url"], timeout=self.config["timeout"])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取标题
            title = news_item.get("title", "")
            if not title:
                title_tag = soup.find('h1') or soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ""
            
            # 提取发布时间
            publish_time = ""
            time_patterns = [
                {'class': 'time'},
                {'class': 'publish-time'},
                {'class': 'article-time'},
                {'id': 'pub_date'}
            ]
            
            for pattern in time_patterns:
                time_tag = soup.find('span', pattern) or soup.find('div', pattern)
                if time_tag:
                    publish_time = time_tag.get_text().strip()
                    break
            
            # 提取正文内容
            content = ""
            content_patterns = [
                {'class': 'article-content'},
                {'class': 'content'},
                {'class': 'post_text'},
                {'id': 'article'}
            ]
            
            for pattern in content_patterns:
                content_tag = soup.find('div', pattern)
                if content_tag:
                    content = content_tag.get_text().strip()
                    break
            
            # 如果还没找到内容，尝试提取所有p标签
            if not content:
                p_tags = soup.find_all('p')
                content = '\n'.join([p.get_text().strip() for p in p_tags if p.get_text().strip()])
            
            result = {
                "title": title,
                "url": news_item["url"],
                "content": content,
                "publish_time": publish_time,
                "source": news_item.get("source", "unknown"),
                "crawl_time": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error crawling news content from {news_item['url']}: {e}")
            return {}


class PDFExtractor:
    """PDF文档内容提取器"""
    
    def __init__(self):
        self.data_dir = Path(CONFIG["data"]["raw_data_dir"])
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                return text
                
        except ImportError:
            logger.error("PyPDF2 not installed. Please install it with: pip install PyPDF2")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def extract_structured_data(self, pdf_path: str) -> Dict[str, Any]:
        """从PDF提取结构化数据"""
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            return {}
        
        # 简单的结构化处理
        sentences = text.split('。')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return {
            "filename": Path(pdf_path).name,
            "full_text": text,
            "sentences": sentences,
            "word_count": len(text),
            "sentence_count": len(sentences),
            "extract_time": time.time()
        }


class DataCollectionManager:
    """数据采集管理器"""
    
    def __init__(self):
        self.wikipedia_crawler = WikipediaCrawler()
        self.baidu_crawler = BaiduBaikeCrawler()
        self.news_crawler = NewsCrawler()
        self.pdf_extractor = PDFExtractor()
        self.output_dir = Path(CONFIG["data"]["raw_data_dir"])
        
    def collect_encyclopedia_data(self, keywords: List[str]) -> None:
        """采集百科数据"""
        logger.info(f"Collecting encyclopedia data for {len(keywords)} keywords")
        
        # Wikipedia数据
        wikipedia_data = []
        for keyword in keywords:
            data = self.wikipedia_crawler.crawl_page(keyword)
            if data:
                wikipedia_data.append(data)
            time.sleep(1)
        
        if wikipedia_data:
            self.wikipedia_crawler.save_data(
                wikipedia_data, 
                "wikipedia_data.json"
            )
        
        # 百度百科数据
        baidu_data = []
        for keyword in keywords:
            data = self.baidu_crawler.crawl_page(keyword)
            if data:
                baidu_data.append(data)
            time.sleep(1)
        
        if baidu_data:
            self.baidu_crawler.save_data(
                baidu_data,
                "baidu_baike_data.json"
            )
    
    def collect_news_data(self, max_articles: int = 100) -> None:
        """采集新闻数据"""
        logger.info(f"Collecting news data, max articles: {max_articles}")
        
        all_news = []
        
        for site in self.news_crawler.news_sites:
            try:
                news_list = self.news_crawler.crawl_news_list(site)
                logger.info(f"Found {len(news_list)} news from {site}")
                
                for news_item in news_list[:max_articles//len(self.news_crawler.news_sites)]:
                    content = self.news_crawler.crawl_news_content(news_item)
                    if content and content.get('content'):
                        all_news.append(content)
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error collecting news from {site}: {e}")
        
        if all_news:
            self.news_crawler.save_data(all_news, "news_data.json")
    
    def collect_document_data(self, document_dir: str) -> None:
        """采集文档数据"""
        logger.info(f"Collecting document data from {document_dir}")
        
        doc_dir = Path(document_dir)
        if not doc_dir.exists():
            logger.error(f"Document directory {document_dir} does not exist")
            return
        
        all_documents = []
        
        # 处理PDF文件
        pdf_files = list(doc_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                doc_data = self.pdf_extractor.extract_structured_data(str(pdf_file))
                if doc_data:
                    all_documents.append(doc_data)
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_file}: {e}")
        
        # 处理文本文件
        txt_files = list(doc_dir.glob("*.txt"))
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                doc_data = {
                    "filename": txt_file.name,
                    "full_text": content,
                    "sentences": content.split('。'),
                    "word_count": len(content),
                    "extract_time": time.time()
                }
                all_documents.append(doc_data)
                
            except Exception as e:
                logger.error(f"Error processing text file {txt_file}: {e}")
        
        if all_documents:
            output_file = self.output_dir / "document_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(all_documents)} documents to {output_file}")
    
    def run_full_collection(self, 
                          keywords: List[str] = None,
                          collect_news: bool = True,
                          collect_encyclopedia: bool = True,
                          document_dir: str = None) -> None:
        """运行完整的数据采集流程"""
        logger.info("Starting full data collection process")
        
        # 默认关键词
        if not keywords:
            keywords = [
                "人工智能", "机器学习", "深度学习", "自然语言处理",
                "计算机视觉", "知识图谱", "大数据", "云计算"
            ]
        
        # 采集百科数据
        if collect_encyclopedia:
            self.collect_encyclopedia_data(keywords)
        
        # 采集新闻数据
        if collect_news:
            self.collect_news_data()
        
        # 采集文档数据
        if document_dir:
            self.collect_document_data(document_dir)
        
        logger.info("Data collection completed successfully")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Data Collection")
    parser.add_argument("--keywords", nargs='+', help="Keywords for encyclopedia data collection")
    parser.add_argument("--no-news", action="store_true", help="Skip news data collection")
    parser.add_argument("--no-encyclopedia", action="store_true", help="Skip encyclopedia data collection")
    parser.add_argument("--document-dir", help="Directory containing documents to process")
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = DataCollectionManager()
    
    # 运行数据采集
    manager.run_full_collection(
        keywords=args.keywords,
        collect_news=not args.no_news,
        collect_encyclopedia=not args.no_encyclopedia,
        document_dir=args.document_dir
    )


if __name__ == "__main__":
    main()