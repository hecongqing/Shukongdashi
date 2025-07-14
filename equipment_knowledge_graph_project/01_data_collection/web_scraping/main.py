#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装备制造故障数据采集主程序
支持多种数据源的自动化采集
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.config import Config
from utils.database import DatabaseManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EquipmentFaultDataCollector:
    """装备制造故障数据采集器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.session = requests.Session()
        self.setup_session()
        
    def setup_session(self):
        """设置请求会话"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session.headers.update(headers)
        
    def setup_selenium_driver(self):
        """设置Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
        
    def collect_from_zhihu(self, keywords: List[str], max_pages: int = 10) -> List[Dict]:
        """从知乎采集装备制造故障相关数据"""
        collected_data = []
        
        for keyword in keywords:
            logger.info(f"开始采集关键词: {keyword}")
            search_url = f"https://www.zhihu.com/search?type=content&q={keyword}"
            
            try:
                driver = self.setup_selenium_driver()
                driver.get(search_url)
                
                # 等待页面加载
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "Search-result"))
                )
                
                for page in range(max_pages):
                    logger.info(f"正在处理第 {page + 1} 页")
                    
                    # 获取搜索结果
                    articles = driver.find_elements(By.CSS_SELECTOR, ".Search-result .ContentItem")
                    
                    for article in articles:
                        try:
                            # 提取文章信息
                            title_elem = article.find_element(By.CSS_SELECTOR, "h2 a")
                            title = title_elem.text.strip()
                            url = title_elem.get_attribute("href")
                            
                            # 提取摘要
                            summary_elem = article.find_element(By.CSS_SELECTOR, ".RichContent-inner")
                            summary = summary_elem.text.strip()
                            
                            # 提取作者信息
                            author_elem = article.find_element(By.CSS_SELECTOR, ".AuthorInfo-name")
                            author = author_elem.text.strip()
                            
                            article_data = {
                                'title': title,
                                'url': url,
                                'summary': summary,
                                'author': author,
                                'keyword': keyword,
                                'source': 'zhihu',
                                'collected_at': datetime.now().isoformat()
                            }
                            
                            collected_data.append(article_data)
                            
                        except Exception as e:
                            logger.warning(f"提取文章信息失败: {e}")
                            continue
                    
                    # 点击下一页
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, ".Pagination-next")
                        if next_button.is_enabled():
                            next_button.click()
                            time.sleep(2)
                        else:
                            break
                    except:
                        break
                        
            except Exception as e:
                logger.error(f"采集知乎数据失败: {e}")
            finally:
                driver.quit()
                
        return collected_data
    
    def collect_from_baidu(self, keywords: List[str], max_pages: int = 5) -> List[Dict]:
        """从百度搜索采集装备制造故障相关数据"""
        collected_data = []
        
        for keyword in keywords:
            logger.info(f"开始采集百度关键词: {keyword}")
            
            for page in range(max_pages):
                try:
                    search_url = f"https://www.baidu.com/s?wd={keyword}&pn={page * 10}"
                    response = self.session.get(search_url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    results = soup.find_all('div', class_='result')
                    
                    for result in results:
                        try:
                            # 提取标题和链接
                            title_elem = result.find('h3', class_='t')
                            if not title_elem:
                                continue
                                
                            title_link = title_elem.find('a')
                            title = title_link.get_text().strip()
                            url = title_link.get('href')
                            
                            # 提取摘要
                            summary_elem = result.find('div', class_='c-abstract')
                            summary = summary_elem.get_text().strip() if summary_elem else ""
                            
                            # 获取实际内容
                            content = self.get_webpage_content(url)
                            
                            article_data = {
                                'title': title,
                                'url': url,
                                'summary': summary,
                                'content': content,
                                'keyword': keyword,
                                'source': 'baidu',
                                'collected_at': datetime.now().isoformat()
                            }
                            
                            collected_data.append(article_data)
                            
                        except Exception as e:
                            logger.warning(f"提取百度搜索结果失败: {e}")
                            continue
                            
                    time.sleep(1)  # 避免请求过快
                    
                except Exception as e:
                    logger.error(f"采集百度数据失败: {e}")
                    break
                    
        return collected_data
    
    def get_webpage_content(self, url: str) -> str:
        """获取网页内容"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
                
            # 提取文本内容
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # 限制长度
            
        except Exception as e:
            logger.warning(f"获取网页内容失败 {url}: {e}")
            return ""
    
    def collect_from_technical_forums(self) -> List[Dict]:
        """从技术论坛采集数据"""
        forums = [
            {
                'name': '机械社区',
                'url': 'http://www.cmiw.cn/forum.php?mod=forumdisplay&fid=2',
                'selector': '.tl'
            },
            {
                'name': '中国机械CAD论坛',
                'url': 'http://www.jxcad.com.cn/forum-2-1.html',
                'selector': '.tl'
            }
        ]
        
        collected_data = []
        
        for forum in forums:
            logger.info(f"开始采集论坛: {forum['name']}")
            
            try:
                response = self.session.get(forum['url'], timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                threads = soup.select(forum['selector'])
                
                for thread in threads[:20]:  # 限制数量
                    try:
                        title_elem = thread.find('a')
                        if not title_elem:
                            continue
                            
                        title = title_elem.get_text().strip()
                        url = urljoin(forum['url'], title_elem.get('href'))
                        
                        # 获取帖子内容
                        content = self.get_webpage_content(url)
                        
                        thread_data = {
                            'title': title,
                            'url': url,
                            'content': content,
                            'source': forum['name'],
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        collected_data.append(thread_data)
                        
                    except Exception as e:
                        logger.warning(f"提取论坛帖子失败: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"采集论坛数据失败 {forum['name']}: {e}")
                
        return collected_data
    
    def save_data(self, data: List[Dict], output_file: str = None):
        """保存采集的数据"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"../data/collected_data_{timestamp}.json"
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"数据已保存到: {output_file}")
        
        # 同时保存到数据库
        self.db_manager.save_collected_data(data)
    
    def run(self):
        """运行数据采集"""
        logger.info("开始装备制造故障数据采集")
        
        # 定义搜索关键词
        keywords = [
            "数控机床故障诊断",
            "设备维修技术",
            "机械故障分析",
            "设备维护保养",
            "故障排除方法",
            "设备故障案例",
            "维修经验分享",
            "故障预防措施"
        ]
        
        all_data = []
        
        # 从知乎采集
        logger.info("开始从知乎采集数据")
        zhihu_data = self.collect_from_zhihu(keywords, max_pages=5)
        all_data.extend(zhihu_data)
        
        # 从百度采集
        logger.info("开始从百度采集数据")
        baidu_data = self.collect_from_baidu(keywords, max_pages=3)
        all_data.extend(baidu_data)
        
        # 从技术论坛采集
        logger.info("开始从技术论坛采集数据")
        forum_data = self.collect_from_technical_forums()
        all_data.extend(forum_data)
        
        # 保存数据
        self.save_data(all_data)
        
        logger.info(f"数据采集完成，共采集 {len(all_data)} 条数据")
        return all_data

def main():
    """主函数"""
    # 加载配置
    config = Config()
    
    # 创建采集器
    collector = EquipmentFaultDataCollector(config)
    
    # 运行采集
    data = collector.run()
    
    print(f"采集完成，共获得 {len(data)} 条数据")

if __name__ == "__main__":
    main()