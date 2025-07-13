"""
网页爬虫模块
支持多源数据采集，包括网页内容、API数据等
"""

import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import pandas as pd
from pathlib import Path

from config.settings import settings


class WebCrawler:
    """网页爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.CRAWLER_USER_AGENT
        })
        self.logger = logging.getLogger(__name__)
        
    def crawl_website(self, url: str, selectors: Dict[str, str] = None) -> Dict[str, Any]:
        """
        爬取单个网页
        
        Args:
            url: 目标URL
            selectors: CSS选择器配置
            
        Returns:
            爬取结果
        """
        try:
            response = self.session.get(url, timeout=settings.CRAWLER_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            result = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'content': soup.get_text(),
                'html': response.text,
                'timestamp': time.time()
            }
            
            # 使用选择器提取特定内容
            if selectors:
                for key, selector in selectors.items():
                    elements = soup.select(selector)
                    result[key] = [elem.get_text().strip() for elem in elements]
            
            return result
            
        except Exception as e:
            self.logger.error(f"爬取失败 {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
    
    def crawl_websites(self, urls: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """
        批量爬取网页
        
        Args:
            urls: URL列表
            output_file: 输出文件路径
            
        Returns:
            爬取结果列表
        """
        results = []
        
        for url in urls:
            self.logger.info(f"正在爬取: {url}")
            result = self.crawl_website(url)
            results.append(result)
            
            # 延迟避免被封
            time.sleep(settings.CRAWLER_DELAY)
        
        # 保存结果
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def crawl_with_selenium(self, url: str, wait_for: str = None, 
                          scroll: bool = False) -> Dict[str, Any]:
        """
        使用Selenium爬取动态网页
        
        Args:
            url: 目标URL
            wait_for: 等待元素出现
            scroll: 是否滚动页面
            
        Returns:
            爬取结果
        """
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            driver.get(url)
            
            # 等待特定元素
            if wait_for:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                )
            
            # 滚动页面
            if scroll:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            result = {
                'url': url,
                'title': driver.title,
                'content': driver.find_element(By.TAG_NAME, 'body').text,
                'html': driver.page_source,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Selenium爬取失败 {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
        finally:
            driver.quit()
    
    def crawl_api(self, api_url: str, params: Dict = None, 
                 headers: Dict = None) -> Dict[str, Any]:
        """
        爬取API数据
        
        Args:
            api_url: API地址
            params: 请求参数
            headers: 请求头
            
        Returns:
            API响应数据
        """
        try:
            response = self.session.get(api_url, params=params, headers=headers)
            response.raise_for_status()
            
            return {
                'url': api_url,
                'data': response.json(),
                'status_code': response.status_code,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"API爬取失败 {api_url}: {str(e)}")
            return {'url': api_url, 'error': str(e)}
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存爬取结果"""
        output_path = Path(settings.RAW_DATA_DIR) / output_file
        
        # 根据文件扩展名选择保存格式
        if output_file.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        elif output_file.endswith('.csv'):
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            # 默认保存为JSON
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {output_path}")


class AsyncWebCrawler:
    """异步网页爬虫类"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
    
    async def crawl_website_async(self, session: aiohttp.ClientSession, 
                                url: str) -> Dict[str, Any]:
        """异步爬取单个网页"""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content': soup.get_text(),
                        'html': content,
                        'timestamp': time.time()
                    }
                else:
                    return {'url': url, 'error': f'HTTP {response.status}'}
                    
        except Exception as e:
            self.logger.error(f"异步爬取失败 {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
    
    async def crawl_websites_async(self, urls: List[str], 
                                 output_file: str = None) -> List[Dict[str, Any]]:
        """异步批量爬取网页"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.crawl_website_async(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤异常结果
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            # 保存结果
            if output_file:
                self._save_results(valid_results, output_file)
            
            return valid_results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存爬取结果"""
        output_path = Path(settings.RAW_DATA_DIR) / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 同步爬虫示例
    crawler = WebCrawler()
    urls = [
        "https://zhuanlan.zhihu.com/p/593008416",
        "https://www.example.com"
    ]
    
    results = crawler.crawl_websites(urls, "crawled_data.json")
    print(f"爬取了 {len(results)} 个网页")
    
    # 异步爬虫示例
    async def main():
        async_crawler = AsyncWebCrawler()
        results = await async_crawler.crawl_websites_async(urls, "async_crawled_data.json")
        print(f"异步爬取了 {len(results)} 个网页")
    
    asyncio.run(main())