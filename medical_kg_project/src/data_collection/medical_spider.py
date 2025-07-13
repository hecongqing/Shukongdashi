"""
医疗百科数据采集爬虫
"""
import scrapy
import json
from typing import Dict, List
from urllib.parse import urljoin
import re


class MedicalBaikeSpider(scrapy.Spider):
    """百度百科医疗词条爬虫"""
    
    name = 'medical_baike'
    allowed_domains = ['baike.baidu.com']
    
    # 医疗相关的种子词条
    start_urls = [
        'https://baike.baidu.com/item/糖尿病',
        'https://baike.baidu.com/item/高血压',
        'https://baike.baidu.com/item/冠心病',
        'https://baike.baidu.com/item/肺炎',
        'https://baike.baidu.com/item/胃炎',
    ]
    
    # 医疗相关的关键词模式
    medical_patterns = [
        r'症状', r'病因', r'治疗', r'诊断', r'预防',
        r'并发症', r'用药', r'检查', r'病理', r'流行病学'
    ]
    
    def parse(self, response):
        """解析百科页面"""
        # 提取基本信息
        item = {
            'url': response.url,
            'title': self._extract_title(response),
            'summary': self._extract_summary(response),
            'infobox': self._extract_infobox(response),
            'content': self._extract_content(response),
            'categories': self._extract_categories(response),
        }
        
        # 提取医疗相关信息
        medical_info = self._extract_medical_info(response)
        item.update(medical_info)
        
        yield item
        
        # 提取相关链接并继续爬取
        for link in self._extract_related_links(response):
            if self._is_medical_related(link):
                yield response.follow(link, self.parse)
    
    def _extract_title(self, response) -> str:
        """提取词条标题"""
        return response.css('.lemmaWgt-lemmaTitle-title h1::text').get('').strip()
    
    def _extract_summary(self, response) -> str:
        """提取词条摘要"""
        summary_parts = response.css('.lemma-summary .para::text').getall()
        return ' '.join(summary_parts).strip()
    
    def _extract_infobox(self, response) -> Dict[str, str]:
        """提取信息框数据"""
        infobox = {}
        for item in response.css('.basic-info .basicInfo-item'):
            name = item.css('.name::text').get('').strip()
            value = ' '.join(item.css('.value::text').getall()).strip()
            if name and value:
                infobox[name] = value
        return infobox
    
    def _extract_content(self, response) -> Dict[str, str]:
        """提取正文内容"""
        content = {}
        current_section = '简介'
        
        for element in response.css('.main-content > *'):
            # 处理标题
            if element.css('h2').get():
                current_section = element.css('h2::text').get('').strip()
                content[current_section] = ''
            # 处理段落
            elif element.css('.para').get():
                text = ' '.join(element.css('.para::text').getall()).strip()
                if current_section in content:
                    content[current_section] += text + '\n'
                else:
                    content[current_section] = text + '\n'
        
        return content
    
    def _extract_medical_info(self, response) -> Dict[str, any]:
        """提取医疗专业信息"""
        medical_info = {
            'symptoms': [],
            'causes': [],
            'treatments': [],
            'examinations': [],
            'preventions': [],
            'complications': [],
            'departments': [],
            'drugs': []
        }
        
        # 从内容中提取医疗信息
        content_text = response.css('.main-content').get('')
        
        # 提取症状
        if '症状' in content_text:
            symptoms_section = response.xpath('//h2[contains(text(), "症状")]/following-sibling::div[1]')
            medical_info['symptoms'] = self._extract_list_items(symptoms_section)
        
        # 提取病因
        if '病因' in content_text or '原因' in content_text:
            causes_section = response.xpath('//h2[contains(text(), "病因") or contains(text(), "原因")]/following-sibling::div[1]')
            medical_info['causes'] = self._extract_list_items(causes_section)
        
        # 提取治疗方法
        if '治疗' in content_text:
            treatment_section = response.xpath('//h2[contains(text(), "治疗")]/following-sibling::div[1]')
            medical_info['treatments'] = self._extract_list_items(treatment_section)
        
        # 从信息框提取科室信息
        infobox = self._extract_infobox(response)
        if '就诊科室' in infobox:
            medical_info['departments'] = [dept.strip() for dept in infobox['就诊科室'].split('，')]
        
        return medical_info
    
    def _extract_list_items(self, selector) -> List[str]:
        """从选择器中提取列表项"""
        items = []
        # 提取有序列表
        items.extend(selector.css('ol li::text').getall())
        # 提取无序列表
        items.extend(selector.css('ul li::text').getall())
        # 提取段落中的要点
        for para in selector.css('.para').getall():
            # 查找"1."、"①"等模式
            matches = re.findall(r'[1-9一二三四五六七八九①②③④⑤⑥⑦⑧⑨][.、）](.*?)[。；]', para)
            items.extend(matches)
        
        return [item.strip() for item in items if item.strip()]
    
    def _extract_categories(self, response) -> List[str]:
        """提取分类标签"""
        return response.css('.taglist .tagItem::text').getall()
    
    def _extract_related_links(self, response) -> List[str]:
        """提取相关链接"""
        links = []
        # 从正文中提取链接
        links.extend(response.css('.main-content a::attr(href)').getall())
        # 从相关词条中提取
        links.extend(response.css('.rs-container-foot a::attr(href)').getall())
        return links
    
    def _is_medical_related(self, url: str) -> bool:
        """判断链接是否与医疗相关"""
        # 检查URL中是否包含医疗关键词
        medical_keywords = ['病', '症', '药', '医', '疗', '治', '诊']
        return any(keyword in url for keyword in medical_keywords)


class MedicalPDFExtractor:
    """医学文献PDF提取器"""
    
    def __init__(self):
        import pdfplumber
        self.pdfplumber = pdfplumber
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """从PDF中提取医学信息"""
        extracted_data = {
            'title': '',
            'abstract': '',
            'keywords': [],
            'sections': {},
            'references': []
        }
        
        with self.pdfplumber.open(pdf_path) as pdf:
            # 提取标题（通常在第一页）
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            lines = text.split('\n')
            
            # 假设标题是第一页最大的文本
            extracted_data['title'] = lines[0] if lines else ''
            
            # 提取摘要
            abstract_start = text.find('Abstract') or text.find('摘要')
            if abstract_start != -1:
                abstract_end = text.find('Keywords', abstract_start) or text.find('关键词', abstract_start)
                if abstract_end != -1:
                    extracted_data['abstract'] = text[abstract_start:abstract_end].strip()
            
            # 提取全文
            full_text = ''
            for page in pdf.pages:
                full_text += page.extract_text() + '\n'
            
            # 按章节分割
            sections = self._split_into_sections(full_text)
            extracted_data['sections'] = sections
            
        return extracted_data
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """将文本分割成章节"""
        sections = {}
        # 常见的医学论文章节标题模式
        section_patterns = [
            r'\n\s*(\d+\.?\s*[A-Za-z\u4e00-\u9fff\s]+)\n',  # 1. Introduction 或 1. 引言
            r'\n\s*([A-Z][A-Za-z\s]+)\n',  # INTRODUCTION
            r'\n\s*([一二三四五六七八九十]+[、.]\s*[\u4e00-\u9fff]+)\n'  # 一、引言
        ]
        
        current_section = 'Abstract'
        current_content = ''
        
        lines = text.split('\n')
        for line in lines:
            is_section_title = False
            for pattern in section_patterns:
                if re.match(pattern, '\n' + line + '\n'):
                    # 保存前一个章节
                    if current_content:
                        sections[current_section] = current_content.strip()
                    # 开始新章节
                    current_section = line.strip()
                    current_content = ''
                    is_section_title = True
                    break
            
            if not is_section_title:
                current_content += line + '\n'
        
        # 保存最后一个章节
        if current_content:
            sections[current_section] = current_content.strip()
        
        return sections


# 使用示例
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    
    # 配置爬虫设置
    settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 1,
        'COOKIES_ENABLED': False,
        'FEEDS': {
            'data/raw/medical_baike.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 4
            }
        }
    }
    
    # 运行爬虫
    process = CrawlerProcess(settings)
    process.crawl(MedicalBaikeSpider)
    process.start()