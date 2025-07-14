#!/usr/bin/env python3
"""
数据采集脚本
包含PDF文档解析、网页爬取、数据库导入等功能
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from loguru import logger

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.config.settings import get_settings
from backend.services.data_collection_service import DataCollectionService
from backend.utils.file_utils import FileProcessor
from backend.utils.text_utils import TextPreprocessor

class DataCollector:
    """数据采集器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_service = DataCollectionService()
        self.file_processor = FileProcessor()
        self.text_preprocessor = TextPreprocessor()
        
        # 创建数据目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.settings.RAW_DATA_DIR,
            self.settings.PROCESSED_DATA_DIR,
            self.settings.KNOWLEDGE_DATA_DIR,
            f"{self.settings.RAW_DATA_DIR}/pdf",
            f"{self.settings.RAW_DATA_DIR}/web",
            f"{self.settings.RAW_DATA_DIR}/excel"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def collect_pdf_data(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """
        采集PDF文档数据
        
        Args:
            pdf_paths: PDF文件路径列表
            
        Returns:
            List[Dict[str, Any]]: 抽取的数据列表
        """
        logger.info(f"开始采集PDF数据，共{len(pdf_paths)}个文件")
        
        all_data = []
        
        for pdf_path in pdf_paths:
            try:
                logger.info(f"处理PDF文件: {pdf_path}")
                
                # 解析PDF文件
                pdf_data = await self.file_processor.parse_pdf(pdf_path)
                
                # 提取故障案例
                fault_cases = self._extract_fault_cases(pdf_data)
                
                # 保存原始数据
                output_path = f"{self.settings.RAW_DATA_DIR}/pdf/{Path(pdf_path).stem}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(fault_cases, f, ensure_ascii=False, indent=2)
                
                all_data.extend(fault_cases)
                logger.info(f"从{pdf_path}中提取了{len(fault_cases)}个故障案例")
                
            except Exception as e:
                logger.error(f"处理PDF文件{pdf_path}失败: {e}")
                continue
        
        logger.info(f"PDF数据采集完成，共收集{len(all_data)}条数据")
        return all_data
    
    def _extract_fault_cases(self, pdf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从PDF数据中提取故障案例"""
        fault_cases = []
        
        pages = pdf_data.get('pages', [])
        
        for page_num, page_content in enumerate(pages):
            text = page_content.get('text', '')
            
            # 按故障案例分割文本
            cases = self._split_by_fault_cases(text)
            
            for case_num, case_text in enumerate(cases):
                if len(case_text.strip()) < 50:  # 过滤太短的文本
                    continue
                
                fault_case = {
                    'id': f"pdf_{page_num}_{case_num}",
                    'source': pdf_data.get('filename', 'unknown'),
                    'page': page_num + 1,
                    'raw_text': case_text,
                    'processed_text': self.text_preprocessor.clean_text(case_text),
                    'timestamp': datetime.now().isoformat(),
                    'data_type': 'pdf'
                }
                
                # 解析故障信息
                fault_info = self._parse_fault_info(case_text)
                fault_case.update(fault_info)
                
                fault_cases.append(fault_case)
        
        return fault_cases
    
    def _split_by_fault_cases(self, text: str) -> List[str]:
        """按故障案例分割文本"""
        import re
        
        # 故障案例分割模式
        patterns = [
            r'案例\d+',
            r'故障\d+',
            r'例\d+',
            r'\d+\.\s*故障',
            r'\d+\.\s*案例'
        ]
        
        # 找到所有分割点
        split_points = []
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                split_points.append(match.start())
        
        # 排序并去重
        split_points = sorted(set(split_points))
        
        # 分割文本
        cases = []
        for i in range(len(split_points)):
            start = split_points[i]
            end = split_points[i + 1] if i + 1 < len(split_points) else len(text)
            case_text = text[start:end].strip()
            if case_text:
                cases.append(case_text)
        
        return cases
    
    def _parse_fault_info(self, text: str) -> Dict[str, Any]:
        """解析故障信息"""
        import re
        
        fault_info = {
            'equipment_brand': None,
            'equipment_model': None,
            'fault_phenomenon': None,
            'fault_cause': None,
            'repair_method': None,
            'alarm_code': None
        }
        
        # 设备品牌模式
        brand_patterns = [
            r'(发那科|FANUC|fanuc)',
            r'(西门子|SIEMENS|siemens)',
            r'(三菱|MITSUBISHI|mitsubishi)',
            r'(海德汉|HEIDENHAIN|heidenhain)',
            r'(华中数控|HNC|hnc)'
        ]
        
        for pattern in brand_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fault_info['equipment_brand'] = match.group(1)
                break
        
        # 设备型号模式
        model_patterns = [
            r'([A-Z]+-\w+)',
            r'(\w+[-_]\w+)',
            r'(0i[-_]?\w*)',
            r'(840D\w*)',
            r'(808D\w*)'
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, text)
            if match:
                fault_info['equipment_model'] = match.group(1)
                break
        
        # 报警代码模式
        alarm_patterns = [
            r'(ALM\d+)',
            r'(ERR\d+)',
            r'(报警\d+)',
            r'(错误代码\d+)'
        ]
        
        for pattern in alarm_patterns:
            match = re.search(pattern, text)
            if match:
                fault_info['alarm_code'] = match.group(1)
                break
        
        # 故障现象（简单提取）
        phenomenon_keywords = ['故障现象', '现象', '表现', '症状']
        for keyword in phenomenon_keywords:
            pattern = f'{keyword}[：:](.*?)(?=故障|原因|解决|$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                fault_info['fault_phenomenon'] = match.group(1).strip()
                break
        
        # 故障原因
        cause_keywords = ['故障原因', '原因', '分析']
        for keyword in cause_keywords:
            pattern = f'{keyword}[：:](.*?)(?=解决|处理|$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                fault_info['fault_cause'] = match.group(1).strip()
                break
        
        # 维修方法
        repair_keywords = ['解决方法', '处理方法', '维修', '解决']
        for keyword in repair_keywords:
            pattern = f'{keyword}[：:](.*?)(?=$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                fault_info['repair_method'] = match.group(1).strip()
                break
        
        return fault_info
    
    async def collect_web_data(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        采集网页数据
        
        Args:
            urls: 网页URL列表
            
        Returns:
            List[Dict[str, Any]]: 抽取的数据列表
        """
        logger.info(f"开始采集网页数据，共{len(urls)}个URL")
        
        all_data = []
        
        for url in urls:
            try:
                logger.info(f"爬取网页: {url}")
                
                # 爬取网页
                web_data = await self.data_service.crawl_web_page(url)
                
                # 提取故障信息
                fault_data = self._extract_web_fault_data(web_data)
                
                # 保存原始数据
                safe_filename = url.replace('/', '_').replace(':', '_')
                output_path = f"{self.settings.RAW_DATA_DIR}/web/{safe_filename}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(fault_data, f, ensure_ascii=False, indent=2)
                
                all_data.extend(fault_data)
                logger.info(f"从{url}中提取了{len(fault_data)}条数据")
                
                # 避免过于频繁的请求
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"爬取网页{url}失败: {e}")
                continue
        
        logger.info(f"网页数据采集完成，共收集{len(all_data)}条数据")
        return all_data
    
    def _extract_web_fault_data(self, web_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从网页数据中提取故障信息"""
        fault_data = []
        
        text = web_data.get('text', '')
        title = web_data.get('title', '')
        url = web_data.get('url', '')
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < 30:  # 过滤太短的段落
                continue
            
            # 检查是否包含故障相关内容
            if not self._contains_fault_keywords(paragraph):
                continue
            
            fault_item = {
                'id': f"web_{hash(url)}_{i}",
                'source': url,
                'title': title,
                'raw_text': paragraph,
                'processed_text': self.text_preprocessor.clean_text(paragraph),
                'timestamp': datetime.now().isoformat(),
                'data_type': 'web'
            }
            
            # 解析故障信息
            fault_info = self._parse_fault_info(paragraph)
            fault_item.update(fault_info)
            
            fault_data.append(fault_item)
        
        return fault_data
    
    def _contains_fault_keywords(self, text: str) -> bool:
        """检查文本是否包含故障相关关键词"""
        fault_keywords = [
            '故障', '报警', '错误', '异常', '维修', '修理', '检修',
            '问题', '不能', '无法', '失效', '损坏', '停机', '中断'
        ]
        
        return any(keyword in text for keyword in fault_keywords)
    
    async def collect_excel_data(self, excel_paths: List[str]) -> List[Dict[str, Any]]:
        """
        采集Excel数据
        
        Args:
            excel_paths: Excel文件路径列表
            
        Returns:
            List[Dict[str, Any]]: 抽取的数据列表
        """
        logger.info(f"开始采集Excel数据，共{len(excel_paths)}个文件")
        
        all_data = []
        
        for excel_path in excel_paths:
            try:
                logger.info(f"处理Excel文件: {excel_path}")
                
                # 读取Excel文件
                df = pd.read_excel(excel_path)
                
                # 转换为字典列表
                excel_data = df.to_dict('records')
                
                # 标准化数据格式
                standardized_data = self._standardize_excel_data(excel_data, excel_path)
                
                # 保存原始数据
                output_path = f"{self.settings.RAW_DATA_DIR}/excel/{Path(excel_path).stem}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(standardized_data, f, ensure_ascii=False, indent=2)
                
                all_data.extend(standardized_data)
                logger.info(f"从{excel_path}中提取了{len(standardized_data)}条数据")
                
            except Exception as e:
                logger.error(f"处理Excel文件{excel_path}失败: {e}")
                continue
        
        logger.info(f"Excel数据采集完成，共收集{len(all_data)}条数据")
        return all_data
    
    def _standardize_excel_data(self, excel_data: List[Dict], file_path: str) -> List[Dict[str, Any]]:
        """标准化Excel数据格式"""
        standardized_data = []
        
        for i, row in enumerate(excel_data):
            # 跳过空行
            if all(pd.isna(value) for value in row.values()):
                continue
            
            # 创建标准化的数据项
            data_item = {
                'id': f"excel_{Path(file_path).stem}_{i}",
                'source': file_path,
                'raw_data': row,
                'timestamp': datetime.now().isoformat(),
                'data_type': 'excel'
            }
            
            # 尝试映射常见字段
            field_mapping = {
                '设备品牌': 'equipment_brand',
                '设备型号': 'equipment_model',
                '故障现象': 'fault_phenomenon',
                '故障原因': 'fault_cause',
                '维修方法': 'repair_method',
                '报警代码': 'alarm_code'
            }
            
            for excel_field, standard_field in field_mapping.items():
                if excel_field in row and not pd.isna(row[excel_field]):
                    data_item[standard_field] = str(row[excel_field])
            
            # 合并所有文本内容
            text_content = ' '.join(str(v) for v in row.values() if not pd.isna(v))
            data_item['raw_text'] = text_content
            data_item['processed_text'] = self.text_preprocessor.clean_text(text_content)
            
            standardized_data.append(data_item)
        
        return standardized_data
    
    async def merge_and_save_data(self, all_data: List[Dict[str, Any]]):
        """合并并保存所有数据"""
        logger.info(f"开始合并数据，共{len(all_data)}条记录")
        
        # 去重
        seen_ids = set()
        unique_data = []
        
        for item in all_data:
            item_id = item.get('id')
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_data.append(item)
        
        logger.info(f"去重后保留{len(unique_data)}条记录")
        
        # 保存合并后的数据
        output_path = f"{self.settings.PROCESSED_DATA_DIR}/all_fault_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        
        # 保存为CSV格式
        df = pd.DataFrame(unique_data)
        csv_path = f"{self.settings.PROCESSED_DATA_DIR}/all_fault_data.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"数据已保存到: {output_path} 和 {csv_path}")
    
    async def run_collection(self):
        """运行数据采集流程"""
        logger.info("开始数据采集流程")
        
        all_data = []
        
        # 1. 采集PDF数据
        pdf_dir = f"{self.settings.RAW_DATA_DIR}/pdf_sources"
        if Path(pdf_dir).exists():
            pdf_files = list(Path(pdf_dir).glob("*.pdf"))
            if pdf_files:
                pdf_data = await self.collect_pdf_data([str(f) for f in pdf_files])
                all_data.extend(pdf_data)
        
        # 2. 采集网页数据
        web_urls = [
            # 添加需要爬取的URL
            "https://example.com/fault-cases",
        ]
        if web_urls:
            web_data = await self.collect_web_data(web_urls)
            all_data.extend(web_data)
        
        # 3. 采集Excel数据
        excel_dir = f"{self.settings.RAW_DATA_DIR}/excel_sources"
        if Path(excel_dir).exists():
            excel_files = list(Path(excel_dir).glob("*.xlsx")) + list(Path(excel_dir).glob("*.xls"))
            if excel_files:
                excel_data = await self.collect_excel_data([str(f) for f in excel_files])
                all_data.extend(excel_data)
        
        # 4. 合并并保存数据
        if all_data:
            await self.merge_and_save_data(all_data)
        else:
            logger.warning("没有采集到任何数据")
        
        logger.info("数据采集流程完成")

async def main():
    """主函数"""
    collector = DataCollector()
    await collector.run_collection()

if __name__ == "__main__":
    asyncio.run(main())