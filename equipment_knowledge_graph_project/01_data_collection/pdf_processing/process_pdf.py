#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF文档处理模块
用于提取装备制造故障相关的PDF文档内容
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import PyPDF2
import pdfplumber
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self, pdf_dir: str = "../data/pdfs"):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_data = []
        
    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取文本"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
                return text
                
        except Exception as e:
            logger.error(f"PyPDF2提取失败 {pdf_path}: {e}")
            return ""
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取文本"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
            return text
            
        except Exception as e:
            logger.error(f"pdfplumber提取失败 {pdf_path}: {e}")
            return ""
    
    def extract_text_with_tables(self, pdf_path: str) -> Tuple[str, List[pd.DataFrame]]:
        """提取文本和表格"""
        try:
            text = ""
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # 提取表格
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
                            
            return text, tables
            
        except Exception as e:
            logger.error(f"提取文本和表格失败 {pdf_path}: {e}")
            return "", []
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
            
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff。，、；：！？""''（）【】《》]', '', text)
        
        # 移除页码
        text = re.sub(r'\d+\s*页', '', text)
        
        return text.strip()
    
    def extract_fault_cases(self, text: str) -> List[Dict]:
        """提取故障案例"""
        fault_cases = []
        
        # 定义故障案例的模式
        patterns = [
            r'故障现象[：:]\s*(.*?)(?=故障原因|排除方法|$)',
            r'故障原因[：:]\s*(.*?)(?=排除方法|故障现象|$)',
            r'排除方法[：:]\s*(.*?)(?=故障现象|故障原因|$)',
            r'案例\d+[：:]\s*(.*?)(?=案例\d+|$)',
            r'故障\d+[：:]\s*(.*?)(?=故障\d+|$)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                case_text = match.group(1).strip()
                if len(case_text) > 10:  # 过滤太短的文本
                    fault_cases.append({
                        'type': 'fault_case',
                        'content': case_text,
                        'pattern': pattern,
                        'position': match.start()
                    })
        
        return fault_cases
    
    def extract_equipment_info(self, text: str) -> List[Dict]:
        """提取设备信息"""
        equipment_info = []
        
        # 设备型号模式
        model_patterns = [
            r'型号[：:]\s*([A-Z0-9\-]+)',
            r'([A-Z]{2,}\d+[A-Z0-9\-]*)',
            r'([A-Z]+-\d+[A-Z0-9\-]*)'
        ]
        
        for pattern in model_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                model = match.group(1)
                if len(model) > 2:
                    equipment_info.append({
                        'type': 'equipment_model',
                        'content': model,
                        'pattern': pattern
                    })
        
        # 故障代码模式
        error_patterns = [
            r'报警[：:]\s*([A-Z0-9]+)',
            r'错误代码[：:]\s*([A-Z0-9]+)',
            r'([A-Z]{2,}\d{3,})'
        ]
        
        for pattern in error_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                error_code = match.group(1)
                equipment_info.append({
                    'type': 'error_code',
                    'content': error_code,
                    'pattern': pattern
                })
        
        return equipment_info
    
    def extract_maintenance_procedures(self, text: str) -> List[Dict]:
        """提取维护程序"""
        procedures = []
        
        # 维护步骤模式
        step_patterns = [
            r'步骤\d+[：:]\s*(.*?)(?=步骤\d+|$)',
            r'\d+[、.．]\s*(.*?)(?=\d+[、.．]|$)',
            r'首先[，,]\s*(.*?)(?=然后|其次|最后|$)',
            r'然后[，,]\s*(.*?)(?=最后|其次|$)'
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                step_text = match.group(1).strip()
                if len(step_text) > 5:
                    procedures.append({
                        'type': 'maintenance_step',
                        'content': step_text,
                        'pattern': pattern
                    })
        
        return procedures
    
    def process_pdf_file(self, pdf_path: str) -> Dict:
        """处理单个PDF文件"""
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        # 提取文本
        text_pypdf2 = self.extract_text_with_pypdf2(pdf_path)
        text_pdfplumber = self.extract_text_with_pdfplumber(pdf_path)
        
        # 选择更好的提取结果
        if len(text_pdfplumber) > len(text_pypdf2):
            text = text_pdfplumber
        else:
            text = text_pypdf2
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        
        # 提取结构化信息
        fault_cases = self.extract_fault_cases(cleaned_text)
        equipment_info = self.extract_equipment_info(cleaned_text)
        maintenance_procedures = self.extract_maintenance_procedures(cleaned_text)
        
        # 提取表格
        _, tables = self.extract_text_with_tables(pdf_path)
        
        result = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'processed_at': datetime.now().isoformat(),
            'text_length': len(cleaned_text),
            'fault_cases': fault_cases,
            'equipment_info': equipment_info,
            'maintenance_procedures': maintenance_procedures,
            'tables_count': len(tables),
            'raw_text': cleaned_text[:10000],  # 保存前10000字符的原始文本
            'tables': [df.to_dict('records') for df in tables]
        }
        
        logger.info(f"PDF处理完成: {pdf_path}, 提取了 {len(fault_cases)} 个故障案例")
        return result
    
    def process_all_pdfs(self) -> List[Dict]:
        """处理所有PDF文件"""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"在目录 {self.pdf_dir} 中没有找到PDF文件")
            return []
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf_file(str(pdf_file))
                self.extracted_data.append(result)
            except Exception as e:
                logger.error(f"处理PDF文件失败 {pdf_file}: {e}")
        
        return self.extracted_data
    
    def save_results(self, output_file: str = None):
        """保存处理结果"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"../data/pdf_extracted_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"PDF处理结果已保存到: {output_file}")
        
        # 生成统计报告
        self.generate_statistics_report(output_file.replace('.json', '_statistics.txt'))
    
    def generate_statistics_report(self, report_file: str):
        """生成统计报告"""
        total_files = len(self.extracted_data)
        total_fault_cases = sum(len(item['fault_cases']) for item in self.extracted_data)
        total_equipment_info = sum(len(item['equipment_info']) for item in self.extracted_data)
        total_procedures = sum(len(item['maintenance_procedures']) for item in self.extracted_data)
        total_tables = sum(item['tables_count'] for item in self.extracted_data)
        
        report = f"""
PDF处理统计报告
================
处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
处理文件数: {total_files}
提取故障案例数: {total_fault_cases}
提取设备信息数: {total_equipment_info}
提取维护程序数: {total_procedures}
提取表格数: {total_tables}

文件详情:
"""
        
        for item in self.extracted_data:
            report += f"""
文件: {item['file_name']}
- 文本长度: {item['text_length']}
- 故障案例: {len(item['fault_cases'])}
- 设备信息: {len(item['equipment_info'])}
- 维护程序: {len(item['maintenance_procedures'])}
- 表格数: {item['tables_count']}
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"统计报告已保存到: {report_file}")
    
    def export_to_csv(self, output_dir: str = "../data/csv"):
        """导出为CSV格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出故障案例
        fault_cases_data = []
        for item in self.extracted_data:
            for case in item['fault_cases']:
                fault_cases_data.append({
                    'file_name': item['file_name'],
                    'content': case['content'],
                    'type': case['type']
                })
        
        if fault_cases_data:
            df_fault = pd.DataFrame(fault_cases_data)
            df_fault.to_csv(f"{output_dir}/fault_cases.csv", index=False, encoding='utf-8-sig')
        
        # 导出设备信息
        equipment_data = []
        for item in self.extracted_data:
            for info in item['equipment_info']:
                equipment_data.append({
                    'file_name': item['file_name'],
                    'content': info['content'],
                    'type': info['type']
                })
        
        if equipment_data:
            df_equipment = pd.DataFrame(equipment_data)
            df_equipment.to_csv(f"{output_dir}/equipment_info.csv", index=False, encoding='utf-8-sig')
        
        # 导出维护程序
        procedures_data = []
        for item in self.extracted_data:
            for proc in item['maintenance_procedures']:
                procedures_data.append({
                    'file_name': item['file_name'],
                    'content': proc['content'],
                    'type': proc['type']
                })
        
        if procedures_data:
            df_procedures = pd.DataFrame(procedures_data)
            df_procedures.to_csv(f"{output_dir}/maintenance_procedures.csv", index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV文件已导出到: {output_dir}")

def main():
    """主函数"""
    # 创建PDF处理器
    processor = PDFProcessor()
    
    # 处理所有PDF文件
    results = processor.process_all_pdfs()
    
    if results:
        # 保存结果
        processor.save_results()
        
        # 导出CSV
        processor.export_to_csv()
        
        print(f"PDF处理完成，共处理 {len(results)} 个文件")
    else:
        print("没有找到PDF文件或处理失败")

if __name__ == "__main__":
    main()