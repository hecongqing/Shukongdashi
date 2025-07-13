"""
PDF文档解析模块
支持PDF文本提取、表格识别、图片提取等功能
"""

import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from PIL import Image
import io

from config.settings import settings


class PDFParser:
    """PDF文档解析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_pypdf2(self, pdf_path: str) -> Dict[str, Any]:
        """
        使用PyPDF2提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                text_content = []
                metadata = pdf_reader.metadata
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_content.append({
                        'page': page_num + 1,
                        'text': text,
                        'length': len(text)
                    })
                
                return {
                    'file_path': pdf_path,
                    'total_pages': len(pdf_reader.pages),
                    'metadata': metadata,
                    'pages': text_content,
                    'total_text': '\n'.join([page['text'] for page in text_content])
                }
                
        except Exception as e:
            self.logger.error(f"PyPDF2解析失败 {pdf_path}: {str(e)}")
            return {'file_path': pdf_path, 'error': str(e)}
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """
        使用pdfplumber提取文本和表格
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本和表格内容
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_content = []
                tables_content = []
                
                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    text = page.extract_text()
                    
                    # 提取表格
                    tables = page.extract_tables()
                    
                    page_content = {
                        'page': page_num + 1,
                        'text': text,
                        'tables': tables,
                        'length': len(text) if text else 0
                    }
                    
                    pages_content.append(page_content)
                    
                    # 处理表格
                    for table_num, table in enumerate(tables):
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables_content.append({
                                'page': page_num + 1,
                                'table': table_num + 1,
                                'data': table,
                                'dataframe': df.to_dict('records')
                            })
                
                return {
                    'file_path': pdf_path,
                    'total_pages': len(pdf.pages),
                    'pages': pages_content,
                    'tables': tables_content,
                    'total_text': '\n'.join([page['text'] for page in pages_content if page['text']])
                }
                
        except Exception as e:
            self.logger.error(f"pdfplumber解析失败 {pdf_path}: {str(e)}")
            return {'file_path': pdf_path, 'error': str(e)}
    
    def extract_text_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        使用PyMuPDF提取文本和图片
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本和图片内容
        """
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            images_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本
                text = page.get_text()
                
                # 提取图片
                image_list = page.get_images()
                page_images = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # 灰度或RGB
                        img_data = pix.tobytes("png")
                        page_images.append({
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'data': img_data
                        })
                    
                    pix = None
                
                page_content = {
                    'page': page_num + 1,
                    'text': text,
                    'images': page_images,
                    'length': len(text)
                }
                
                pages_content.append(page_content)
                images_content.extend(page_images)
            
            doc.close()
            
            return {
                'file_path': pdf_path,
                'total_pages': len(doc),
                'pages': pages_content,
                'images': images_content,
                'total_text': '\n'.join([page['text'] for page in pages_content])
            }
            
        except Exception as e:
            self.logger.error(f"PyMuPDF解析失败 {pdf_path}: {str(e)}")
            return {'file_path': pdf_path, 'error': str(e)}
    
    def extract_structured_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        提取结构化内容（标题、段落、列表等）
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            结构化内容
        """
        try:
            # 使用pdfplumber提取文本
            content = self.extract_text_pdfplumber(pdf_path)
            
            if 'error' in content:
                return content
            
            structured_content = {
                'file_path': pdf_path,
                'sections': [],
                'tables': content.get('tables', []),
                'metadata': {}
            }
            
            # 分析文本结构
            full_text = content['total_text']
            
            # 分割章节
            sections = self._split_into_sections(full_text)
            
            for section in sections:
                structured_section = {
                    'title': section.get('title', ''),
                    'content': section.get('content', ''),
                    'type': section.get('type', 'text'),
                    'level': section.get('level', 1)
                }
                structured_content['sections'].append(structured_section)
            
            return structured_content
            
        except Exception as e:
            self.logger.error(f"结构化内容提取失败 {pdf_path}: {str(e)}")
            return {'file_path': pdf_path, 'error': str(e)}
    
    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        将文本分割为章节
        
        Args:
            text: 输入文本
            
        Returns:
            章节列表
        """
        sections = []
        
        # 匹配标题模式
        title_patterns = [
            r'^第[一二三四五六七八九十\d]+章\s*(.+)$',  # 第X章
            r'^\d+\.\d+\s+(.+)$',  # 1.1 标题
            r'^\d+\.\s+(.+)$',     # 1. 标题
            r'^[一二三四五六七八九十]+、\s*(.+)$',  # 一、标题
        ]
        
        lines = text.split('\n')
        current_section = {'title': '', 'content': '', 'type': 'text', 'level': 1}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是标题
            is_title = False
            title_level = 1
            
            for pattern in title_patterns:
                match = re.match(pattern, line)
                if match:
                    # 保存当前章节
                    if current_section['content']:
                        sections.append(current_section.copy())
                    
                    # 开始新章节
                    current_section = {
                        'title': match.group(1) if len(match.groups()) > 0 else line,
                        'content': '',
                        'type': 'section',
                        'level': title_level
                    }
                    is_title = True
                    break
                title_level += 1
            
            if not is_title:
                current_section['content'] += line + '\n'
        
        # 添加最后一个章节
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def save_extracted_content(self, content: Dict[str, Any], output_path: str):
        """
        保存提取的内容
        
        Args:
            content: 提取的内容
            output_path: 输出路径
        """
        output_file = Path(settings.PROCESSED_DATA_DIR) / output_path
        
        if output_path.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        elif output_path.endswith('.txt'):
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content.get('total_text', ''))
        elif output_path.endswith('.csv'):
            # 保存表格数据
            tables = content.get('tables', [])
            if tables:
                all_tables = []
                for table in tables:
                    df = pd.DataFrame(table['data'])
                    all_tables.append(df)
                
                with pd.ExcelWriter(output_file) as writer:
                    for i, df in enumerate(all_tables):
                        df.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)
        
        self.logger.info(f"内容已保存到: {output_file}")
    
    def batch_process_pdfs(self, pdf_dir: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        批量处理PDF文件
        
        Args:
            pdf_dir: PDF文件目录
            output_dir: 输出目录
            
        Returns:
            处理结果列表
        """
        pdf_path = Path(pdf_dir)
        results = []
        
        if output_dir is None:
            output_dir = settings.PROCESSED_DATA_DIR
        
        for pdf_file in pdf_path.glob('*.pdf'):
            self.logger.info(f"正在处理: {pdf_file}")
            
            # 提取结构化内容
            content = self.extract_structured_content(str(pdf_file))
            
            if 'error' not in content:
                # 保存结果
                output_name = f"{pdf_file.stem}_extracted.json"
                self.save_extracted_content(content, output_name)
            
            results.append(content)
        
        return results


# 使用示例
if __name__ == "__main__":
    parser = PDFParser()
    
    # 处理单个PDF文件
    pdf_path = "data/raw/example.pdf"
    if Path(pdf_path).exists():
        # 提取文本
        text_content = parser.extract_text_pdfplumber(pdf_path)
        print(f"提取了 {len(text_content.get('pages', []))} 页内容")
        
        # 提取结构化内容
        structured_content = parser.extract_structured_content(pdf_path)
        print(f"提取了 {len(structured_content.get('sections', []))} 个章节")
        
        # 保存结果
        parser.save_extracted_content(structured_content, "example_extracted.json")
    
    # 批量处理
    # results = parser.batch_process_pdfs("data/raw")
    # print(f"批量处理了 {len(results)} 个PDF文件")