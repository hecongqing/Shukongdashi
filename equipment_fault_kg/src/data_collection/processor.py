"""
数据处理模块

负责对采集的原始数据进行清洗、预处理和格式化
"""

import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import jieba
from loguru import logger


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_jieba()
    
    def setup_jieba(self):
        """设置jieba分词"""
        # 添加装备制造领域的专业词汇
        equipment_terms = [
            '数控机床', '加工中心', '车床', '铣床', '钻床', '磨床',
            '主轴', '进给系统', '伺服电机', '编码器', '传感器',
            '液压系统', '润滑系统', '冷却系统', '电气系统',
            'PLC', 'CNC', '驱动器', '变频器', '变压器'
        ]
        
        for term in equipment_terms:
            jieba.add_word(term)
    
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:()（）\-_]+', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def extract_equipment_info(self, text: str) -> Dict[str, Any]:
        """提取装备信息"""
        equipment_info = {
            'equipment_type': None,
            'model': None,
            'manufacturer': None,
            'components': []
        }
        
        # 装备类型识别
        equipment_types = [
            '数控机床', '加工中心', '车床', '铣床', '钻床', '磨床',
            '镗床', '刨床', '拉床', '锯床', '冲床'
        ]
        
        for eq_type in equipment_types:
            if eq_type in text:
                equipment_info['equipment_type'] = eq_type
                break
        
        # 型号识别（通常是字母数字组合）
        model_pattern = r'[A-Z]{2,}\d+[A-Z]*'
        models = re.findall(model_pattern, text)
        if models:
            equipment_info['model'] = models[0]
        
        # 制造商识别
        manufacturers = [
            '西门子', '发那科', '三菱', '海德汉', '马扎克', '德马吉',
            '森精机', '大隈', '东芝', '日立', '安川', '松下'
        ]
        
        for manufacturer in manufacturers:
            if manufacturer in text:
                equipment_info['manufacturer'] = manufacturer
                break
        
        # 部件识别
        components = [
            '主轴', '进给系统', '伺服电机', '编码器', '传感器',
            '液压系统', '润滑系统', '冷却系统', '电气系统',
            '刀库', '工作台', '导轨', '丝杠', '轴承'
        ]
        
        for component in components:
            if component in text:
                equipment_info['components'].append(component)
        
        return equipment_info
    
    def extract_fault_info(self, text: str) -> Dict[str, Any]:
        """提取故障信息"""
        fault_info = {
            'fault_type': None,
            'symptoms': [],
            'causes': [],
            'solutions': []
        }
        
        # 故障类型识别
        fault_types = [
            '机械故障', '电气故障', '液压故障', '润滑故障',
            '冷却故障', '控制系统故障', '传感器故障', '通信故障'
        ]
        
        for fault_type in fault_types:
            if fault_type in text:
                fault_info['fault_type'] = fault_type
                break
        
        # 症状识别
        symptom_keywords = ['异常', '故障', '报警', '错误', '停机', '振动', '噪音', '发热']
        sentences = text.split('。')
        
        for sentence in sentences:
            for keyword in symptom_keywords:
                if keyword in sentence:
                    fault_info['symptoms'].append(sentence.strip())
                    break
        
        # 原因识别
        cause_keywords = ['原因', '导致', '引起', '造成', '由于', '因为']
        for sentence in sentences:
            for keyword in cause_keywords:
                if keyword in sentence:
                    fault_info['causes'].append(sentence.strip())
                    break
        
        # 解决方案识别
        solution_keywords = ['解决', '修复', '更换', '调整', '维修', '处理']
        for sentence in sentences:
            for keyword in solution_keywords:
                if keyword in sentence:
                    fault_info['solutions'].append(sentence.strip())
                    break
        
        return fault_info
    
    def process_fault_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个故障案例"""
        processed_case = {
            'id': case_data.get('id', ''),
            'title': self.clean_text(case_data.get('title', '')),
            'content': self.clean_text(case_data.get('content', '')),
            'source': case_data.get('source', ''),
            'url': case_data.get('url', ''),
            'equipment_info': {},
            'fault_info': {},
            'processed_at': pd.Timestamp.now().isoformat()
        }
        
        # 提取装备信息
        full_text = f"{processed_case['title']} {processed_case['content']}"
        processed_case['equipment_info'] = self.extract_equipment_info(full_text)
        
        # 提取故障信息
        processed_case['fault_info'] = self.extract_fault_info(full_text)
        
        return processed_case
    
    def process_batch(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理数据"""
        processed_data = []
        
        for i, data in enumerate(data_list):
            try:
                processed_item = self.process_fault_case(data)
                processed_data.append(processed_item)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(data_list)} 条数据")
                    
            except Exception as e:
                logger.error(f"处理第{i+1}条数据时出错: {e}")
                continue
        
        logger.info(f"批量处理完成，共处理 {len(processed_data)} 条数据")
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """保存处理后的数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为CSV格式（扁平化）
        df_data = []
        for item in data:
            flat_item = {
                'id': item['id'],
                'title': item['title'],
                'content': item['content'],
                'source': item['source'],
                'url': item['url'],
                'equipment_type': item['equipment_info'].get('equipment_type'),
                'model': item['equipment_info'].get('model'),
                'manufacturer': item['equipment_info'].get('manufacturer'),
                'components': ','.join(item['equipment_info'].get('components', [])),
                'fault_type': item['fault_info'].get('fault_type'),
                'symptoms': ';'.join(item['fault_info'].get('symptoms', [])),
                'causes': ';'.join(item['fault_info'].get('causes', [])),
                'solutions': ';'.join(item['fault_info'].get('solutions', [])),
                'processed_at': item['processed_at']
            }
            df_data.append(flat_item)
        
        df = pd.DataFrame(df_data)
        csv_path = output_file.with_suffix('.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"数据已保存到: {output_file} 和 {csv_path}")
    
    def generate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成数据统计信息"""
        stats = {
            'total_cases': len(data),
            'equipment_types': {},
            'fault_types': {},
            'manufacturers': {},
            'sources': {}
        }
        
        for item in data:
            # 装备类型统计
            eq_type = item['equipment_info'].get('equipment_type')
            if eq_type:
                stats['equipment_types'][eq_type] = stats['equipment_types'].get(eq_type, 0) + 1
            
            # 故障类型统计
            fault_type = item['fault_info'].get('fault_type')
            if fault_type:
                stats['fault_types'][fault_type] = stats['fault_types'].get(fault_type, 0) + 1
            
            # 制造商统计
            manufacturer = item['equipment_info'].get('manufacturer')
            if manufacturer:
                stats['manufacturers'][manufacturer] = stats['manufacturers'].get(manufacturer, 0) + 1
            
            # 数据源统计
            source = item['source']
            if source:
                stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats