"""
基于规则的装备制造领域实体抽取器

作为临时解决方案，使用规则和词典来抽取实体
"""

import re
from typing import List, Dict, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str
    start_pos: int
    end_pos: int


class RuleBasedEntityExtractor:
    """基于规则的实体抽取器"""
    
    def __init__(self):
        # 部件单元词典
        self.component_dict = {
            # 电机类
            '伺服电机', '步进电机', '直流电机', '交流电机', '电机', '马达',
            # 机械部件
            '轴承', '齿轮', '皮带', '链条', '联轴器', '离合器', '制动器',
            # 液压气动
            '液压泵', '气泵', '油泵', '水泵', '压缩机', '气缸', '液压缸',
            # 电气部件
            '电路', '电线', '电缆', '开关', '继电器', '接触器', '断路器',
            # 传感器
            '传感器', '压力传感器', '温度传感器', '流量传感器', '位移传感器',
            # 阀门
            '阀门', '球阀', '闸阀', '蝶阀', '止回阀', '安全阀',
            # 管道
            '管道', '软管', '接头', '法兰', '密封圈',
            # 其他
            '发动机', '变速箱', '传动轴', '车轮', '轮胎', '刹车片', '离合器片',
            '散热器', '水箱', '油箱', '滤清器', '过滤器', '消音器', '排气管'
        }
        
        # 性能表征词典
        self.performance_dict = {
            # 压力相关
            '压力', '油压', '气压', '水压', '液压', '系统压力', '工作压力',
            # 温度相关
            '温度', '油温', '水温', '气温', '工作温度', '运行温度', '环境温度',
            # 速度相关
            '转速', '速度', '线速度', '角速度', '运行速度', '工作速度',
            # 流量相关
            '流量', '油流量', '水流量', '气流量', '体积流量', '质量流量',
            # 电流电压
            '电流', '电压', '功率', '电阻', '电容', '电感',
            # 其他性能
            '精度', '准确度', '灵敏度', '分辨率', '响应时间', '效率', '功率因数'
        }
        
        # 故障状态词典
        self.fault_dict = {
            # 异常类
            '异常', '故障', '失效', '损坏', '断裂', '变形', '磨损', '腐蚀',
            # 运行问题
            '不启动', '启动困难', '运行不稳定', '振动', '噪音', '过热', '过载',
            # 泄漏类
            '泄漏', '漏油', '漏水', '漏气', '渗漏', '滴漏',
            # 堵塞类
            '堵塞', '卡死', '卡住', '卡滞', '卡阻',
            # 松动类
            '松动', '松旷', '间隙过大', '配合不良',
            # 其他故障
            '短路', '断路', '接触不良', '绝缘不良', '接地', '过流', '过压',
            '欠压', '缺相', '不平衡', '不对中', '不对齐'
        }
        
        # 检测工具词典
        self.tool_dict = {
            # 测量工具
            '万用表', '示波器', '频谱仪', '振动仪', '温度计', '压力表', '流量计',
            # 检测设备
            '检测仪', '测试仪', '分析仪', '诊断仪', '监控器', '传感器',
            # 专业工具
            '听诊器', '红外测温仪', '超声波检测仪', '磁粉探伤仪', '渗透检测剂',
            # 其他工具
            '扳手', '螺丝刀', '钳子', '锤子', '钻头', '砂轮', '焊机'
        }
        
        # 构建正则表达式模式
        self.patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[str, List[str]]:
        """构建正则表达式模式"""
        patterns = {
            'COMPONENT': [],
            'PERFORMANCE': [],
            'FAULT_STATE': [],
            'DETECTION_TOOL': []
        }
        
        # 为每个词典构建模式
        for component in self.component_dict:
            patterns['COMPONENT'].append(re.escape(component))
        
        for performance in self.performance_dict:
            patterns['PERFORMANCE'].append(re.escape(performance))
        
        for fault in self.fault_dict:
            patterns['FAULT_STATE'].append(re.escape(fault))
        
        for tool in self.tool_dict:
            patterns['DETECTION_TOOL'].append(re.escape(tool))
        
        return patterns
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取文本中的实体"""
        entities = []
        
        # 使用词典匹配
        entities.extend(self._dict_match(text))
        
        # 使用正则表达式匹配
        entities.extend(self._pattern_match(text))
        
        # 去重和排序
        entities = self._deduplicate_and_sort(entities)
        
        return entities
    
    def _dict_match(self, text: str) -> List[Dict]:
        """使用词典进行匹配"""
        entities = []
        
        # 部件单元匹配
        for component in self.component_dict:
            for match in re.finditer(re.escape(component), text):
                entities.append({
                    'name': component,
                    'type': '部件单元',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # 性能表征匹配
        for performance in self.performance_dict:
            for match in re.finditer(re.escape(performance), text):
                entities.append({
                    'name': performance,
                    'type': '性能表征',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # 故障状态匹配
        for fault in self.fault_dict:
            for match in re.finditer(re.escape(fault), text):
                entities.append({
                    'name': fault,
                    'type': '故障状态',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # 检测工具匹配
        for tool in self.tool_dict:
            for match in re.finditer(re.escape(tool), text):
                entities.append({
                    'name': tool,
                    'type': '检测工具',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        return entities
    
    def _pattern_match(self, text: str) -> List[Dict]:
        """使用正则表达式模式匹配"""
        entities = []
        
        # 匹配"运行异常"、"工作异常"等模式
        fault_patterns = [
            r'运行异常', r'工作异常', r'运转异常', r'操作异常',
            r'运行故障', r'工作故障', r'运转故障', r'操作故障',
            r'运行不良', r'工作不良', r'运转不良', r'操作不良'
        ]
        
        for pattern in fault_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'name': match.group(),
                    'type': '故障状态',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # 匹配"检测"、"测试"等动词+工具的模式
        tool_patterns = [
            r'使用\s*(\w+)\s*检测',
            r'用\s*(\w+)\s*检测',
            r'通过\s*(\w+)\s*检测',
            r'利用\s*(\w+)\s*检测'
        ]
        
        for pattern in tool_patterns:
            for match in re.finditer(pattern, text):
                tool_name = match.group(1)
                if tool_name in self.tool_dict:
                    entities.append({
                        'name': tool_name,
                        'type': '检测工具',
                        'start_pos': match.start(1),
                        'end_pos': match.end(1)
                    })
        
        return entities
    
    def _deduplicate_and_sort(self, entities: List[Dict]) -> List[Dict]:
        """去重和排序"""
        # 使用(start_pos, end_pos, type)作为唯一标识
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['start_pos'], entity['end_pos'], entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        # 按位置排序
        unique_entities.sort(key=lambda x: x['start_pos'])
        
        return unique_entities
    
    def extract_entities_batch(self, texts: List[str]) -> List[List[Dict]]:
        """批量抽取实体"""
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results


def test_rule_based_ner():
    """测试基于规则的NER"""
    extractor = RuleBasedEntityExtractor()
    
    # 测试案例
    test_cases = [
        "伺服电机运行异常，维修人员使用万用表检测电路故障。",
        "液压泵压力不足，需要更换密封圈解决泄漏问题。",
        "轴承温度过高导致振动，使用振动仪检测发现不对中。",
        "发动机启动困难，检查发现燃油泵故障。"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n=== 测试案例 {i} ===")
        print(f"输入文本: {text}")
        
        entities = extractor.extract_entities(text)
        
        print("抽取的实体:")
        for entity in entities:
            print(f"  - {entity['name']} [{entity['type']}] at position {entity['start_pos']}-{entity['end_pos']}")


if __name__ == "__main__":
    test_rule_based_ner()