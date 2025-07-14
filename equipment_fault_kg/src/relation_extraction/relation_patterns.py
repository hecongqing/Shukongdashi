"""
关系模式定义

定义各种关系类型的正则表达式模式，用于从文本中提取实体间关系
"""

from typing import Dict, List
import re

class RelationPatterns:
    """关系模式类"""
    
    def __init__(self):
        # 故障诊断相关的关系模式
        self.fault_patterns = {
            # 故障-症状关系
            'fault_symptom': [
                r'([^，。；]*故障[^，。；]*)(导致|引起|造成|产生)([^，。；]*症状[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(出现|显示|表现)([^，。；]*症状[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(伴随|伴有)([^，。；]*症状[^，。；]*)',
            ],
            
            # 故障-原因关系
            'fault_cause': [
                r'([^，。；]*故障[^，。；]*)(由|由于|因为|源于)([^，。；]*原因[^，。；]*)',
                r'([^，。；]*原因[^，。；]*)(导致|引起|造成)([^，。；]*故障[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(由于|因为)([^，。；]*)',
            ],
            
            # 故障-解决方法关系
            'fault_solution': [
                r'([^，。；]*故障[^，。；]*)(解决方法|解决方案|处理办法)([^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(修复|修理|维修)([^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(排除|解决)([^，。；]*)',
            ],
            
            # 设备-故障关系
            'equipment_fault': [
                r'([^，。；]*设备[^，。；]*)(出现|发生|产生)([^，。；]*故障[^，。；]*)',
                r'([^，。；]*设备[^，。；]*)(故障|问题)([^，。；]*)',
                r'([^，。；]*设备[^，。；]*)(异常|不正常)([^，。；]*)',
            ],
            
            # 部件-故障关系
            'component_fault': [
                r'([^，。；]*部件[^，。；]*)(损坏|故障|问题)([^，。；]*)',
                r'([^，。；]*部件[^，。；]*)(出现|发生)([^，。；]*故障[^，。；]*)',
                r'([^，。；]*部件[^，。；]*)(失效|失灵)([^，。；]*)',
            ],
            
            # 故障-影响关系
            'fault_impact': [
                r'([^，。；]*故障[^，。；]*)(影响|影响)([^，。；]*功能[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(导致|造成)([^，。；]*停机[^，。；]*)',
                r'([^，。；]*故障[^，。；]*)(影响|影响)([^，。；]*精度[^，。；]*)',
            ]
        }
        
        # 维修相关的关系模式
        self.maintenance_patterns = {
            # 维修-工具关系
            'maintenance_tool': [
                r'([^，。；]*维修[^，。；]*)(使用|需要)([^，。；]*工具[^，。；]*)',
                r'([^，。；]*维修[^，。；]*)(工具|设备)([^，。；]*)',
            ],
            
            # 维修-人员关系
            'maintenance_personnel': [
                r'([^，。；]*维修[^，。；]*)(由|需要)([^，。；]*人员[^，。；]*)',
                r'([^，。；]*维修[^，。；]*)(技术|专业)([^，。；]*人员[^，。；]*)',
            ],
            
            # 维修-时间关系
            'maintenance_time': [
                r'([^，。；]*维修[^，。；]*)(时间|耗时)([^，。；]*)',
                r'([^，。；]*维修[^，。；]*)(需要|大约)([^，。；]*时间[^，。；]*)',
            ]
        }
        
        # 检测相关的关系模式
        self.detection_patterns = {
            # 检测-方法关系
            'detection_method': [
                r'([^，。；]*检测[^，。；]*)(方法|方式)([^，。；]*)',
                r'([^，。；]*检测[^，。；]*)(使用|采用)([^，。；]*方法[^，。；]*)',
            ],
            
            # 检测-设备关系
            'detection_equipment': [
                r'([^，。；]*检测[^，。；]*)(设备|仪器)([^，。；]*)',
                r'([^，。；]*检测[^，。；]*)(使用|需要)([^，。；]*设备[^，。；]*)',
            ]
        }
        
        # 常见的关系谓词
        self.common_predicates = {
            '导致', '引起', '造成', '产生', '出现', '显示', '表现',
            '由', '由于', '因为', '源于', '解决方法', '解决方案', '处理办法',
            '修复', '修理', '维修', '损坏', '故障', '问题', '影响',
            '使用', '需要', '采用', '伴随', '伴有', '排除', '解决',
            '异常', '不正常', '失效', '失灵', '停机', '精度'
        }
        
        # 关系类型映射
        self.relation_type_mapping = {
            'fault_symptom': '故障-症状',
            'fault_cause': '故障-原因', 
            'fault_solution': '故障-解决方法',
            'equipment_fault': '设备-故障',
            'component_fault': '部件-故障',
            'fault_impact': '故障-影响',
            'maintenance_tool': '维修-工具',
            'maintenance_personnel': '维修-人员',
            'maintenance_time': '维修-时间',
            'detection_method': '检测-方法',
            'detection_equipment': '检测-设备'
        }
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """获取所有关系模式"""
        all_patterns = {}
        all_patterns.update(self.fault_patterns)
        all_patterns.update(self.maintenance_patterns)
        all_patterns.update(self.detection_patterns)
        return all_patterns
    
    def get_patterns_by_type(self, pattern_type: str) -> List[str]:
        """根据类型获取关系模式"""
        if pattern_type in self.fault_patterns:
            return self.fault_patterns[pattern_type]
        elif pattern_type in self.maintenance_patterns:
            return self.maintenance_patterns[pattern_type]
        elif pattern_type in self.detection_patterns:
            return self.detection_patterns[pattern_type]
        else:
            return []
    
    def get_relation_type_name(self, relation_type: str) -> str:
        """获取关系类型的中文名称"""
        return self.relation_type_mapping.get(relation_type, relation_type)
    
    def add_custom_pattern(self, pattern_type: str, pattern: str):
        """添加自定义关系模式"""
        if pattern_type not in self.fault_patterns:
            self.fault_patterns[pattern_type] = []
        self.fault_patterns[pattern_type].append(pattern)
    
    def validate_pattern(self, pattern: str) -> bool:
        """验证正则表达式模式是否有效"""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False
    
    def get_pattern_statistics(self) -> Dict:
        """获取模式统计信息"""
        stats = {
            'total_pattern_types': 0,
            'total_patterns': 0,
            'pattern_types': {}
        }
        
        # 统计故障模式
        for pattern_type, patterns in self.fault_patterns.items():
            stats['pattern_types'][pattern_type] = len(patterns)
            stats['total_patterns'] += len(patterns)
        
        # 统计维修模式
        for pattern_type, patterns in self.maintenance_patterns.items():
            stats['pattern_types'][pattern_type] = len(patterns)
            stats['total_patterns'] += len(patterns)
        
        # 统计检测模式
        for pattern_type, patterns in self.detection_patterns.items():
            stats['pattern_types'][pattern_type] = len(patterns)
            stats['total_patterns'] += len(patterns)
        
        stats['total_pattern_types'] = len(stats['pattern_types'])
        
        return stats