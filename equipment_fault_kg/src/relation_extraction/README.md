# 关系抽取模块

## 概述

关系抽取模块是数控机床故障诊断知识图谱系统的核心组件之一，用于从文本中提取实体间的关系。该模块支持多种关系抽取方法，包括基于规则模式、启发式方法和基于已知实体的关系抽取。

## 功能特性

- **多种抽取方法**：支持规则模式匹配、启发式方法和基于实体的关系抽取
- **关系验证**：提供关系质量验证和过滤功能
- **关系统计**：提供详细的关系统计和分析功能
- **模式管理**：支持自定义关系模式的添加和管理
- **多种关系类型**：支持故障-症状、故障-原因、维修-工具等多种关系类型

## 模块结构

```
relation_extraction/
├── __init__.py              # 模块初始化文件
├── relation_patterns.py     # 关系模式定义
├── relation_extractor.py    # 关系抽取器
├── relation_validator.py    # 关系验证器
└── README.md               # 本文档
```

## 核心组件

### 1. RelationPatterns (关系模式)

定义各种关系类型的正则表达式模式，用于从文本中提取实体间关系。

**主要功能：**
- 预定义故障诊断相关的关系模式
- 支持自定义关系模式的添加
- 提供模式验证和统计功能

**支持的关系类型：**
- `fault_symptom`: 故障-症状关系
- `fault_cause`: 故障-原因关系
- `fault_solution`: 故障-解决方法关系
- `equipment_fault`: 设备-故障关系
- `component_fault`: 部件-故障关系
- `fault_impact`: 故障-影响关系
- `maintenance_tool`: 维修-工具关系
- `maintenance_personnel`: 维修-人员关系
- `maintenance_time`: 维修-时间关系
- `detection_method`: 检测-方法关系
- `detection_equipment`: 检测-设备关系

### 2. RelationExtractor (关系抽取器)

实现从文本中提取实体间关系的核心功能。

**主要功能：**
- 基于预定义模式的关系抽取
- 基于启发式方法的关系抽取
- 基于已知实体的关系抽取
- 关系去重和排序
- 关系统计和分析

**抽取方法：**

1. **模式匹配抽取**：使用预定义的正则表达式模式匹配文本中的关系
2. **启发式抽取**：基于常见关系谓词和句子结构进行关系抽取
3. **实体引导抽取**：基于已知实体列表进行关系抽取

### 3. RelationValidator (关系验证器)

用于验证和过滤从文本中提取的关系。

**验证规则：**
- 实体长度验证：检查实体长度是否在合理范围内
- 实体质量验证：过滤低质量实体
- 关系语义验证：验证关系的语义合理性
- 重复检查：检查并移除重复关系
- 置信度阈值：过滤低置信度关系

## 使用方法

### 基本使用

```python
from relation_extraction import RelationExtractor, RelationValidator

# 创建关系抽取器
extractor = RelationExtractor()

# 从文本中提取关系
text = "主轴故障导致加工精度下降，需要更换轴承解决。"
relations = extractor.extract_relations(text)

# 打印提取的关系
for relation in relations:
    print(f"{relation.subject} --{relation.predicate}--> {relation.object}")
    print(f"类型: {relation.relation_type}, 置信度: {relation.confidence}")
```

### 基于实体的关系抽取

```python
# 基于已知实体提取关系
text = "主轴故障导致加工精度下降，需要更换轴承解决。"
entities = ["主轴", "加工精度", "轴承"]

relations = extractor.extract_relations(text, entities)
```

### 关系验证

```python
# 创建关系验证器
validator = RelationValidator()

# 验证关系
validated_relations = validator.validate_relations(relations, min_confidence=0.6)

# 获取验证统计
stats = validator.get_validation_statistics(relations, validated_relations)
print(f"过滤率: {stats['filter_rate']:.2%}")
```

### 关系统计

```python
# 获取关系统计信息
stats = extractor.get_relation_statistics(relations)

print(f"总关系数: {stats['total_relations']}")
print(f"关系类型分布: {stats['relation_type_counts']}")
print(f"置信度分布: {stats['confidence_distribution']}")
```

### 关系过滤

```python
# 按置信度过滤
high_confidence = extractor.filter_relations_by_confidence(relations, 0.7)

# 按类型过滤
fault_relations = extractor.filter_relations_by_type(relations, 'fault_symptom')
```

### 模式管理

```python
from relation_extraction import RelationPatterns

# 创建模式管理器
patterns = RelationPatterns()

# 获取所有模式
all_patterns = patterns.get_all_patterns()

# 添加自定义模式
patterns.add_custom_pattern('custom_fault', 
    r'([^，。；]*故障[^，。；]*)(引起|造成)([^，。；]*问题[^，。；]*)')

# 验证模式
is_valid = patterns.validate_pattern(r'([^，。；]*故障[^，。；]*)(引起|造成)([^，。；]*问题[^，。；]*)')
```

## API 接口

### RelationExtractor

#### 方法

- `extract_relations(text: str, entities: Optional[List[str]] = None) -> List[Relation]`
  - 从文本中提取关系
  - 参数：
    - `text`: 输入文本
    - `entities`: 可选的实体列表
  - 返回：关系列表

- `get_relation_statistics(relations: List[Relation]) -> Dict`
  - 获取关系统计信息
  - 返回：统计信息字典

- `filter_relations_by_type(relations: List[Relation], relation_type: str) -> List[Relation]`
  - 根据关系类型过滤关系

- `filter_relations_by_confidence(relations: List[Relation], min_confidence: float) -> List[Relation]`
  - 根据置信度过滤关系

### RelationValidator

#### 方法

- `validate_relations(relations: List[Relation], min_confidence: float = 0.5, enable_rules: Set[str] = None) -> List[Relation]`
  - 验证关系列表
  - 参数：
    - `relations`: 关系列表
    - `min_confidence`: 最小置信度阈值
    - `enable_rules`: 启用的验证规则集合
  - 返回：验证通过的关系列表

- `get_validation_statistics(original_relations: List[Relation], validated_relations: List[Relation]) -> Dict`
  - 获取验证统计信息

### RelationPatterns

#### 方法

- `get_all_patterns() -> Dict[str, List[str]]`
  - 获取所有关系模式

- `get_patterns_by_type(pattern_type: str) -> List[str]`
  - 根据类型获取关系模式

- `add_custom_pattern(pattern_type: str, pattern: str)`
  - 添加自定义关系模式

- `validate_pattern(pattern: str) -> bool`
  - 验证正则表达式模式是否有效

## 数据格式

### Relation 数据类

```python
@dataclass
class Relation:
    subject: str          # 主语实体
    predicate: str        # 关系谓词
    object: str          # 宾语实体
    relation_type: str    # 关系类型
    confidence: float     # 置信度
    source_text: str      # 来源文本
    start_pos: int        # 开始位置
    end_pos: int          # 结束位置
```

## 示例

### 示例1：基本关系抽取

```python
from relation_extraction import RelationExtractor

extractor = RelationExtractor()

text = """
数控机床主轴故障导致加工精度下降，需要更换轴承解决。
数控系统出现报警，原因是电源电压不稳定。
刀具磨损严重，影响加工质量，建议更换新刀具。
"""

relations = extractor.extract_relations(text)

for relation in relations:
    print(f"{relation.subject} --{relation.predicate}--> {relation.object}")
    print(f"类型: {relation.relation_type}, 置信度: {relation.confidence:.2f}")
```

### 示例2：关系验证和统计

```python
from relation_extraction import RelationExtractor, RelationValidator

extractor = RelationExtractor()
validator = RelationValidator()

# 提取关系
relations = extractor.extract_relations(text)

# 验证关系
validated_relations = validator.validate_relations(relations, min_confidence=0.6)

# 获取统计
stats = extractor.get_relation_statistics(validated_relations)
print(f"验证后关系数: {stats['total_relations']}")
```

## 测试

运行测试脚本：

```bash
cd equipment_fault_kg
python3 test_relation_extraction.py
```

运行演示脚本：

```bash
cd equipment_fault_kg
python3 demo_relation_extraction.py
```

## 配置

可以通过修改 `relation_patterns.py` 中的模式定义来自定义关系抽取规则，或者通过 `RelationPatterns` 类的 `add_custom_pattern` 方法动态添加新的关系模式。

## 注意事项

1. **文本预处理**：建议在使用关系抽取前对文本进行适当的预处理，如去除特殊字符、标准化格式等。

2. **置信度阈值**：根据具体应用场景调整置信度阈值，平衡召回率和精确率。

3. **模式优化**：根据实际数据特点优化关系模式，提高抽取效果。

4. **性能考虑**：对于大量文本，建议分批处理以提高性能。

## 扩展

该模块设计为可扩展的，可以通过以下方式进行扩展：

1. **添加新的关系类型**：在 `RelationPatterns` 中添加新的关系类型和对应模式
2. **自定义验证规则**：在 `RelationValidator` 中添加新的验证规则
3. **集成机器学习模型**：可以集成预训练的关系抽取模型来提高抽取效果
4. **多语言支持**：可以扩展支持其他语言的关系抽取

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个模块。