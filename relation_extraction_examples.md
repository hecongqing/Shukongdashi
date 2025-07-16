# 关系抽取使用示例

本文档展示如何使用关系抽取器（RelationExtractor）和联合抽取器（JointExtractor）进行实体关系抽取。

## 1. 关系抽取器（RelationExtractor）

### 1.1 基本初始化

```python
from relation_extraction.deploy_relation import RelationExtractor

# 初始化关系抽取器，需要提供训练好的模型路径
model_path = "path/to/relation_model.pth"
relation_extractor = RelationExtractor(model_path)
```

### 1.2 单文本关系抽取

```python
# 准备文本和实体数据
text = "数控机床主轴故障导致加工精度下降，需要更换轴承解决。"

# 实体列表（需要预先准备，可以通过实体抽取器获得）
entities = [
    {'name': '主轴', 'type': 'COMPONENT'},
    {'name': '加工精度', 'type': 'PERFORMANCE'},
    {'name': '轴承', 'type': 'COMPONENT'}
]

# 抽取关系
relations = relation_extractor.extract_relations(text, entities)

# 输出结果
for relation in relations:
    print(f"头实体: {relation['head_entity']}")
    print(f"尾实体: {relation['tail_entity']}")
    print(f"关系类型: {relation['relation_type']}")
    print(f"置信度: {relation['confidence']:.2f}")
    print("-" * 40)
```

### 1.3 批量关系抽取

```python
# 准备多个文本和对应的实体列表
texts = [
    "主轴故障导致加工精度下降，需要更换轴承解决。",
    "数控系统出现报警，原因是电源电压不稳定。",
    "刀具磨损严重，影响加工质量，建议更换新刀具。"
]

entities_list = [
    [
        {'name': '主轴', 'type': 'COMPONENT'},
        {'name': '加工精度', 'type': 'PERFORMANCE'},
        {'name': '轴承', 'type': 'COMPONENT'}
    ],
    [
        {'name': '数控系统', 'type': 'COMPONENT'},
        {'name': '电源电压', 'type': 'COMPONENT'}
    ],
    [
        {'name': '刀具', 'type': 'COMPONENT'},
        {'name': '加工质量', 'type': 'PERFORMANCE'}
    ]
]

# 批量抽取关系
results = relation_extractor.extract_relations_batch(texts, entities_list)

# 输出结果
for i, relations in enumerate(results):
    print(f"文本 {i+1} 的关系:")
    for relation in relations:
        print(f"  {relation['head_entity']} -> {relation['tail_entity']} ({relation['relation_type']})")
    print()
```

### 1.4 按关系类型过滤

```python
# 获取特定类型的关系
fault_relations = relation_extractor.get_relations_by_type(
    text, entities, relation_type="部件故障"
)

print("部件故障关系:")
for relation in fault_relations:
    print(f"  {relation['head_entity']} -> {relation['tail_entity']}")
```

## 2. 联合抽取器（JointExtractor）

### 2.1 基本初始化

```python
from relation_extraction.deploy_relation import JointExtractor

# 初始化联合抽取器，需要提供NER和关系抽取模型路径
ner_model_path = "path/to/ner_model.pth"
relation_model_path = "path/to/relation_model.pth"

joint_extractor = JointExtractor(ner_model_path, relation_model_path)
```

### 2.2 SPO三元组抽取

```python
# 输入文本
text = "数控机床主轴故障导致加工精度下降，维修人员使用万用表检测电路。"

# 一步完成实体抽取和关系抽取
result = joint_extractor.extract_spo(text)

# 输出结果
print(f"原文: {result['text']}")
print()

print("抽取的实体:")
for entity in result['entities']:
    print(f"  {entity['name']} ({entity['type']})")
print()

print("抽取的关系:")
for relation in result['relations']:
    print(f"  {relation['head_entity']} --{relation['relation_type']}--> {relation['tail_entity']}")
    print(f"    置信度: {relation['confidence']:.2f}")
print()

print("SPO三元组:")
for spo in result['spo_list']:
    print(f"  主语: {spo['h']['name']}")
    print(f"  谓语: {spo['relation']}")
    print(f"  宾语: {spo['t']['name']}")
    print()
```

### 2.3 批量SPO抽取

```python
# 多个文本
texts = [
    "主轴故障导致加工精度下降。",
    "伺服电机出现异常，需要专业维修。",
    "操作员使用卡尺测量工件尺寸。"
]

# 批量抽取SPO
results = joint_extractor.extract_spo_batch(texts)

# 输出结果
for i, result in enumerate(results):
    print(f"文本 {i+1}: {result['text']}")
    print(f"实体数: {len(result['entities'])}")
    print(f"关系数: {len(result['relations'])}")
    print(f"SPO数: {len(result['spo_list'])}")
    
    for spo in result['spo_list']:
        print(f"  ({spo['h']['name']}, {spo['relation']}, {spo['t']['name']})")
    print("-" * 50)
```

## 3. 完整示例

```python
#!/usr/bin/env python3
"""
关系抽取完整示例
"""

def main():
    # 模型路径（需要根据实际路径修改）
    ner_model_path = "equipment_fault_kg/models/ner_model.pth"
    relation_model_path = "equipment_fault_kg/models/relation_model.pth"
    
    # 初始化联合抽取器
    print("初始化联合抽取器...")
    joint_extractor = JointExtractor(ner_model_path, relation_model_path)
    
    # 测试文本
    test_texts = [
        "数控机床主轴故障导致加工精度下降，需要更换轴承解决问题。",
        "伺服电机运行异常，维修人员使用万用表检测电路故障。",
        "刀具磨损严重影响工件表面质量，建议及时更换新刀具。",
        "数控系统出现报警，检查发现是电源电压不稳定造成的。"
    ]
    
    print("开始抽取...")
    
    # 逐个处理文本
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"处理文本 {i}: {text}")
        print(f"{'='*60}")
        
        try:
            # 抽取SPO三元组
            result = joint_extractor.extract_spo(text)
            
            # 显示实体
            print(f"\n实体 ({len(result['entities'])}个):")
            for entity in result['entities']:
                print(f"  - {entity['name']} [{entity['type']}]")
            
            # 显示关系
            print(f"\n关系 ({len(result['relations'])}个):")
            for relation in result['relations']:
                print(f"  - {relation['head_entity']} --{relation['relation_type']}--> {relation['tail_entity']}")
                print(f"    置信度: {relation['confidence']:.3f}")
            
            # 显示SPO三元组
            print(f"\nSPO三元组 ({len(result['spo_list'])}个):")
            for spo in result['spo_list']:
                print(f"  - ({spo['h']['name']}, {spo['relation']}, {spo['t']['name']})")
        
        except Exception as e:
            print(f"处理失败: {e}")
    
    print(f"\n{'='*60}")
    print("抽取完成!")

if __name__ == "__main__":
    main()
```

## 4. 高级用法

### 4.1 自定义置信度阈值

```python
# 在RelationPredictor中设置更高的置信度阈值
# 修改 predict_all_relations 方法中的阈值
relations = []
for head_entity, tail_entity in entity_pairs:
    relation = relation_extractor.predictor.predict_relation(
        text, head_entity['name'], tail_entity['name']
    )
    
    # 自定义阈值
    if relation['has_relation'] and relation['binary_confidence'] > 0.7:  # 提高阈值
        relations.append(relation)
```

### 4.2 结果后处理

```python
def filter_relations_by_confidence(relations, min_confidence=0.6):
    """根据置信度过滤关系"""
    return [rel for rel in relations if rel['confidence'] >= min_confidence]

def group_relations_by_type(relations):
    """按关系类型分组"""
    groups = {}
    for relation in relations:
        rel_type = relation['relation_type']
        if rel_type not in groups:
            groups[rel_type] = []
        groups[rel_type].append(relation)
    return groups

# 使用示例
relations = joint_extractor.extract_spo(text)['relations']
high_conf_relations = filter_relations_by_confidence(relations, 0.8)
grouped_relations = group_relations_by_type(high_conf_relations)

for rel_type, rels in grouped_relations.items():
    print(f"{rel_type}: {len(rels)}个关系")
```

## 5. 注意事项

1. **模型路径**: 确保模型文件路径正确，模型文件应该是训练好的`.pth`文件
2. **实体格式**: 实体列表中每个实体应该包含`name`和`type`字段
3. **文本长度**: 输入文本不宜过长，建议单句或短段落
4. **置信度阈值**: 可以根据实际需求调整置信度阈值来控制结果质量
5. **GPU支持**: 如果有GPU，模型会自动使用GPU加速推理

## 6. 支持的关系类型

根据模型训练，支持以下关系类型：
- **部件故障**: 表示部件与故障之间的关系
- **性能故障**: 表示性能指标与故障的关系  
- **检测工具**: 表示检测工具与被检测对象的关系
- **组成**: 表示部件之间的组成关系