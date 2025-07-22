# 简洁版知识图谱问答框架 (Easy KGQA Framework)

## 概述

这是一个**简洁版**的知识图谱问答框架，专门为教学目的设计。保留了Neo4j图数据库和实体识别等核心技术，但简化了其他流程。特点：

- 🚀 **核心技术**: 使用Neo4j图数据库 + 实体识别
- 📚 **易于理解**: 清晰的代码结构和注释  
- 🔧 **依赖最少**: 仅需neo4j-driver包
- 💡 **教学友好**: 适合学习知识图谱和问答系统的核心概念

## 框架结构

```
easy_kgqa_framework/
├── __init__.py                    # 框架入口
├── core/                          # 核心模块
│   ├── __init__.py
│   └── kgqa_engine.py            # KGQA引擎 (Neo4j)
└── utils/                         # 工具模块
    ├── __init__.py
    ├── text_utils.py              # 文本处理工具
    └── entity_recognizer.py       # 实体识别器
```

## 核心功能

### 1. 知识管理 (Neo4j)
- ✅ 实体管理 (Entity Management)
- ✅ 关系管理 (Relation Management)  
- ✅ 故障案例管理 (Fault Case Management)
- ✅ 图数据库操作 (Graph Database Operations)

### 2. 实体识别 (NER)
- ✅ 设备实体识别 (Equipment Entity Recognition)
- ✅ 症状实体识别 (Symptom Entity Recognition)
- ✅ 动作实体识别 (Action Entity Recognition)
- ✅ 实体分类 (Entity Classification)

### 3. 问答功能
- ✅ 智能问题类型判断 (Question Type Detection)
- ✅ 实体查询 (Entity Query)
- ✅ 关系查找 (Relation Finding)
- ✅ 故障案例匹配 (Fault Case Matching)

### 4. 工具函数
- ✅ 文本清理 (Text Cleaning)
- ✅ 关键词提取 (Keyword Extraction)
- ✅ 相似度计算 (Similarity Calculation)
- ✅ 统计信息 (Statistics)

## 快速开始

### 1. 环境准备

首先安装Neo4j并启动服务：

```bash
# 安装依赖
pip install neo4j

# 确保Neo4j服务运行在 bolt://localhost:7687
# 默认用户名密码: neo4j/password
```

### 2. 基本使用

```python
from easy_kgqa_framework import EasyKGQA, SimpleEntityRecognizer

# 初始化系统 (连接Neo4j)
kgqa = EasyKGQA()

# 添加实体
kgqa.add_entity("数控机床", "设备", "用于精密加工的自动化机床")
kgqa.add_entity("主轴", "部件", "机床的核心旋转部件")

# 添加关系
kgqa.add_relation("数控机床", "包含", "主轴")

# 添加故障案例
kgqa.add_fault_case("数控机床", "主轴不转", "电机故障", "检查电机状态")

# 实体识别
recognizer = SimpleEntityRecognizer()
entities = recognizer.recognize("数控机床主轴不转了")
print(f"识别实体: {entities}")

# 问答
result = kgqa.answer_question("主轴不转怎么办？")
print(f"识别实体: {result['entities_found']}")
print(f"回答: {result['answer']}")
```

### 3. 运行演示

```bash
python easy_demo.py
```

## 设计理念

### 简化的架构

与复杂的原框架相比，简洁版做了以下简化：

| 原框架 | 简洁版 | 说明 |
|--------|--------|------|
| Neo4j + 复杂查询 | Neo4j + 简化查询 | 保留核心技术，简化查询逻辑 |
| 复杂NLP流水线 | 字典匹配实体识别 | 保留实体识别，但简化实现 |
| 多种相似度算法 | 基础文本匹配 | 减少算法复杂度 |
| 多层抽象架构 | 单引擎设计 | 保持核心功能，简化结构 |
| 复杂配置系统 | 直接参数传递 | 减少配置复杂度 |

### 核心类设计

```python
class EasyKGQA:
    """简洁版知识图谱问答引擎 - 使用Neo4j"""
    
    # 数据管理 (Neo4j)
    def add_entity(name, type, description)     # 添加实体到图数据库
    def add_relation(subject, predicate, obj)   # 添加关系到图数据库
    def add_fault_case(...)                     # 添加故障案例到图数据库
    
    # 查询功能 (Cypher查询)
    def query_entity(name)                      # 查询实体
    def find_relations(entity)                  # 查找关系
    def search_fault_cases(query)               # 搜索故障案例
    
    # 问答接口 (实体识别 + 图查询)
    def answer_question(question)               # 主要问答接口

class SimpleEntityRecognizer:
    """简洁版实体识别器"""
    
    def recognize(text)                         # 识别实体
    def recognize_by_type(text)                 # 按类型识别实体
```

## 教学要点

### 1. 知识图谱基础概念
- **实体(Entity)**: 现实世界中的对象，如"数控机床"
- **关系(Relation)**: 实体间的联系，如"包含"、"驱动"
- **三元组(Triple)**: (主体, 谓语, 客体) 的知识表示

### 2. 问答系统流程
1. **问题理解**: 提取关键词
2. **知识检索**: 查找相关实体和关系
3. **答案生成**: 基于检索结果生成回答
4. **置信度评估**: 评估答案的可信度

### 3. 可扩展方向
- 更复杂的NLP处理
- 图神经网络集成
- 多跳推理
- 知识图谱补全

## 示例场景

框架主要针对**设备故障诊断**场景，但可以轻松扩展到其他领域：

- 🔧 故障诊断: "主轴不转怎么办？"
- 📖 知识查询: "什么是数控机床？"
- 🔍 关系查找: "数控机床包含什么部件？"

## 与原框架对比

| 特性 | 原框架 | 简洁版 |
|------|--------|--------|
| 代码行数 | ~3000行 | ~400行 |
| 文件数量 | 15+ | 6 |
| 外部依赖 | 10+ | 1 (neo4j) |
| 数据库 | Neo4j | Neo4j |
| 实体识别 | 复杂NLP | 字典匹配 |
| 学习曲线 | 陡峭 | 平缓 |
| 功能完整性 | 完整 | 核心功能 |
| 教学适用性 | 困难 | 优秀 |

## 扩展建议

学习者可以基于这个简洁版框架进行以下扩展练习：

1. **文本处理增强**: 集成jieba分词
2. **相似度算法**: 实现余弦相似度、编辑距离
3. **图数据库**: 迁移到Neo4j
4. **机器学习**: 添加分类模型
5. **Web界面**: 开发简单的Web UI

## 总结

简洁版KGQA框架保留了知识图谱问答系统的核心概念和基本功能，去除了复杂的技术细节，是学习和理解KGQA系统的最佳起点。

通过这个框架，学习者可以：
- 理解知识图谱的基本概念
- 掌握问答系统的基本流程
- 学习代码组织和模块化设计
- 为后续深入学习打下基础