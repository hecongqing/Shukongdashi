# EASY KGQA Framework - 简化版知识图谱问答框架

## 项目简介

这是一个专为教学设计的简化版知识图谱问答框架。我们保留了KGQA的核心功能，去除了复杂的相似度匹配、解决方案推荐等组件，让学习者能够专注于理解KGQA的核心原理。

## 设计理念

**保留核心，简化复杂**
- ✅ 保留：文本处理、实体识别、知识图谱推理
- ❌ 移除：相似度匹配器（similarity_matcher.py）
- ❌ 移除：解决方案推荐器（solution_recommender.py）
- ❌ 移除：复杂的配置管理和日志系统

## 系统架构

```
用户问题
    ↓
文本清理 (SimpleTextProcessor)
    ↓
实体识别 (EntityService + 外部服务 http://127.0.0.1:50003/extract_entities)
    ↓
知识图谱查询 (KnowledgeGraphEngine + Neo4j bolt://localhost:50002)
    ↓
结果整合 (EasyAnalyzer)
    ↓
返回答案
```

## 项目结构

```
easy_kgqa_framework/
├── __init__.py                 # 框架入口
├── config.py                   # 简化配置
├── models/
│   ├── __init__.py
│   └── entities.py             # 数据模型定义
├── core/
│   ├── __init__.py
│   ├── kg_engine.py            # 知识图谱引擎（核心）
│   └── easy_analyzer.py        # 主分析器
└── utils/
    ├── __init__.py
    ├── text_processor.py       # 基础文本处理
    └── entity_service.py       # 外部实体识别服务接口

easy_kgqa_demo.py               # 演示程序
requirements_easy_kgqa.txt      # 依赖包
README_EASY_KGQA.md            # 本文档
```

## 功能特点

### 1. 简化的文本处理 (SimpleTextProcessor)
- 基础中文分词（jieba）
- 关键词提取
- 故障元素识别
- 报警代码提取

### 2. 外部实体识别服务集成 (EntityService)
- 调用您已有的实体识别服务
- 自动回退到内部基础处理
- 实体类型映射

### 3. 知识图谱查询引擎 (KnowledgeGraphEngine)
- Neo4j数据库连接
- 基础图谱查询
- 关系推理
- 路径查找

### 4. 统一分析器 (EasyAnalyzer)
- 整合所有组件
- 简单易用的API
- 置信度计算
- 上下文管理

## 安装和使用

### 1. 安装依赖
```bash
pip install -r requirements_easy_kgqa.txt
```

### 2. 配置服务
确保以下服务正在运行：
- Neo4j: `bolt://localhost:50002`
- 实体识别服务: `http://127.0.0.1:50003/extract_entities`

### 3. 运行演示
```bash
python easy_kgqa_demo.py
```

## 使用示例

### 基础使用
```python
from easy_kgqa_framework import EasyAnalyzer

# 初始化分析器
with EasyAnalyzer() as analyzer:
    # 分析问题
    result = analyzer.analyze_question("主轴不转是什么原因")
    
    # 查看结果
    print(f"置信度: {result.confidence}")
    print("提取的元素:")
    for element in result.elements:
        print(f"  - {element.content} ({element.element_type.value})")
    
    print("推理路径:")
    for path in result.reasoning_path:
        print(f"  - {path}")
```

### 简单问答
```python
with EasyAnalyzer() as analyzer:
    # 简单问答
    results = analyzer.simple_qa("什么是主轴")
    for result in results:
        print(result['name'], result.get('content', ''))
```

### 系统状态检查
```python
with EasyAnalyzer() as analyzer:
    status = analyzer.get_system_status()
    print("Neo4j状态:", status['neo4j']['status'])
    print("实体识别服务状态:", status['entity_service']['status'])
```

## API接口

### EasyAnalyzer 主要方法

1. **analyze_question(question, use_entity_service=True)**
   - 分析用户问题
   - 返回 `AnalysisResult` 对象

2. **simple_qa(question)**
   - 简单问答查询
   - 返回匹配的知识图谱节点列表

3. **get_system_status()**
   - 获取系统状态
   - 返回各组件连接状态和统计信息

### 数据模型

```python
@dataclass
class FaultElement:
    content: str                # 元素内容
    element_type: FaultType     # 元素类型
    confidence: float           # 置信度

@dataclass  
class AnalysisResult:
    question: str               # 原始问题
    elements: List[FaultElement] # 提取的元素
    kg_results: List[Dict]      # 知识图谱查询结果
    reasoning_path: List[str]   # 推理路径
    confidence: float           # 整体置信度
```

## 教学优势

### 1. 结构清晰
- 模块化设计，每个组件职责明确
- 代码简洁，易于理解和修改
- 去除了复杂的相似度计算逻辑

### 2. 易于扩展
- 可以轻松添加新的文本处理功能
- 支持自定义实体类型
- 灵活的知识图谱查询接口

### 3. 实用性强
- 直接使用您现有的服务
- 保留了KGQA的核心流程
- 提供完整的演示程序

## 与原框架的对比

| 功能 | 原框架 | EASY框架 | 说明 |
|------|--------|----------|------|
| 文本处理 | ✅ 复杂 | ✅ 简化 | 保留基础功能 |
| 实体识别 | ✅ 内置+外部 | ✅ 主要外部 | 使用您的现有服务 |
| 知识图谱查询 | ✅ 复杂推理 | ✅ 基础查询 | 核心功能保留 |
| 相似度匹配 | ✅ TF-IDF + 余弦 | ❌ 删除 | 简化复杂度 |
| 解决方案推荐 | ✅ 多源整合 | ❌ 删除 | 专注KGQA核心 |
| 配置管理 | ✅ 多环境配置 | ✅ 简化配置 | 只保留必要配置 |
| 日志系统 | ✅ 完整日志 | ❌ 简化 | 使用基础print |

## 常见问题

### Q: 如何添加新的故障类型？
A: 在 `text_processor.py` 的 `fault_keywords` 字典中添加新的关键词映射。

### Q: 如何自定义实体识别逻辑？
A: 修改 `entity_service.py` 中的 `entity_type_mapping` 字典。

### Q: 如何扩展知识图谱查询？
A: 在 `kg_engine.py` 中添加新的查询方法。

### Q: 为什么要删除相似度匹配？
A: 相似度匹配需要大量的案例数据和复杂的向量计算，对于教学来说过于复杂。KGQA的核心是基于知识图谱的推理，而不是案例匹配。

## 许可证

本项目采用开源许可证，可自由用于教学和研究目的。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**EASY KGQA Framework - 让KGQA学习更简单！**