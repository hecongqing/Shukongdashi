# 知识图谱项目 API 文档

## 📖 概述

本文档描述了知识图谱项目的主要API接口和使用方法。项目提供了完整的数据采集、模型训练、图谱构建和问答功能的API接口。

## 🏗️ 架构概述

```
知识图谱API架构
├── 数据采集模块 (data_collection)
├── 模型训练模块 (models)
├── 图谱构建模块 (knowledge_graph)  
├── 问答系统模块 (qa_system)
└── Web前端模块 (frontend)
```

## 📊 数据采集模块 API

### DataCollectionManager

主要的数据采集管理类，支持多种数据源。

#### 初始化
```python
from data_collection.crawler import DataCollectionManager

manager = DataCollectionManager()
```

#### 主要方法

##### collect_data()
采集指定源的数据

```python
# 方法签名
def collect_data(self, source: str, query: str, limit: int = 100) -> List[Dict[str, Any]]

# 使用示例
data = manager.collect_data(
    source="wikipedia",
    query="人工智能",
    limit=50
)
```

**参数说明:**
- `source`: 数据源类型 (`wikipedia`, `baidu_baike`, `news`)
- `query`: 搜索关键词
- `limit`: 返回结果数量限制

**返回值:**
```python
[
    {
        "title": "文档标题",
        "content": "文档内容",
        "url": "原始URL",
        "source": "数据源",
        "timestamp": "采集时间"
    }
]
```

##### batch_collect()
批量采集多个关键词的数据

```python
# 方法签名
def batch_collect(self, queries: List[str], source: str = "wikipedia") -> Dict[str, List]

# 使用示例
results = manager.batch_collect(
    queries=["机器学习", "深度学习", "自然语言处理"],
    source="wikipedia"
)
```

## 🤖 模型训练模块 API

### NER模型 (NERTrainer)

命名实体识别模型训练和预测。

#### 初始化
```python
from models.ner_model import NERTrainer

trainer = NERTrainer()
```

#### 主要方法

##### train()
训练NER模型

```python
# 方法签名
def train(self, train_data_path: str, val_data_path: str = None) -> Dict[str, Any]

# 使用示例
results = trainer.train(
    train_data_path="data/train_ner.json",
    val_data_path="data/val_ner.json"
)
```

**数据格式:**
```json
[
    {
        "text": "北京是中国的首都",
        "labels": ["B-LOC", "O", "B-GPE", "O", "O", "O"]
    }
]
```

##### predict()
实体识别预测

```python
# 方法签名
def predict(self, text: str) -> List[Tuple[str, str]]

# 使用示例
entities = trainer.predict("苹果公司位于美国加利福尼亚州")
# 返回: [("苹果公司", "ORG"), ("美国", "GPE"), ("加利福尼亚州", "LOC")]
```

### 关系抽取模型 (RelationExtractionTrainer)

实体间关系抽取模型。

#### 初始化
```python
from models.relation_extraction import RelationExtractionTrainer

re_trainer = RelationExtractionTrainer()
```

#### 主要方法

##### extract_relations()
抽取文本中的实体关系

```python
# 方法签名
def extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]

# 使用示例
relations = re_trainer.extract_relations(
    text="苹果公司的CEO是蒂姆·库克",
    entities=[("苹果公司", "ORG"), ("蒂姆·库克", "PER")]
)
# 返回: [{"head": "苹果公司", "relation": "CEO", "tail": "蒂姆·库克", "confidence": 0.95}]
```

## 🕸️ 知识图谱构建模块 API

### KnowledgeGraphBuilder

知识图谱构建和管理核心类。

#### 初始化
```python
from knowledge_graph.graph_builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()
```

#### 主要方法

##### add_triple()
添加三元组到知识图谱

```python
# 方法签名
def add_triple(self, head: str, relation: str, tail: str, properties: Dict = None) -> bool

# 使用示例
success = builder.add_triple(
    head="苹果公司",
    relation="CEO",
    tail="蒂姆·库克",
    properties={"start_date": "2011-08-24"}
)
```

##### batch_add_triples()
批量添加三元组

```python
# 方法签名
def batch_add_triples(self, triples: List[Triple]) -> Dict[str, int]

# 使用示例
from knowledge_graph.graph_builder import Triple

triples = [
    Triple("苹果公司", "CEO", "蒂姆·库克"),
    Triple("苹果公司", "总部", "库比蒂诺"),
    Triple("苹果公司", "成立时间", "1976年")
]

result = builder.batch_add_triples(triples)
# 返回: {"success": 3, "failed": 0}
```

##### query_graph()
查询知识图谱

```python
# 方法签名
def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]

# 使用示例
results = builder.query_graph(
    "MATCH (n)-[r]->(m) WHERE n.name = '苹果公司' RETURN n, r, m"
)
```

##### visualize_subgraph()
可视化子图

```python
# 方法签名
def visualize_subgraph(self, center_node: str, depth: int = 2) -> str

# 使用示例
html_content = builder.visualize_subgraph("苹果公司", depth=2)
```

## 💬 问答系统模块 API

### KnowledgeGraphQA

基于知识图谱的智能问答系统。

#### 初始化
```python
from qa_system.qa_engine import KnowledgeGraphQA

qa_system = KnowledgeGraphQA()
```

#### 主要方法

##### answer_question()
回答用户问题

```python
# 方法签名
def answer_question(self, question: str) -> Dict[str, Any]

# 使用示例
result = qa_system.answer_question("苹果公司的CEO是谁？")
```

**返回格式:**
```python
{
    "question": "苹果公司的CEO是谁？",
    "answer": "苹果公司的CEO是蒂姆·库克。",
    "confidence": 0.95,
    "source": "knowledge_graph",
    "entities": ["苹果公司", "CEO"],
    "query_time": 0.123,
    "cypher_queries": ["MATCH (n)-[r:CEO]->(m) WHERE n.name = '苹果公司' RETURN m.name"]
}
```

##### batch_answer()
批量回答问题

```python
# 方法签名
def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]

# 使用示例
questions = [
    "苹果公司的CEO是谁？",
    "苹果公司总部在哪里？",
    "苹果公司什么时候成立？"
]

results = qa_system.batch_answer(questions)
```

##### get_similar_questions()
获取相似问题

```python
# 方法签名
def get_similar_questions(self, question: str, threshold: float = 0.8) -> List[str]

# 使用示例
similar = qa_system.get_similar_questions("苹果公司CEO是谁", threshold=0.8)
```

## 🎨 Web前端模块 API

### KnowledgeGraphApp

Streamlit Web应用主类。

#### 主要页面功能

##### 数据采集页面
- 多源数据采集界面
- 实时采集进度显示
- 数据预览和导出

##### 数据标注页面
- 实体标注工具
- 关系标注界面
- 标注质量检查

##### 模型训练页面
- NER模型训练
- 关系抽取模型训练
- 训练过程监控

##### 图谱构建页面
- 知识图谱构建
- 图谱可视化
- 图谱统计信息

##### 问答系统页面
- 智能问答界面
- 问题历史记录
- 相似问题推荐

## 🔧 配置管理 API

### 环境配置

主要配置项位于 `config/settings.py`：

```python
from config.settings import CONFIG

# 数据库配置
neo4j_config = CONFIG["database"]["neo4j"]
redis_config = CONFIG["database"]["redis"]

# 模型配置
ner_config = CONFIG["models"]["ner"]
llm_config = CONFIG["models"]["llm"]
```

### 环境变量

支持的环境变量：

```bash
# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledge123

# Redis配置
REDIS_URL=redis://:redis123@localhost:6379

# 模型配置
CUDA_VISIBLE_DEVICES=0
MODEL_CACHE_DIR=/path/to/models
```

## 📝 使用示例

### 完整流程示例

```python
#!/usr/bin/env python3
"""
知识图谱完整流程示例
"""
from data_collection.crawler import DataCollectionManager
from models.ner_model import NERTrainer
from models.relation_extraction import RelationExtractionTrainer
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from qa_system.qa_engine import KnowledgeGraphQA

# 1. 数据采集
print("1. 开始数据采集...")
data_manager = DataCollectionManager()
documents = data_manager.collect_data(
    source="wikipedia",
    query="人工智能",
    limit=100
)

# 2. 实体识别
print("2. 开始实体识别...")
ner_trainer = NERTrainer()
all_entities = []
for doc in documents:
    entities = ner_trainer.predict(doc["content"])
    all_entities.extend(entities)

# 3. 关系抽取
print("3. 开始关系抽取...")
re_trainer = RelationExtractionTrainer()
relations = []
for doc in documents:
    doc_entities = ner_trainer.predict(doc["content"])
    doc_relations = re_trainer.extract_relations(doc["content"], doc_entities)
    relations.extend(doc_relations)

# 4. 构建知识图谱
print("4. 开始构建知识图谱...")
graph_builder = KnowledgeGraphBuilder()
for relation in relations:
    graph_builder.add_triple(
        head=relation["head"],
        relation=relation["relation"],
        tail=relation["tail"]
    )

# 5. 智能问答
print("5. 开始智能问答...")
qa_system = KnowledgeGraphQA()
questions = [
    "什么是人工智能？",
    "人工智能有哪些应用？",
    "机器学习和深度学习的关系是什么？"
]

for question in questions:
    result = qa_system.answer_question(question)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    print("-" * 50)
```

## ⚠️ 注意事项

### 性能优化
1. **批量操作**: 优先使用批量API减少网络开销
2. **缓存机制**: 问答系统自动缓存常见问题
3. **异步处理**: 大批量数据处理使用异步API

### 错误处理
1. **重试机制**: 网络请求自动重试
2. **异常捕获**: 完整的异常处理机制
3. **日志记录**: 详细的操作日志

### 资源管理
1. **内存管理**: 大文件流式处理
2. **GPU使用**: 自动检测和使用GPU加速
3. **连接池**: 数据库连接池管理

## 📞 技术支持

如有问题请参考：
1. **项目文档**: `README.md`
2. **部署指南**: `DEPLOYMENT.md`
3. **API示例**: `examples/` 目录
4. **单元测试**: `tests/` 目录

---

**最后更新**: 2024年12月  
**API版本**: v1.0.0