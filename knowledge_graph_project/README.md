# 知识图谱实战项目

## 项目概述

本项目是一个完整的知识图谱构建与应用系统，包含以下核心功能：

1. **数据采集与预处理** - 多源数据爬取、清洗和标准化
2. **信息抽取** - 基于大模型的实体抽取、关系抽取
3. **知识图谱构建** - Neo4j图数据库存储和查询
4. **本地大模型部署** - 用于信息抽取和问答推理
5. **智能问答系统** - 基于知识图谱的自然语言问答

## 技术栈

- **后端框架**: FastAPI + Python 3.8+
- **图数据库**: Neo4j 4.4+
- **大模型**: ChatGLM3-6B (本地部署)
- **向量数据库**: Milvus/Chroma
- **前端**: Vue.js 3 + Element Plus
- **数据处理**: Pandas, NumPy, SpaCy
- **深度学习**: PyTorch, Transformers
- **部署**: Docker + Docker Compose

## 项目结构

```
knowledge_graph_project/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后数据
│   └── annotations/               # 标注数据
├── src/                           # 源代码
│   ├── data_collection/           # 数据采集模块
│   ├── information_extraction/    # 信息抽取模块
│   ├── knowledge_graph/           # 知识图谱模块
│   ├── llm/                       # 大模型模块
│   ├── qa_system/                 # 问答系统
│   └── api/                       # API接口
├── models/                        # 训练好的模型
├── config/                        # 配置文件
├── tests/                         # 测试代码
├── docker/                        # Docker配置
├── frontend/                      # 前端代码
└── docs/                          # 文档
```

## 核心功能模块

### 1. 数据采集模块
- 网页爬虫 (Scrapy)
- PDF文档解析
- 结构化数据导入
- 数据清洗和预处理

### 2. 信息抽取模块
- 实体识别 (NER)
- 关系抽取 (RE)
- 事件抽取
- 基于大模型的抽取

### 3. 知识图谱模块
- 图数据库设计
- 知识融合
- 质量评估
- 可视化展示

### 4. 大模型模块
- ChatGLM3-6B本地部署
- 模型微调
- 推理优化
- 多模态支持

### 5. 问答系统
- 问题理解
- 知识检索
- 答案生成
- 多轮对话

## 快速开始

### 环境要求
- Python 3.8+
- Docker & Docker Compose
- 16GB+ RAM (用于大模型)
- GPU支持 (推荐)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd knowledge_graph_project
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动服务**
```bash
docker-compose up -d
```

4. **初始化数据库**
```bash
python scripts/init_database.py
```

5. **启动大模型**
```bash
python scripts/start_llm.py
```

## 使用指南

### 数据采集
```python
from src.data_collection.crawler import WebCrawler

crawler = WebCrawler()
crawler.crawl_websites(['url1', 'url2'])
```

### 信息抽取
```python
from src.information_extraction.extractor import InformationExtractor

extractor = InformationExtractor()
entities = extractor.extract_entities(text)
relations = extractor.extract_relations(text)
```

### 知识图谱查询
```python
from src.knowledge_graph.graph_manager import GraphManager

graph = GraphManager()
results = graph.query("MATCH (n:Entity) RETURN n LIMIT 10")
```

### 问答系统
```python
from src.qa_system.qa_engine import QAEngine

qa = QAEngine()
answer = qa.answer("什么是知识图谱？")
```

## API文档

启动服务后访问: http://localhost:8000/docs

### 主要接口

- `POST /api/extract` - 信息抽取
- `GET /api/query` - 图谱查询
- `POST /api/qa` - 智能问答
- `GET /api/entities` - 实体列表
- `POST /api/relations` - 关系查询

## 模型训练

### 实体识别模型
```bash
python scripts/train_ner.py --data_path data/annotations/ner --model_name bert-base-chinese
```

### 关系抽取模型
```bash
python scripts/train_re.py --data_path data/annotations/re --model_name bert-base-chinese
```

### 大模型微调
```bash
python scripts/finetune_llm.py --base_model THUDM/chatglm3-6b --data_path data/finetune
```

## 部署指南

### 开发环境
```bash
docker-compose -f docker-compose.dev.yml up
```

### 生产环境
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 性能优化

- 模型量化 (INT8/INT4)
- 图数据库索引优化
- 缓存策略
- 负载均衡

## 监控与日志

- Prometheus + Grafana 监控
- ELK 日志分析
- 性能指标追踪

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [GitHub Repository URL]