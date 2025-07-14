# 装备制造故障知识图谱构建及其应用案例剖析

## 项目概述

本项目是一个完整的装备制造故障知识图谱构建学习项目，涵盖了从数据采集、标注、分析、建模到最终部署的全流程。项目重点学习实体抽取、关系抽取、本地大模型部署、Neo4j图谱构建与问答等核心技术。

## 项目特色

- **全流程覆盖**：从数据采集到最终部署的完整流程
- **多种技术栈**：NLP、深度学习、图数据库、大模型等
- **实用性强**：基于真实装备制造故障数据
- **教学友好**：详细的步骤说明和代码注释

## 技术栈

- **后端框架**：Django + FastAPI
- **数据库**：Neo4j图数据库 + MySQL关系数据库
- **NLP处理**：jieba分词、BERT、RoBERTa
- **深度学习**：CNN、LSTM、Transformer
- **大模型**：本地部署ChatGLM、Qwen等
- **前端**：Vue.js + Element UI
- **部署**：Docker + Nginx

## 项目结构

```
equipment_knowledge_graph_project/
├── 01_data_collection/          # 数据采集模块
│   ├── web_scraping/            # 网络爬虫
│   ├── pdf_processing/          # PDF文档处理
│   └── data_cleaning/           # 数据清洗
├── 02_data_annotation/          # 数据标注模块
│   ├── entity_annotation/       # 实体标注
│   ├── relation_annotation/     # 关系标注
│   └── annotation_tools/        # 标注工具
├── 03_information_extraction/   # 信息抽取模块
│   ├── entity_extraction/       # 实体抽取
│   ├── relation_extraction/     # 关系抽取
│   └── llm_extraction/          # 大模型抽取
├── 04_knowledge_graph/          # 知识图谱构建
│   ├── neo4j_construction/      # Neo4j图谱构建
│   ├── graph_analysis/          # 图谱分析
│   └── visualization/           # 图谱可视化
├── 05_qa_system/               # 问答系统
│   ├── question_analysis/       # 问题分析
│   ├── answer_generation/       # 答案生成
│   └── evaluation/              # 系统评估
├── 06_deployment/              # 部署模块
│   ├── docker/                 # Docker配置
│   ├── api_server/             # API服务
│   └── web_interface/          # Web界面
├── 07_llm_deployment/          # 本地大模型部署
│   ├── model_download/         # 模型下载
│   ├── model_serving/          # 模型服务
│   └── api_integration/        # API集成
├── data/                       # 数据目录
├── models/                     # 模型目录
├── docs/                       # 文档目录
└── requirements.txt            # 依赖包
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd equipment_knowledge_graph_project

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据库配置

```bash
# 启动Neo4j
docker run -p 7474:7474 -p 7687:7687 neo4j:latest

# 启动MySQL
docker run -p 3306:3306 mysql:8.0
```

### 3. 数据采集

```bash
cd 01_data_collection
python web_scraping/main.py
python pdf_processing/process_pdf.py
```

### 4. 数据标注

```bash
cd 02_data_annotation
python annotation_tools/start_annotation.py
```

### 5. 信息抽取

```bash
cd 03_information_extraction
python entity_extraction/train_entity_model.py
python relation_extraction/train_relation_model.py
```

### 6. 知识图谱构建

```bash
cd 04_knowledge_graph
python neo4j_construction/build_graph.py
```

### 7. 启动问答系统

```bash
cd 05_qa_system
python app.py
```

## 学习路径

### 第一阶段：数据采集与预处理
1. 学习网络爬虫技术
2. 掌握PDF文档处理
3. 理解数据清洗流程

### 第二阶段：数据标注与信息抽取
1. 学习实体标注方法
2. 掌握关系抽取技术
3. 了解大模型在信息抽取中的应用

### 第三阶段：知识图谱构建
1. 学习Neo4j图数据库
2. 掌握图谱构建流程
3. 理解图谱分析技术

### 第四阶段：问答系统开发
1. 学习问题理解技术
2. 掌握答案生成方法
3. 了解系统评估指标

### 第五阶段：大模型部署与应用
1. 学习本地大模型部署
2. 掌握模型优化技术
3. 理解大模型在知识图谱中的应用

## 核心功能

### 1. 数据采集模块
- 支持多种数据源（网页、PDF、数据库）
- 自动数据清洗和预处理
- 数据质量评估

### 2. 信息抽取模块
- 基于规则和深度学习的实体抽取
- 多种关系抽取方法
- 大模型辅助抽取

### 3. 知识图谱模块
- 自动图谱构建
- 图谱质量评估
- 可视化展示

### 4. 问答系统模块
- 自然语言问题理解
- 多路径答案生成
- 答案可信度评估

### 5. 大模型集成模块
- 本地大模型部署
- 模型性能优化
- API接口封装

## 技术亮点

1. **多模态数据处理**：支持文本、图像、结构化数据
2. **混合抽取方法**：结合规则、统计、深度学习
3. **实时图谱更新**：支持增量式图谱构建
4. **智能问答推理**：基于图谱的多跳推理
5. **本地化部署**：完整的大模型本地部署方案

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License

## 联系方式

如有问题，请通过Issue或邮件联系。