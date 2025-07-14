# 装备制造故障知识图谱构建项目

## 项目概述

本项目是一个完整的装备制造故障知识图谱构建系统，包含数据采集、实体抽取、关系抽取、知识图谱构建和智能问答等功能。

## 项目结构

```
equipment_fault_kg/
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后数据
│   └── annotated/           # 标注数据
├── src/                     # 源代码
│   ├── data_collection/     # 数据采集模块
│   ├── entity_extraction/   # 实体抽取模块
│   ├── relation_extraction/ # 关系抽取模块
│   ├── kg_construction/     # 知识图谱构建
│   ├── llm_deployment/      # 本地大模型部署
│   └── neo4j_qa/           # Neo4j问答系统
├── models/                  # 模型文件
├── config/                  # 配置文件
├── notebooks/               # Jupyter notebooks
├── tests/                   # 测试文件
└── docs/                    # 文档
```

## 主要功能

1. **数据采集模块**
   - 装备故障案例数据爬取
   - 故障诊断手册数据提取
   - 专家知识库数据整理

2. **实体抽取模块**
   - 装备实体识别
   - 故障类型实体识别
   - 故障原因实体识别
   - 解决方案实体识别

3. **关系抽取模块**
   - 装备-故障关系抽取
   - 故障-原因关系抽取
   - 故障-解决方案关系抽取
   - 装备-部件关系抽取

4. **知识图谱构建**
   - Neo4j图数据库设计
   - 实体关系建模
   - 图谱可视化

5. **本地大模型部署**
   - 信息抽取模型训练
   - 模型本地部署
   - API接口开发

6. **智能问答系统**
   - 基于Neo4j的问答逻辑
   - 自然语言查询处理
   - 故障诊断推理

## 技术栈

- **编程语言**: Python 3.8+
- **数据库**: Neo4j, SQLite
- **机器学习**: PyTorch, Transformers
- **大模型**: ChatGLM, LLaMA
- **Web框架**: Flask, FastAPI
- **前端**: Vue.js, ECharts
- **数据处理**: Pandas, NumPy
- **可视化**: NetworkX, Matplotlib

## 安装和运行

1. 克隆项目
```bash
git clone <repository-url>
cd equipment_fault_kg
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境
```bash
cp config/config.example.yaml config/config.yaml
# 编辑配置文件
```

4. 运行项目
```bash
python src/main.py
```

## 使用说明

详细的安装和使用说明请参考 `docs/` 目录下的文档。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License