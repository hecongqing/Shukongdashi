# 医疗健康知识图谱实战项目

这是一个完整的医疗健康领域知识图谱构建项目，涵盖了从数据采集到智能问答系统的全流程实现。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/medical-kg-project.git
cd medical-kg-project

# 创建虚拟环境
conda create -n medical_kg python=3.8
conda activate medical_kg

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

```bash
# 下载中文医疗BERT模型
python scripts/download_models.py --model chinese-bert-wwm
```

### 3. 启动Neo4j数据库

```bash
# 使用Docker启动Neo4j
docker-compose up -d neo4j
```

### 4. 运行示例

```bash
# 数据采集示例
python src/data_collection/run_spider.py --source baidu_baike

# 实体抽取示例
python src/models/ner_demo.py --text "糖尿病患者常见症状包括多饮、多尿"

# 知识图谱构建
python src/kg_builder/build_graph.py --input data/processed/

# 启动问答系统
python src/qa_system/app.py
```

## 📁 项目结构

```
medical_kg_project/
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── annotations/          # 标注数据
├── models/                   # 模型文件
│   ├── ner/                 # NER模型
│   ├── relation/            # 关系抽取模型
│   └── llm/                 # 大模型
├── src/                     # 源代码
│   ├── data_collection/     # 数据采集
│   ├── preprocessing/       # 数据预处理
│   ├── models/             # 模型训练
│   ├── kg_builder/         # 图谱构建
│   └── qa_system/          # 问答系统
├── scripts/                 # 工具脚本
├── notebooks/              # Jupyter notebooks
├── tests/                  # 测试代码
├── docs/                   # 文档
└── docker-compose.yml      # Docker配置
```

## 🛠️ 核心功能

### 1. 数据采集
- 百度百科医疗词条爬虫
- 医学文献PDF解析
- 医疗网站数据采集

### 2. 信息抽取
- 基于BERT的命名实体识别
- 关系抽取（疾病-症状、疾病-药物等）
- 大模型辅助抽取

### 3. 知识图谱构建
- 实体消歧与融合
- Neo4j图数据库存储
- 知识图谱可视化

### 4. 智能问答
- 意图识别
- 实体链接
- 基于图谱的答案生成

## 📊 数据集

项目提供了示例数据集用于快速体验：
- 1000条标注的医疗文本
- 5000个医疗实体
- 10000条关系三元组

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

## 📄 许可证

MIT License