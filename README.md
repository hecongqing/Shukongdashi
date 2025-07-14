# 装备制造故障知识图谱构建及应用案例剖析

## 项目概述

本项目是一个基于知识图谱的装备制造故障诊断专家系统，主要面向数控机床、工业设备等装备制造领域的故障诊断与维修指导。系统采用自然语言处理、实体关系抽取、知识图谱构建等技术，构建了一个完整的故障诊断知识库和问答系统。

## 技术栈

### 后端技术
- **Python 3.8+**: 主要开发语言
- **Neo4j**: 图数据库，用于存储知识图谱
- **FastAPI**: Web框架，提供API服务
- **spaCy**: 自然语言处理库
- **transformers**: 预训练模型库
- **torch**: 深度学习框架
- **pandas**: 数据处理
- **numpy**: 数值计算

### 前端技术
- **Vue.js 3**: 前端框架
- **Element Plus**: UI组件库
- **ECharts**: 数据可视化
- **D3.js**: 图谱可视化

### 数据库与存储
- **Neo4j**: 图数据库存储知识图谱
- **MySQL**: 存储结构化数据
- **Redis**: 缓存服务

## 功能特性

### 1. 数据采集与预处理
- 多源数据采集（PDF文档、网页、数据库等）
- 数据清洗和标注
- 文本预处理和分词

### 2. 实体关系抽取
- 基于深度学习的命名实体识别（NER）
- 关系抽取模型
- 实体链接和消歧

### 3. 知识图谱构建
- 实体-关系-属性三元组构建
- 图谱质量评估
- 知识融合与更新

### 4. 本地大模型部署
- 支持多种开源大模型（LLaMA、ChatGLM等）
- 模型量化和加速
- 信息抽取任务适配

### 5. 故障诊断系统
- 基于规则的推理诊断
- 智能问答系统
- 故障预测与预警

### 6. 可视化界面
- 知识图谱可视化
- 故障诊断流程展示
- 系统管理界面

## 项目结构

```
equipment-fault-kg/
├── backend/                 # 后端代码
│   ├── api/                # API接口
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   ├── utils/              # 工具函数
│   └── config/             # 配置文件
├── frontend/               # 前端代码
│   ├── src/                # 源代码
│   ├── public/             # 静态资源
│   └── dist/               # 构建输出
├── data/                   # 数据文件
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── knowledge/          # 知识库数据
├── models/                 # 机器学习模型
│   ├── ner/                # 命名实体识别
│   ├── relation/           # 关系抽取
│   └── llm/                # 大语言模型
├── scripts/                # 脚本文件
│   ├── data_collection/    # 数据采集
│   ├── preprocessing/      # 数据预处理
│   ├── training/           # 模型训练
│   └── deployment/         # 部署脚本
├── tests/                  # 测试文件
├── docs/                   # 文档
├── requirements.txt        # Python依赖
├── docker-compose.yml      # Docker配置
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/equipment-fault-kg.git
cd equipment-fault-kg

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据库配置

```bash
# 启动Neo4j数据库
docker-compose up -d neo4j

# 启动MySQL数据库
docker-compose up -d mysql

# 启动Redis缓存
docker-compose up -d redis
```

### 3. 数据初始化

```bash
# 数据采集
python scripts/data_collection/collect_data.py

# 数据预处理
python scripts/preprocessing/preprocess_data.py

# 构建知识图谱
python scripts/knowledge_graph/build_kg.py
```

### 4. 模型训练

```bash
# 训练NER模型
python scripts/training/train_ner.py

# 训练关系抽取模型
python scripts/training/train_relation.py

# 部署大模型
python scripts/deployment/deploy_llm.py
```

### 5. 启动服务

```bash
# 启动后端服务
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 启动前端服务
cd frontend
npm install
npm run dev
```

## 使用说明

### 1. 故障诊断
- 访问 `http://localhost:3000` 打开前端界面
- 输入故障描述或选择故障类型
- 系统会基于知识图谱进行推理诊断
- 查看诊断结果和维修建议

### 2. 知识图谱浏览
- 点击"知识图谱"菜单
- 浏览设备实体、故障类型、维修方法等
- 支持图谱搜索和过滤功能

### 3. 问答系统
- 使用自然语言提问
- 系统基于知识图谱和大模型回答
- 支持多轮对话和上下文理解

## 部署指南

### Docker部署

```bash
# 构建镜像
docker-compose build

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

### 生产环境部署

请参考 `docs/deployment.md` 获取详细的生产环境部署指南。

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目使用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目负责人: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [https://github.com/yourusername/equipment-fault-kg]

## 致谢

感谢以下开源项目对本项目的支持：
- [Neo4j](https://neo4j.com/)
- [spaCy](https://spacy.io/)
- [transformers](https://huggingface.co/transformers/)
- [Vue.js](https://vuejs.org/)
- [ECharts](https://echarts.apache.org/)
