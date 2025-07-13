# 知识图谱实战项目部署指南

## 📋 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细安装步骤](#详细安装步骤)
- [服务启动](#服务启动)
- [使用指南](#使用指南)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

## 🖥️ 环境要求

### 系统要求
- **操作系统**: Linux (推荐 Ubuntu 20.04+), macOS, Windows 10+
- **Python**: 3.9+
- **内存**: 最低 8GB, 推荐 16GB+
- **存储**: 最低 50GB 可用空间
- **GPU**: 可选，推荐 NVIDIA GPU (8GB+ 显存)

### 依赖服务
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Neo4j**: 5.0+ (通过Docker部署)
- **Redis**: 7.0+ (通过Docker部署)

## 🚀 快速开始

### 方式一：Docker Compose 一键部署 (推荐)

```bash
# 1. 克隆项目
git clone <your-repository-url>
cd knowledge-graph-project

# 2. 启动所有服务
docker-compose up -d

# 3. 等待服务启动完成 (约2-3分钟)
docker-compose ps

# 4. 访问Web界面
open http://localhost:8501
```

### 方式二：本地开发环境

```bash
# 1. 创建虚拟环境
conda create -n kg_project python=3.9
conda activate kg_project

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动数据库服务
docker-compose up neo4j redis mongodb -d

# 4. 启动前端应用
streamlit run frontend/app.py
```

## 📦 详细安装步骤

### 步骤1: 环境准备

#### 1.1 安装 Docker

**Ubuntu/Debian:**
```bash
# 更新包索引
sudo apt update

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

**macOS:**
```bash
# 使用Homebrew
brew install docker docker-compose

# 或下载Docker Desktop
# https://www.docker.com/products/docker-desktop
```

**Windows:**
```bash
# 下载并安装Docker Desktop
# https://www.docker.com/products/docker-desktop

# 在PowerShell中验证
docker --version
docker-compose --version
```

#### 1.2 安装 Python 环境

**使用 Conda (推荐):**
```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建项目环境
conda create -n kg_project python=3.9
conda activate kg_project
```

**使用 pyenv:**
```bash
# 安装pyenv
curl https://pyenv.run | bash

# 安装Python 3.9
pyenv install 3.9.16
pyenv global 3.9.16

# 创建虚拟环境
python -m venv kg_project
source kg_project/bin/activate  # Linux/macOS
# kg_project\Scripts\activate  # Windows
```

### 步骤2: 项目配置

#### 2.1 克隆项目
```bash
git clone <your-repository-url>
cd knowledge-graph-project
```

#### 2.2 安装Python依赖
```bash
# 激活虚拟环境
conda activate kg_project

# 安装基础依赖
pip install -r requirements.txt

# 安装GPU支持 (可选)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2.3 环境变量配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量 (可选)
vim .env
```

**.env 文件示例:**
```bash
# 环境配置
ENV=development
DEBUG=True
LOG_LEVEL=INFO

# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledge123

REDIS_URL=redis://:redis123@localhost:6379
MONGODB_URL=mongodb://admin:mongo123@localhost:27017

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# GPU配置
CUDA_VISIBLE_DEVICES=0
```

### 步骤3: 数据库初始化

#### 3.1 启动数据库服务
```bash
# 启动所有数据库
docker-compose up neo4j redis mongodb elasticsearch -d

# 检查服务状态
docker-compose ps
```

#### 3.2 验证数据库连接
```bash
# 测试Neo4j连接
python -c "
from py2neo import Graph
graph = Graph('bolt://localhost:7687', auth=('neo4j', 'knowledge123'))
print('Neo4j connected successfully')
"

# 测试Redis连接
python -c "
import redis
r = redis.from_url('redis://:redis123@localhost:6379')
print('Redis connected successfully')
"
```

## 🔧 服务启动

### 启动方式一：Docker Compose (生产环境)

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 启动方式二：本地开发 (开发环境)

```bash
# 1. 启动数据库服务
docker-compose up neo4j redis mongodb -d

# 2. 启动后端API (新终端)
cd knowledge-graph-project
conda activate kg_project
python qa_system/main.py

# 3. 启动前端界面 (新终端)
cd knowledge-graph-project
conda activate kg_project
streamlit run frontend/app.py

# 4. 启动Jupyter开发环境 (可选)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 服务端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端界面 | 8501 | Streamlit Web界面 |
| 后端API | 8000 | FastAPI接口 |
| Neo4j | 7474/7687 | 图数据库 |
| Redis | 6379 | 缓存数据库 |
| MongoDB | 27017 | 文档数据库 |
| Elasticsearch | 9200 | 搜索引擎 |
| Kibana | 5601 | 可视化界面 |
| Jupyter | 8888 | 开发环境 |

## 📚 使用指南

### 数据采集

#### 1. 百科数据采集
```bash
# 命令行方式
python data_collection/crawler.py --keywords "人工智能" "机器学习" --no-news

# Web界面方式
# 访问 http://localhost:8501 -> 数据采集 -> 百科数据
```

#### 2. 新闻数据采集
```bash
# 采集新闻数据
python data_collection/crawler.py --no-encyclopedia

# 自定义采集源
python data_collection/crawler.py --news-sites "sina.com" "163.com"
```

#### 3. 文档处理
```bash
# 处理PDF文档
python data_collection/crawler.py --document-dir ./documents
```

### 模型训练

#### 1. 实体识别模型
```bash
# 训练NER模型
python models/ner_model.py --train --train-data ./data/ner_train.json

# 使用预训练模型预测
python models/ner_model.py --predict "李彦宏是百度公司的创始人"
```

#### 2. 关系抽取模型
```bash
# 训练关系抽取模型
python models/relation_extraction.py --train --train-data ./data/relation_train.json

# 预测实体关系
python models/relation_extraction.py --predict "李彦宏是百度公司的创始人" --entity1 "李彦宏" --entity2 "百度公司"
```

### 知识图谱构建

#### 1. 从文本构建
```bash
# 从文本文件构建图谱
python knowledge_graph/graph_builder.py --build-from-text ./data/texts.txt --ner-model ./models/ner --re-model ./models/relation_extraction

# 清空图数据库
python knowledge_graph/graph_builder.py --clear
```

#### 2. 从三元组文件导入
```bash
# 导入JSON格式的三元组
python knowledge_graph/graph_builder.py --build-from-file ./data/triples.json

# 导入CSV格式的三元组
python knowledge_graph/graph_builder.py --build-from-file ./data/triples.csv
```

#### 3. 图谱查询
```bash
# 查找实体
python knowledge_graph/graph_builder.py --find-entity "李彦宏"

# 执行Cypher查询
python knowledge_graph/graph_builder.py --query "MATCH (n:Person) RETURN n.name LIMIT 10"

# 可视化子图
python knowledge_graph/graph_builder.py --visualize "李彦宏"
```

### 智能问答

#### 1. 命令行问答
```bash
# 单个问题
python qa_system/qa_engine.py --question "李彦宏是谁？"

# 交互式问答
python qa_system/qa_engine.py --interactive

# 批量问答
python qa_system/qa_engine.py --batch-file ./questions.txt
```

#### 2. API调用
```bash
# 启动API服务
python qa_system/main.py

# 测试API
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "李彦宏是谁？"}'
```

#### 3. Web界面使用
```bash
# 启动Web界面
streamlit run frontend/app.py

# 访问界面
open http://localhost:8501
```

## ❓ 常见问题

### Q1: Docker服务启动失败
**问题**: `docker-compose up` 失败

**解决方案**:
```bash
# 检查Docker状态
sudo systemctl status docker

# 重启Docker服务
sudo systemctl restart docker

# 清理Docker缓存
docker system prune -a

# 重新构建镜像
docker-compose build --no-cache
```

### Q2: Neo4j连接失败
**问题**: `Unable to connect to Neo4j`

**解决方案**:
```bash
# 检查Neo4j状态
docker-compose logs neo4j

# 重启Neo4j服务
docker-compose restart neo4j

# 修改密码 (如果需要)
docker exec -it kg_neo4j cypher-shell -u neo4j -p neo4j
# 在cypher-shell中执行: ALTER USER neo4j SET PASSWORD 'knowledge123'
```

### Q3: 模型加载失败
**问题**: `Model file not found`

**解决方案**:
```bash
# 检查模型文件路径
ls -la models/

# 下载预训练模型
python -c "
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
tokenizer.save_pretrained('./models/bert-base-chinese')
model.save_pretrained('./models/bert-base-chinese')
"
```

### Q4: 内存不足错误
**问题**: `CUDA out of memory` 或 `Out of memory`

**解决方案**:
```python
# 在config/settings.py中调整批次大小
MODEL_CONFIG = {
    "ner": {
        "batch_size": 8,  # 减小批次大小
        # ...
    },
    "relation_extraction": {
        "batch_size": 8,  # 减小批次大小
        # ...
    }
}

# 或使用CPU训练
export CUDA_VISIBLE_DEVICES=""
```

### Q5: 端口冲突
**问题**: `Port already in use`

**解决方案**:
```bash
# 查找占用端口的进程
sudo lsof -i :8501
sudo lsof -i :7474

# 杀死进程
sudo kill -9 <PID>

# 或修改docker-compose.yml中的端口映射
ports:
  - "8502:8501"  # 改为其他端口
```

## ⚡ 性能优化

### 硬件优化

#### GPU加速
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证GPU可用性
python -c "import torch; print(torch.cuda.is_available())"

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0
```

#### 内存优化
```python
# 在config/settings.py中优化内存设置
TRAINING_CONFIG = {
    "mixed_precision": True,  # 启用混合精度
    "gradient_accumulation_steps": 4,  # 梯度累积
    "dataloader_num_workers": 2,  # 减少数据加载进程
}
```

### 数据库优化

#### Neo4j优化
```bash
# 在docker-compose.yml中调整Neo4j内存
environment:
  NEO4J_dbms_memory_heap_initial__size: 1G
  NEO4j_dbms_memory_heap_max__size: 4G
  NEO4J_dbms_memory_pagecache_size: 2G
```

#### Redis优化
```bash
# Redis配置优化
command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

### 应用层优化

#### 并发优化
```python
# 在config/settings.py中调整API配置
API_CONFIG = {
    "workers": 4,  # 增加worker数量
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 1000,
}
```

#### 缓存优化
```python
# 启用多级缓存
QA_CONFIG = {
    "use_cache": True,
    "cache_ttl": 3600,
    "memory_cache_size": 1000,
}
```

## 📊 监控和维护

### 健康检查
```bash
# 检查所有服务状态
curl http://localhost:8000/health

# 检查数据库连接
python -c "
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
builder = KnowledgeGraphBuilder()
print('Graph database connection OK')
"
```

### 日志管理
```bash
# 查看应用日志
tail -f logs/app.log

# 查看Docker服务日志
docker-compose logs -f qa_system

# 设置日志轮转
# 在config/settings.py中配置LOGGING_CONFIG
```

### 备份和恢复
```bash
# 备份Neo4j数据
docker exec kg_neo4j neo4j-admin dump --database=neo4j --to=/var/lib/neo4j/backups/backup.dump

# 恢复Neo4j数据
docker exec kg_neo4j neo4j-admin load --from=/var/lib/neo4j/backups/backup.dump --database=neo4j --force
```

## 📞 技术支持

如果您在部署过程中遇到问题，请通过以下方式获取帮助：

1. **查看文档**: 首先查看本部署指南和README.md
2. **检查日志**: 查看应用和Docker服务日志
3. **搜索问题**: 在GitHub Issues中搜索类似问题
4. **提交Issue**: 在GitHub仓库中创建新的Issue
5. **联系维护者**: 通过邮件联系项目维护者

---

## 📝 版本说明

- **v1.0.0**: 初始版本，包含基础功能
- **v1.1.0**: 添加大模型支持和性能优化
- **v1.2.0**: 增加Web界面和批量处理功能

---

**最后更新**: 2025-01-07
**维护者**: [Your Name]
**项目地址**: [GitHub Repository]