# 装备制造故障知识图谱构建项目指南

## 项目概述

本项目是一个完整的装备制造故障知识图谱构建系统，旨在通过人工智能技术自动构建装备故障诊断知识库，并提供智能问答服务。

## 技术架构

### 1. 数据采集层
- **网络爬虫**: 自动采集装备故障案例数据
- **PDF提取器**: 从技术手册中提取故障信息
- **API接口**: 集成专家知识库数据

### 2. 数据处理层
- **文本清洗**: 去除噪声和格式化数据
- **信息提取**: 识别装备、故障、原因等关键信息
- **数据标注**: 为机器学习模型准备训练数据

### 3. 知识抽取层
- **实体识别**: 使用BERT等预训练模型识别装备制造领域实体
- **关系抽取**: 抽取实体间的语义关系
- **规则引擎**: 基于领域知识的规则抽取

### 4. 知识图谱层
- **图数据库**: 使用Neo4j存储知识图谱
- **图构建**: 自动构建实体关系图
- **图查询**: 支持复杂的图查询操作

### 5. 应用服务层
- **问答系统**: 基于知识图谱的智能问答
- **大模型集成**: 集成本地大模型进行信息抽取
- **Web接口**: 提供RESTful API服务

## 核心功能模块

### 1. 数据采集模块 (`src/data_collection/`)

#### 网络爬虫 (`crawler.py`)
```python
from data_collection import WebCrawler

# 初始化爬虫
crawler = WebCrawler(config)

# 爬取故障案例
cases = crawler.crawl_fault_cases("https://example.com/fault-cases")
```

#### PDF提取器 (`crawler.py`)
```python
from data_collection import PDFExtractor

# 提取PDF文本
extractor = PDFExtractor()
text = extractor.extract_text("manual.pdf")

# 提取故障信息
fault_data = extractor.extract_fault_manual("manual.pdf")
```

#### 数据处理器 (`processor.py`)
```python
from data_collection import DataProcessor

# 处理原始数据
processor = DataProcessor(config)
processed_data = processor.process_batch(raw_data)

# 保存处理结果
processor.save_processed_data(processed_data, "output.json")
```

### 2. 实体抽取模块 (`src/entity_extraction/`)

#### NER模型 (`ner_model.py`)
```python
from entity_extraction import NERModel

# 初始化模型
ner_model = NERModel(config)

# 训练模型
train_loader, val_loader = ner_model.prepare_data(texts, labels)
ner_model.train(train_loader, val_loader)

# 预测实体
entities = ner_model.predict("数控车床主轴异常振动")
```

### 3. 大模型部署模块 (`src/llm_deployment/`)

#### 模型加载器 (`model_loader.py`)
```python
from llm_deployment import ModelLoader

# 加载大模型
loader = ModelLoader(config)
loader.load_model()

# 实体抽取
entities = loader.extract_entities(text)

# 关系抽取
relations = loader.extract_relations(text)

# 问答
answer = loader.answer_question(question)
```

### 4. 知识图谱模块 (`src/neo4j_qa/`)

#### 图管理器 (`graph_manager.py`)
```python
from neo4j_qa import GraphManager

# 连接数据库
graph_manager = GraphManager(config)

# 构建知识图谱
graph_manager.build_knowledge_graph(entities, relations)

# 查询装备故障
results = graph_manager.query_equipment_faults("数控车床")

# 查询故障原因
causes = graph_manager.query_fault_causes("主轴振动")

# 查询解决方案
solutions = graph_manager.query_fault_solutions("主轴振动")
```

## 安装和配置

### 1. 环境要求
- Python 3.8+
- Neo4j 4.4+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd equipment_fault_kg

# 安装Python依赖
pip install -r requirements.txt

# 安装Neo4j (Ubuntu/Debian)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j
```

### 3. 配置Neo4j
```bash
# 启动Neo4j服务
sudo systemctl start neo4j

# 设置密码
cypher-shell -u neo4j -p neo4j
# 在Neo4j shell中执行：
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'your_password'
```

### 4. 配置文件
编辑 `config/config.yaml`:
```yaml
database:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "your_password"
    database: "neo4j"

llm:
  model_name: "THUDM/chatglm2-6b"
  model_path: "models/chatglm2-6b"
  quantization:
    load_in_8bit: true
```

## 使用指南

### 1. 快速开始
```bash
# 运行演示
python run_demo.py

# 运行主程序
python src/main.py --mode interactive
```

### 2. 数据采集
```python
from src.main import EquipmentFaultKG

# 创建系统实例
kg_system = EquipmentFaultKG()

# 采集数据
raw_data = kg_system.collect_data()

# 处理数据
processed_data = kg_system.process_data(raw_data)
```

### 3. 知识图谱构建
```python
# 抽取实体和关系
entities, relations = kg_system.extract_entities(processed_data)

# 构建知识图谱
kg_system.build_knowledge_graph(entities, relations)
```

### 4. 智能问答
```python
# 交互式问答
kg_system.interactive_qa()

# 单次问答
answer = kg_system.answer_question("数控车床主轴振动故障的原因是什么？")
```

## 数据格式

### 1. 原始数据格式
```json
{
  "id": "001",
  "title": "数控车床主轴异常振动故障诊断",
  "content": "某工厂数控车床在加工过程中出现主轴异常振动...",
  "source": "故障案例库",
  "url": "https://example.com/case/001"
}
```

### 2. 处理后数据格式
```json
{
  "id": "001",
  "title": "数控车床主轴异常振动故障诊断",
  "content": "某工厂数控车床在加工过程中出现主轴异常振动...",
  "equipment_info": {
    "equipment_type": "数控车床",
    "model": "CK6136",
    "manufacturer": "沈阳机床",
    "components": ["主轴", "轴承"]
  },
  "fault_info": {
    "fault_type": "机械故障",
    "symptoms": ["主轴异常振动"],
    "causes": ["轴承磨损严重"],
    "solutions": ["更换轴承"]
  }
}
```

### 3. 实体格式
```json
{
  "type": "Equipment",
  "text": "数控车床",
  "start": 0,
  "end": 4
}
```

### 4. 关系格式
```json
{
  "head": "数控车床",
  "relation": "HAS_FAULT",
  "tail": "主轴异常振动"
}
```

## 实体和关系类型

### 实体类型
- **Equipment**: 装备（数控机床、车床等）
- **Component**: 部件（主轴、伺服电机等）
- **Fault**: 故障（振动、过热等）
- **Cause**: 原因（磨损、老化等）
- **Solution**: 解决方案（更换、维修等）
- **Symptom**: 症状（异常、报警等）
- **Material**: 材料（润滑油、轴承等）
- **Tool**: 工具（扳手、测量仪等）

### 关系类型
- **HAS_FAULT**: 装备-故障关系
- **HAS_COMPONENT**: 装备-部件关系
- **CAUSES**: 原因-故障关系
- **SOLVES**: 解决方案-故障关系
- **HAS_SYMPTOM**: 故障-症状关系
- **REQUIRES_TOOL**: 解决方案-工具关系
- **REQUIRES_MATERIAL**: 解决方案-材料关系

## 性能优化

### 1. 模型优化
- 使用量化技术减少模型大小
- 使用GPU加速推理
- 模型蒸馏和剪枝

### 2. 数据处理优化
- 批量处理提高效率
- 多进程并行处理
- 数据缓存机制

### 3. 查询优化
- 建立数据库索引
- 查询结果缓存
- 图查询优化

## 扩展开发

### 1. 添加新的实体类型
```python
# 在config.yaml中添加新实体类型
entity_types:
  - NewEntity

# 在NER模型中添加标签
label2id = {
    'B-NewEntity': 17,
    'I-NewEntity': 18
}
```

### 2. 添加新的关系类型
```python
# 在config.yaml中添加新关系类型
relation_types:
  - NEW_RELATION

# 在图管理器中添加查询方法
def query_new_relation(self, entity_name: str):
    query = """
    MATCH (e:Entity)-[:NEW_RELATION]->(t:Target)
    WHERE e.name CONTAINS $entity_name
    RETURN e, t
    """
    return self.graph.run(query, entity_name=entity_name).data()
```

### 3. 集成新的数据源
```python
# 创建新的数据采集器
class NewDataSourceCollector:
    def collect_data(self, source_config):
        # 实现数据采集逻辑
        pass

# 在主程序中集成
def collect_data(self):
    # 添加新的数据源
    new_collector = NewDataSourceCollector()
    new_data = new_collector.collect_data(config['new_source'])
    return new_data
```

## 故障排除

### 1. Neo4j连接问题
```bash
# 检查Neo4j服务状态
sudo systemctl status neo4j

# 检查端口是否开放
netstat -tlnp | grep 7687

# 重启Neo4j服务
sudo systemctl restart neo4j
```

### 2. 模型加载问题
```python
# 检查GPU可用性
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查模型文件
import os
print(f"Model exists: {os.path.exists('models/chatglm2-6b')}")
```

### 3. 内存不足问题
```python
# 减少批处理大小
config['model']['batch_size'] = 8

# 使用量化
config['llm']['quantization']['load_in_8bit'] = True
```

## 贡献指南

### 1. 代码规范
- 使用Python PEP 8规范
- 添加详细的文档字符串
- 编写单元测试

### 2. 提交规范
- 使用清晰的提交信息
- 一个提交只包含一个功能
- 提交前运行测试

### 3. 问题报告
- 提供详细的错误信息
- 包含复现步骤
- 说明环境信息

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

*本指南将随着项目的发展持续更新，请关注最新版本。*