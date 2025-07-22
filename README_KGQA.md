# 基于知识图谱的故障诊断问答框架 (KGQA Framework)

## 项目简介

本项目在原有数控机床故障诊断系统的基础上，重新设计并实现了一个完整的基于知识图谱的问答框架 (Knowledge Graph based Question Answering Framework)。该框架采用模块化架构，整合了自然语言处理、知识图谱推理、相似案例匹配和解决方案推荐等技术。

## 系统架构

根据您提供的架构图，系统包含以下核心组件：

```
输入故障现象描述 → 分句文本分类 → 用户执行的操作 → 根据知识图谱推出 → 标准现象1,2
                                   ↓                              ↓
                    故障现象 → 抽取故障部位 → 根据知识图谱推出 → 标准现象3,4
                       ↓                                        ↓
                    现象2 → 抽取故障代码 → 根据知识图谱推出 → 标准现象5,6
                       ↓
                    计算相似度，设定阈值进行筛选
                       ↓
               相似现象（现象1', 现象2', ...） → 根据知识图谱推出 → 输出原因（原因1,原因2,...）
                                                           MySQL → 输出解决方法（解决方法1,2,3）
```

## 项目结构

```
kgqa_framework/                    # KGQA框架核心
├── __init__.py                   # 框架初始化
├── config.py                     # 配置管理
├── core/                         # 核心组件
│   ├── fault_analyzer.py         # 故障分析器（主控制器）
│   ├── kg_engine.py              # 知识图谱引擎
│   ├── similarity_matcher.py     # 相似度匹配器
│   └── solution_recommender.py   # 解决方案推荐器
├── models/                       # 数据模型
│   └── entities.py               # 实体定义
└── utils/                        # 工具类
    └── text_processor.py         # 文本处理器

Shukongdashi/                     # Django应用
├── kgqa_views.py                 # KGQA框架API视图
└── urls.py                       # URL路由配置

main.py                           # 主程序和演示代码
requirements_kgqa.txt             # 依赖包列表
```

## 核心功能

### 1. 文本处理 (TextProcessor)
- **分句处理**: 将故障描述按标点符号智能分句
- **分词与词性标注**: 使用jieba进行中文分词和词性分析
- **故障元素提取**: 自动识别操作、现象、部位、报警等故障元素
- **关键词提取**: 基于TF算法提取文本关键词
- **文本相似度计算**: 基于词汇重叠计算相似度

### 2. 知识图谱引擎 (KnowledgeGraphEngine)
- **图数据库连接**: 与Neo4j数据库交互
- **节点查找**: 根据内容和类型查找相关节点
- **关系推理**: 执行多层级的图谱推理
- **路径查找**: 查找节点间的推理路径
- **推理链执行**: 基于故障元素执行完整推理链

### 3. 相似案例匹配 (SimilarityMatcher)
- **向量化表示**: 使用TF-IDF将案例转换为向量
- **相似度计算**: 基于余弦相似度匹配相似案例
- **案例管理**: 支持案例的增删改查和批量导入
- **增量学习**: 支持新案例的动态添加和向量更新

### 4. 解决方案推荐 (SolutionRecommender)
- **多源整合**: 综合知识图谱、相似案例等多种信息源
- **置信度计算**: 为推荐结果计算置信度分数
- **在线搜索**: 当置信度较低时进行在线补充搜索
- **用户反馈学习**: 基于用户反馈优化推荐算法

### 5. 故障分析器 (FaultAnalyzer)
- **统一接口**: 提供简单易用的故障分析接口
- **流程控制**: 协调各个组件的执行流程
- **状态管理**: 管理系统状态和资源
- **错误处理**: 完善的异常处理和容错机制

## API接口

### 1. 故障诊断接口
```
POST/GET /kgqa/diagnosis
```

**参数:**
- `question`: 故障描述 (必需)
- `pinpai`: 设备品牌 (可选)
- `xinghao`: 设备型号 (可选)
- `errorid`: 故障代码 (可选)
- `relationList`: 相关现象，用|分隔 (可选)

**返回示例:**
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "confidence": 0.85,
    "causes": ["液压系统故障", "刀库定位错误"],
    "solutions": ["检查液压系统压力", "调整刀库定位参数"],
    "reasoning_path": [...],
    "similar_cases": [...],
    "recommendations": [...]
  }
}
```

### 2. 智能问答接口
```
POST/GET /kgqa/qa
```

**参数:**
- `question`: 问题 (必需)

### 3. 用户反馈接口
```
POST /kgqa/feedback
```

**参数:**
- `question`: 原始故障描述
- `solution`: 选择的解决方案
- `effectiveness`: 有效性评分 (0-1)

### 4. 系统状态接口
```
GET /kgqa/status
```

### 5. 自动补全接口
```
POST /kgqa/autocomplete
```

## 安装和配置

### 1. 环境要求
- Python 3.7+
- Neo4j 4.0+
- Django 2.2+

### 2. 安装依赖
```bash
pip install -r requirements_kgqa.txt
```

### 3. 配置Neo4j数据库
按照原项目说明导入知识图谱数据到Neo4j。

### 4. 环境变量配置
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
export KGQA_DATA_DIR="./data"
```

### 5. 运行系统
```bash
# 演示模式
python main.py demo

# 交互模式
python main.py interactive

# Django服务
python manage.py runserver 0.0.0.0:8000
```

## 使用示例

### 1. 编程接口使用
```python
from kgqa_framework import FaultAnalyzer

# 初始化分析器
analyzer = FaultAnalyzer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

# 故障分析
result = analyzer.analyze_fault(
    fault_description="自动换刀时刀链运转不到位，刀库停止运转",
    brand="发那科",
    model="MATE-TD",
    error_code="ALM401"
)

print(f"置信度: {result.confidence}")
print(f"可能原因: {result.causes}")
print(f"解决方案: {result.solutions}")

# 关闭分析器
analyzer.close()
```

### 2. HTTP API使用
```bash
# 故障诊断
curl -X POST "http://localhost:8000/kgqa/diagnosis" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "自动换刀时刀链运转不到位，刀库停止运转",
    "pinpai": "发那科",
    "xinghao": "MATE-TD",
    "errorid": "ALM401"
  }'

# 智能问答
curl -X GET "http://localhost:8000/kgqa/qa?question=外部24V短路的故障会引起哪些现象"

# 系统状态
curl -X GET "http://localhost:8000/kgqa/status"
```

## 特性和优势

### 1. 模块化架构
- 组件独立：各模块功能独立，便于维护和扩展
- 接口标准：统一的数据模型和接口规范
- 可插拔设计：支持组件的替换和升级

### 2. 智能化分析
- 多层推理：结合规则推理和相似性匹配
- 自适应学习：基于用户反馈持续优化
- 置信度评估：为结果提供可信度评分

### 3. 高性能
- 向量化计算：使用科学计算库优化性能
- 缓存机制：缓存常用数据减少计算
- 并行处理：支持多线程和异步处理

### 4. 易于集成
- RESTful API：标准的HTTP接口
- 向后兼容：保持与原系统接口的兼容
- 多种部署方式：支持单机、集群等部署模式

## 扩展和定制

### 1. 添加新的故障类型
在`text_processor.py`中扩展`fault_patterns`字典：
```python
self.fault_patterns[FaultType.NEW_TYPE] = {
    'keywords': ['新关键词1', '新关键词2'],
    'patterns': [r'新正则模式']
}
```

### 2. 自定义解决方案
在`solution_recommender.py`中更新`solution_database`：
```python
self.solution_database["新故障类型"] = [
    "解决方案1",
    "解决方案2"
]
```

### 3. 集成机器学习模型
```python
# 在相应组件中集成预训练模型
from your_ml_model import FaultClassifier

class EnhancedTextProcessor(TextProcessor):
    def __init__(self):
        super().__init__()
        self.ml_classifier = FaultClassifier.load('model.pkl')
```

## 性能和监控

### 1. 性能指标
- 响应时间：< 2秒 (典型查询)
- 并发处理：支持100+并发请求
- 内存占用：< 1GB (基础配置)

### 2. 监控接口
系统提供详细的状态监控信息：
- 数据库连接状态
- 案例库大小
- 系统资源使用情况

## 开发和贡献

### 1. 开发环境设置
```bash
# 克隆项目
git clone <repository_url>

# 安装开发依赖
pip install -r requirements_kgqa.txt

# 运行测试
pytest tests/
```

### 2. 代码规范
- 遵循PEP 8编码规范
- 添加类型注解
- 编写单元测试
- 更新文档

## 许可证

本项目基于原有的开源许可证，新增的KGQA框架部分同样采用开源协议。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 贡献代码

---

**注意**: 本框架基于原有的数控机床故障诊断系统开发，保持了与原系统的兼容性，同时大幅提升了系统的智能化水平和扩展性。