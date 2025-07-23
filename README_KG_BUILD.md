# 知识图谱构建指南

本文档说明如何使用修改后的 easy_kgqa_framework 构建你的知识图谱。

## 修改内容

### 1. 新增实体类型
- 主体 (Subject)
- 客体 (Object)  
- 部件单元 (ComponentUnit)
- 故障状态 (FaultState)
- 性能表征 (PerformanceFeature)
- 检测工具 (DetectionTool)

### 2. 新增关系类型
- 部件故障
- 性能故障
- 检测工具
- 组成

### 3. 新增功能模块
- `GraphManager` 类：负责知识图谱的构建和管理
- 增强的 `KnowledgeGraphEngine`：支持新的实体和关系类型查询
- 配置文件支持：使用 YAML 格式的配置

## 使用步骤

### 1. 安装依赖
```bash
pip install -r requirements_easy_kgqa.txt
```

### 2. 配置数据库
编辑 `config/config.yaml` 文件：
```yaml
database:
  neo4j:
    uri: "bolt://localhost:50002"  # 你的Neo4j地址
    username: "neo4j"
    password: "your_password"      # 你的密码
```

### 3. 准备数据
将你的数据文件 `train.json` 放在 `data/` 目录下。数据格式示例：
```json
{
  "spo_list": [
    {
      "h": {"name": "主轴电机"},
      "t": {"name": "轴承故障"},
      "relation": "部件故障"
    }
  ]
}
```

### 4. 构建知识图谱

#### 方法1：使用改进的构建脚本
```bash
python build_knowledge_graph.py
```

#### 方法2：使用你原来的代码（需要修改）
将你原来的代码修改为：
```python
import json
from easy_kgqa_framework.utils.graph_manager import GraphManager
import yaml

# 1. 加载Neo4j配置
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

graph_manager = GraphManager(config['database']['neo4j'])

# 2. 读取数据
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())

def extract_entities_relations(data_iter):
    entities_dict = {}
    relations = []
    for item in data_iter:
        for spo in item.get('spo_list', []):
            h = spo['h']
            t = spo['t']
            rel = spo['relation']
            
            # 改进的实体类型分类
            h_type = classify_entity_type(h['name'], rel)
            t_type = classify_entity_type(t['name'], rel)

            # 添加头实体
            if h['name'] not in entities_dict:
                entities_dict[h['name']] = {
                    "type": h_type,
                    "text": h['name'],
                    "description": f"实体类型: {h_type}"
                }
            # 添加尾实体
            if t['name'] not in entities_dict:
                entities_dict[t['name']] = {
                    "type": t_type,
                    "text": t['name'],
                    "description": f"实体类型: {t_type}"
                }
            # 添加关系
            relations.append({
                "head": h['name'],
                "head_type": h_type,
                "tail": t['name'],
                "tail_type": t_type,
                "relation": rel
            })
    return list(entities_dict.values()), relations

def classify_entity_type(entity_name, relation=""):
    """改进的实体分类函数"""
    if any(keyword in entity_name for keyword in ['电机', '轴承', '齿轮', '泵', '阀门']):
        return "部件单元"
    elif any(keyword in entity_name for keyword in ['故障', '损坏', '失效']):
        return "故障状态"
    elif any(keyword in entity_name for keyword in ['检测', '测试', '仪器']):
        return "检测工具"
    elif any(keyword in entity_name for keyword in ['性能', '效率', '温度', '压力']):
        return "性能表征"
    elif "部件故障" in relation:
        return "部件单元" if "主体" not in entity_name else "主体"
    else:
        return "主体"

# 3. 批量处理数据
entities, relations = [], []
for path in ["data/train.json"]:
    data_iter = load_data(path)
    ents, rels = extract_entities_relations(data_iter)
    entities.extend(ents)
    relations.extend(rels)

# 去重实体
unique_entities = {e['text']: e for e in entities}.values()

# 4. 构建知识图谱
graph_manager.build_knowledge_graph(list(unique_entities), relations)

# 5. 打印统计信息
stats = graph_manager.get_statistics()
print("✓ 知识图谱构建完成")
print(f"  节点统计: {stats.get('nodes', {})}")
print(f"  关系统计: {stats.get('relations', {})}")
```

### 5. 测试构建结果
```bash
python test_kg_build.py
```

## 查询示例

### 1. 按实体类型查询
```python
from easy_kgqa_framework.core.kg_engine import KnowledgeGraphEngine

engine = KnowledgeGraphEngine('bolt://localhost:50002', 'neo4j', 'password')

# 查询所有部件单元
components = engine.query_by_entity_type("部件单元")
print(f"找到 {len(components)} 个部件单元")

# 查询所有故障状态
faults = engine.query_by_entity_type("故障状态")
print(f"找到 {len(faults)} 个故障状态")
```

### 2. 按关系类型查询
```python
# 查询所有部件故障关系
relations = engine.query_by_relation_type("部件故障")
print(f"找到 {len(relations)} 个部件故障关系")
```

### 3. 简单问答
```python
# 问答查询
results = engine.simple_qa("主轴电机故障怎么检测")
for result in results:
    print(f"实体: {result['name']}, 类型: {result['labels']}")
```

## 故障排除

### 1. 连接问题
确保Neo4j数据库正在运行，并且配置正确：
```bash
# 检查Neo4j状态
sudo systemctl status neo4j

# 或使用Docker
docker ps | grep neo4j
```

### 2. 权限问题
确保用户有足够的权限创建节点和关系。

### 3. 数据格式问题
检查JSON数据格式是否正确，每行必须是有效的JSON对象。

### 4. 中文编码问题
确保所有文件使用UTF-8编码。

## 扩展功能

你可以进一步扩展系统：

1. **添加更多实体类型**：修改 `EntityType` 枚举
2. **添加更多关系类型**：修改 `RelationType` 枚举  
3. **改进实体分类**：修改 `classify_entity_type` 函数
4. **添加向量化支持**：集成词向量或图嵌入
5. **添加推理规则**：实现基于规则的推理

## 注意事项

1. 数据库操作会修改数据，建议在测试环境中先验证
2. 大数据量构建时建议分批处理
3. 定期备份知识图谱数据
4. 监控数据库性能和内存使用情况