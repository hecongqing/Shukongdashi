# 故障设备知识图谱信息抽取系统

本系统专门针对工业制造领域的故障案例文本，实现实体抽取和关系抽取功能，支持4种实体类型和4种关系类型。

## 支持的实体类型

1. **部件单元** - 高端装备制造领域中的各种单元、零件、设备
2. **性能表征** - 部件的特征或者性能描述  
3. **故障状态** - 系统或部件的故障状态描述，多为故障类型
4. **检测工具** - 用于检测某些故障的专用仪器

## 支持的关系类型

1. **部件故障** - 部件单元 → 故障状态
2. **性能故障** - 性能表征 → 故障状态
3. **检测工具** - 检测工具 → 性能表征
4. **组成** - 部件单元 → 部件单元

## 目录结构

```
src/
├── entity_extraction/          # 实体抽取模块
│   ├── fault_ner_model.py     # NER模型实现
│   ├── train_ner.py           # 训练脚本
│   └── deploy_ner.py          # 部署脚本
├── relation_extraction/        # 关系抽取模块
│   ├── fault_relation_model.py # 关系抽取模型实现
│   ├── train_relation.py      # 训练脚本
│   └── deploy_relation.py     # 部署脚本
├── pipeline.py                 # 完整流水线
└── README.md                   # 说明文档
```

## 环境要求

```bash
pip install torch transformers sklearn loguru flask tqdm numpy
```

## 使用指南

### 1. 准备数据

数据格式应为JSON Lines，每行一个样本：

```json
{
    "ID": "AT0001",
    "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
    "spo_list": [
        {
            "h": {"name": "发动机盖", "pos": [14, 18]},
            "t": {"name": "抖动", "pos": [24, 26]},
            "relation": "部件故障"
        }
    ]
}
```

### 2. 训练实体抽取模型

```bash
cd entity_extraction

# 训练NER模型
python train_ner.py \
    --train_data /path/to/train_data.jsonl \
    --output_dir /path/to/ner_model \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5
```

### 3. 训练关系抽取模型

```bash
cd relation_extraction

# 训练关系抽取模型
python train_relation.py \
    --train_data /path/to/train_data.jsonl \
    --output_dir /path/to/relation_model \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5
```

### 4. 部署和推理

#### 4.1 使用完整流水线（推荐）

```bash
# API模式 - 启动Web服务
python pipeline.py \
    --ner_model /path/to/ner_model \
    --relation_model /path/to/relation_model \
    --mode api \
    --port 5002

# CLI模式 - 处理单个文本
python pipeline.py \
    --ner_model /path/to/ner_model \
    --relation_model /path/to/relation_model \
    --mode cli \
    --text "燃油泵损坏后，燃油将不能正常喷入发动机气缸"

# 批量处理文件
python pipeline.py \
    --ner_model /path/to/ner_model \
    --relation_model /path/to/relation_model \
    --mode cli \
    --input_file /path/to/test_data.jsonl \
    --output_file /path/to/results.jsonl

# 生成竞赛提交文件
python pipeline.py \
    --ner_model /path/to/ner_model \
    --relation_model /path/to/relation_model \
    --mode submission \
    --input_file /path/to/test_data.jsonl \
    --output_file /path/to/submission.jsonl
```

#### 4.2 单独部署实体抽取

```bash
cd entity_extraction

# API模式
python deploy_ner.py \
    --model_dir /path/to/ner_model \
    --mode api \
    --port 5000

# CLI模式
python deploy_ner.py \
    --model_dir /path/to/ner_model \
    --mode cli \
    --text "发动机盖后部随着车速抖动"

# 批量处理
python deploy_ner.py \
    --model_dir /path/to/ner_model \
    --mode cli \
    --input_file /path/to/test_data.jsonl \
    --output_file /path/to/ner_results.jsonl
```

#### 4.3 单独部署关系抽取

```bash
cd relation_extraction

# API模式
python deploy_relation.py \
    --model_dir /path/to/relation_model \
    --mode api \
    --port 5001

# CLI模式（需要提供实体）
python deploy_relation.py \
    --model_dir /path/to/relation_model \
    --mode cli \
    --text "发动机盖后部随着车速抖动" \
    --entities '[{"text":"发动机盖","type":"部件单元","start":0,"end":4}]'
```

### 5. API接口

#### 5.1 完整流水线API

启动服务后，可以通过以下接口使用：

**信息抽取接口**
```bash
curl -X POST http://localhost:5002/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "燃油泵损坏后，燃油将不能正常喷入发动机气缸"}'
```

**批量抽取接口**
```bash
curl -X POST http://localhost:5002/batch_extract \
    -H "Content-Type: application/json" \
    -d '{"texts": ["文本1", "文本2"]}'
```

**SPO格式抽取接口**
```bash
curl -X POST http://localhost:5002/extract_spo \
    -H "Content-Type: application/json" \
    -d '{"text": "燃油泵损坏后，燃油将不能正常喷入发动机气缸"}'
```

#### 5.2 实体抽取API

```bash
curl -X POST http://localhost:5000/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "发动机盖后部随着车速抖动"}'
```

#### 5.3 关系抽取API

```bash
curl -X POST http://localhost:5001/extract_relations \
    -H "Content-Type: application/json" \
    -d '{
        "text": "发动机盖后部随着车速抖动",
        "entities": [
            {"text":"发动机盖","type":"部件单元","start":0,"end":4},
            {"text":"抖动","type":"故障状态","start":9,"end":11}
        ]
    }'
```

## 输出格式

### 实体抽取结果

```json
{
    "text": "发动机盖后部随着车速抖动",
    "entities": [
        {
            "text": "发动机盖",
            "type": "部件单元",
            "start": 0,
            "end": 4
        },
        {
            "text": "抖动",
            "type": "故障状态", 
            "start": 9,
            "end": 11
        }
    ],
    "count": 2
}
```

### 关系抽取结果

```json
{
    "text": "发动机盖后部随着车速抖动",
    "entities": [...],
    "relations": [
        {
            "h": {
                "name": "发动机盖",
                "type": "部件单元",
                "pos": [0, 4]
            },
            "t": {
                "name": "抖动", 
                "type": "故障状态",
                "pos": [9, 11]
            },
            "relation": "部件故障",
            "confidence": 0.95
        }
    ],
    "relation_count": 1
}
```

### SPO格式结果（竞赛格式）

```json
{
    "ID": "AT0001",
    "text": "发动机盖后部随着车速抖动",
    "spo_list": [
        {
            "h": {
                "name": "发动机盖",
                "pos": [0, 4]
            },
            "t": {
                "name": "抖动",
                "pos": [9, 11]
            },
            "relation": "部件故障"
        }
    ]
}
```

## 模型配置

### 默认配置

- 预训练模型：`bert-base-chinese`
- 最大序列长度：512
- 批次大小：16
- 学习率：2e-5
- 训练轮数：10

### 自定义配置

可以通过命令行参数调整配置：

```bash
python train_ner.py \
    --model_name bert-base-chinese \
    --max_length 512 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --epochs 15 \
    --dropout 0.2
```

## 性能优化建议

1. **GPU加速**：如果有GPU可用，模型会自动使用CUDA
2. **批处理**：使用批处理API提高处理效率
3. **模型量化**：可以考虑模型量化减少内存占用
4. **缓存结果**：对于重复文本可以缓存抽取结果

## 故障排除

### 常见问题

1. **内存不足**：减少batch_size或max_length
2. **模型加载失败**：检查模型路径和文件完整性
3. **中文编码问题**：确保文件使用UTF-8编码
4. **API响应慢**：考虑使用GPU或减少序列长度

### 日志调试

系统使用loguru进行日志记录，可以通过以下方式查看详细日志：

```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题请联系项目维护者。