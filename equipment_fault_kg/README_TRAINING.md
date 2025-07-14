# 设备故障知识图谱信息抽取系统

本项目实现了基于BERT的设备故障知识图谱信息抽取系统，包括实体抽取和关系抽取两个核心模块。

## 系统架构

```
equipment_fault_kg/
├── src/
│   ├── entity_extraction/          # 实体抽取模块
│   │   ├── data_processor.py       # 数据预处理
│   │   ├── trainer.py              # NER模型训练
│   │   └── ner_model.py            # NER模型定义
│   ├── relation_extraction/        # 关系抽取模块
│   │   ├── trainer.py              # RE模型训练
│   │   └── relation_extractor.py   # RE模型定义
│   └── deployment/                 # 部署模块
│       └── pipeline.py             # 端到端推理管道
├── config/                         # 配置文件
│   └── training_config.json        # 训练配置
├── data/                           # 数据目录
├── models/                         # 模型保存目录
├── results/                        # 评估结果
├── train_models.py                 # 训练脚本
├── evaluate.py                     # 评估脚本
└── deploy.py                       # 部署脚本
```

## 实体类型

系统支持4种实体类型的抽取：

1. **部件单元 (COMPONENT)**: 高端装备制造领域中的各种单元、零件、设备
   - 示例: "燃油泵"、"换流变压器"、"分离器"

2. **性能表征 (PERFORMANCE)**: 部件的特征或者性能描述
   - 示例: "压力"、"转速"、"温度"

3. **故障状态 (FAULT_STATE)**: 系统或部件的故障状态描述，多为故障类型
   - 示例: "漏油"、"断裂"、"变形"、"卡滞"

4. **检测工具 (DETECTION_TOOL)**: 用于检测某些故障的专用仪器
   - 示例: "零序互感器"、"保护器"、"漏电测试仪"

## 关系类型

系统支持4种关系类型的抽取：

1. **部件故障**: 部件单元 → 故障状态
   - 示例: 发动机盖 → 抖动

2. **性能故障**: 性能表征 → 故障状态
   - 示例: 液面 → 变低

3. **检测工具**: 检测工具 → 性能表征
   - 示例: 漏电测试仪 → 电流

4. **组成**: 部件单元 → 部件单元
   - 示例: 断路器 → 换流变压器

## 安装依赖

```bash
pip install torch transformers scikit-learn fastapi uvicorn tqdm numpy
```

## 数据格式

训练数据应为JSON格式，每行一个样本：

```json
{
    "ID": "AT0001",
    "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。故障原因简要分析:经技术人员试车；怀疑发动机盖锁或发动机盖铰链松旷。",
    "spo_list": [
        {
            "h": {"name": "发动机盖", "pos": [14, 18]},
            "t": {"name": "抖动", "pos": [24, 26]},
            "relation": "部件故障"
        }
    ]
}
```

## 快速开始

### 1. 创建示例数据

```bash
python train_models.py --create_sample
```

### 2. 训练模型

训练所有模型：
```bash
python train_models.py --train_file data/train.json
```

只训练NER模型：
```bash
python train_models.py --train_file data/train.json --ner_only
```

只训练RE模型：
```bash
python train_models.py --train_file data/train.json --re_only
```

### 3. 评估模型

```bash
python evaluate.py --test_file data/test.json
```

### 4. 部署服务

```bash
python deploy.py
```

服务将在 http://localhost:8000 启动，API文档在 http://localhost:8000/docs

## 详细使用说明

### 训练配置

可以通过修改 `config/training_config.json` 来调整训练参数：

```json
{
  "model_name": "bert-base-chinese",
  "batch_size": 16,
  "epochs": 10,
  "learning_rate": 2e-5,
  "warmup_steps": 500
}
```

### API使用

#### 单文本抽取

```python
import requests

url = "http://localhost:8000/extract"
data = {
    "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。"
}

response = requests.post(url, json=data)
result = response.json()

print("实体:", result["entities"])
print("关系:", result["relations"])
```

#### 批量抽取

```python
import requests

url = "http://localhost:8000/extract_batch"
texts = [
    "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
    "燃油泵损坏导致发动机无法启动。"
]

response = requests.post(url, json=texts)
results = response.json()["results"]
```

### 模型文件结构

训练完成后，模型文件将保存在以下目录：

```
models/
├── ner_model/
│   ├── ner_model.pth              # NER模型权重
│   ├── label_mapping.json         # 标签映射
│   ├── config.json                # 模型配置
│   └── vocab.txt                  # 词汇表
└── re_model/
    ├── re_model.pth               # RE模型权重
    ├── relation_mapping.json      # 关系映射
    ├── config.json                # 模型配置
    └── vocab.txt                  # 词汇表
```

## 性能优化

### 1. GPU加速

确保安装了CUDA版本的PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 模型优化

- 使用更小的batch_size如果GPU内存不足
- 调整max_length参数以适应文本长度
- 使用混合精度训练 (fp16: true)

### 3. 数据优化

- 确保训练数据质量
- 平衡各类实体和关系的样本数量
- 使用数据增强技术

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小max_length
   - 使用CPU训练 (device: "cpu")

2. **模型加载失败**
   - 检查模型文件路径
   - 确保模型文件完整
   - 检查依赖版本兼容性

3. **训练效果不佳**
   - 检查数据质量
   - 调整学习率
   - 增加训练轮数
   - 使用预训练模型微调

### 日志查看

训练过程中的日志会显示：
- 训练损失
- 验证指标
- 模型保存信息
- 错误信息

## 扩展功能

### 1. 添加新的实体类型

在 `data_processor.py` 中修改 `entity_types` 字典：

```python
self.entity_types = {
    "部件单元": "COMPONENT",
    "性能表征": "PERFORMANCE", 
    "故障状态": "FAULT_STATE",
    "检测工具": "DETECTION_TOOL",
    "新类型": "NEW_TYPE"  # 添加新类型
}
```

### 2. 添加新的关系类型

在 `trainer.py` 中修改 `relation2id` 字典：

```python
self.relation2id = {
    '部件故障': 0,
    '性能故障': 1,
    '检测工具': 2,
    '组成': 3,
    '新关系': 4,  # 添加新关系
    'no_relation': 5
}
```

### 3. 自定义模型

可以替换BERT模型为其他预训练模型：

```python
# 使用RoBERTa
trainer = NERTrainer(model_name="hfl/chinese-roberta-wwm-ext")

# 使用ALBERT
trainer = NERTrainer(model_name="hfl/chinese-albert-wwm-ext")
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。