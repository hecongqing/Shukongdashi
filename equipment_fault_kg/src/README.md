# 设备故障知识图谱 - 实体抽取与关系抽取系统

本项目实现了基于BERT的设备故障知识图谱信息抽取系统，包括实体抽取和关系抽取两个核心模块。

## 项目结构

```
equipment_fault_kg/src/
├── entity_extraction/          # 实体抽取模块
│   ├── data_processor.py       # 数据预处理
│   ├── train_ner.py           # 实体抽取训练
│   ├── deploy_ner.py          # 实体抽取部署
│   └── ner_model.py           # 实体抽取模型
├── relation_extraction/        # 关系抽取模块
│   ├── data_processor.py      # 数据预处理
│   ├── train_relation.py      # 关系抽取训练
│   ├── deploy_relation.py     # 关系抽取部署
│   └── relation_extractor.py  # 关系抽取器
├── config.py                  # 配置文件
├── train_models.py            # 训练主脚本
├── deploy_models.py           # 部署主脚本
└── README.md                  # 说明文档
```

## 功能特性

### 实体抽取
- 支持4种实体类型：部件单元、性能表征、故障状态、检测工具
- 基于BERT的序列标注模型
- 支持BIO标注方案
- 提供训练和部署接口

### 关系抽取
- 支持4种关系类型：部件故障、性能故障、检测工具、组成
- 基于BERT的分类模型
- 支持正负样本平衡
- 提供联合抽取功能

### 系统特性
- 完整的训练和部署流程
- RESTful API接口
- 支持批量处理
- 提供Web演示界面
- 详细的日志记录

## 安装依赖

```bash
pip install torch transformers scikit-learn flask tqdm numpy
```

## 快速开始

### 1. 训练模型

#### 使用示例数据训练
```bash
cd equipment_fault_kg/src
python train_models.py
```

#### 使用自定义数据训练
```bash
python train_models.py --data_path /path/to/your/data.json --output_dir ./models
```

#### 只创建示例数据
```bash
python train_models.py --create_sample
```

### 2. 部署服务

#### 启动统一API服务
```bash
python deploy_models.py --ner_model_path ./models/ner_models/best_ner_model.pth --relation_model_path ./models/relation_models/best_relation_model.pth --port 5000
```

#### 启动分离服务
```bash
python deploy_models.py --mode separate --ner_model_path ./models/ner_models/best_ner_model.pth --relation_model_path ./models/relation_models/best_relation_model.pth
```

### 3. 使用API

#### 实体抽取
```bash
curl -X POST http://localhost:5000/extract_entities \
  -H "Content-Type: application/json" \
  -d '{"text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。"}'
```

#### 关系抽取
```bash
curl -X POST http://localhost:5000/extract_relations \
  -H "Content-Type: application/json" \
  -d '{
    "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
    "entities": [
      {"name": "发动机盖", "type": "部件单元", "start_pos": 14, "end_pos": 18},
      {"name": "抖动", "type": "故障状态", "start_pos": 24, "end_pos": 26}
    ]
  }'
```

#### SPO三元组抽取
```bash
curl -X POST http://localhost:5000/extract_spo \
  -H "Content-Type: application/json" \
  -d '{"text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。"}'
```

### 4. Web演示

访问 `http://localhost:5000/demo` 查看Web演示界面。

## 数据格式

### 训练数据格式
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

### API响应格式

#### 实体抽取响应
```json
{
  "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
  "entities": [
    {
      "name": "发动机盖",
      "type": "部件单元",
      "start_pos": 14,
      "end_pos": 18
    },
    {
      "name": "抖动",
      "type": "故障状态",
      "start_pos": 24,
      "end_pos": 26
    }
  ]
}
```

#### 关系抽取响应
```json
{
  "text": "故障现象:车速到100迈以上发动机盖后部随着车速抖动。",
  "entities": [...],
  "relations": [
    {
      "head_entity": "发动机盖",
      "tail_entity": "抖动",
      "relation_type": "部件故障",
      "confidence": 0.95
    }
  ]
}
```

## 配置说明

### 模型配置
- `bert_model_name`: BERT模型名称，默认使用 `bert-base-chinese`
- `max_length`: 最大序列长度，默认512
- `batch_size`: 批次大小，默认16
- `epochs`: 训练轮数，默认10
- `learning_rate`: 学习率，默认2e-5

### 部署配置
- `host`: 服务主机地址，默认 `0.0.0.0`
- `port`: 服务端口，默认5000
- `debug`: 调试模式，默认False

## API接口说明

### 健康检查
- **GET** `/health`
- 返回服务状态信息

### 加载模型
- **POST** `/load_models`
- 请求体: `{"ner_model_path": "...", "relation_model_path": "..."}`

### 实体抽取
- **POST** `/extract_entities`
- 请求体: `{"text": "..."}`
- 返回抽取的实体列表

### 关系抽取
- **POST** `/extract_relations`
- 请求体: `{"text": "...", "entities": [...]}`
- 返回抽取的关系列表

### SPO三元组抽取
- **POST** `/extract_spo`
- 请求体: `{"text": "..."}`
- 返回完整的SPO三元组

### 批量处理
- **POST** `/extract_spo_batch`
- 请求体: `{"texts": ["...", "..."]}`
- 返回批量处理结果

## 实体类型

| 实体类型 | 说明 | 示例 |
|---------|------|------|
| 部件单元 | 高端装备制造领域中的各种单元、零件、设备 | "燃油泵"、"换流变压器"、"分离器" |
| 性能表征 | 部件的特征或者性能描述 | "压力"、"转速"、"温度" |
| 故障状态 | 系统或部件的故障状态描述，多为故障类型 | "漏油"、"断裂"、"变形"、"卡滞" |
| 检测工具 | 用于检测某些故障的专用仪器 | "零序互感器"、"保护器"、"漏电测试仪" |

## 关系类型

| 主体 | 客体 | 关系 | 主体示例 | 客体示例 |
|------|------|------|----------|----------|
| 部件单元 | 故障状态 | 部件故障 | 发动机盖 | 抖动 |
| 性能表征 | 故障状态 | 性能故障 | 液面 | 变低 |
| 检测工具 | 性能表征 | 检测工具 | 漏电测试仪 | 电流 |
| 部件单元 | 部件单元 | 组成 | 断路器 | 换流变压器 |

## 性能优化

### 训练优化
- 使用学习率调度器
- 支持梯度累积
- 支持混合精度训练
- 支持多GPU训练

### 推理优化
- 模型量化
- 批处理优化
- 缓存机制
- 异步处理

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确保模型文件完整
   - 检查CUDA版本兼容性

2. **内存不足**
   - 减小batch_size
   - 使用CPU推理
   - 启用梯度检查点

3. **训练效果不佳**
   - 增加训练数据
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 日志查看
```bash
# 查看训练日志
tail -f train.log

# 查看服务日志
tail -f service.log
```

## 扩展开发

### 添加新的实体类型
1. 在 `config.py` 中添加新的实体类型
2. 更新数据预处理逻辑
3. 重新训练模型

### 添加新的关系类型
1. 在 `config.py` 中添加新的关系类型
2. 更新关系抽取模型
3. 重新训练模型

### 自定义模型
1. 继承基础模型类
2. 实现自定义的前向传播逻辑
3. 更新训练脚本

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者