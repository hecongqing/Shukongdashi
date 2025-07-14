# 故障设备知识图谱信息抽取系统完整实现

## 项目概述

本项目为工业制造领域故障案例文本的信息抽取系统，专门支持竞赛任务要求的4种实体类型和4种关系类型。

## 实现的功能模块

### 1. 实体抽取模块 (`src/entity_extraction/`)

#### 支持的实体类型
- **部件单元** - 各种单元、零件、设备 (如："燃油泵"、"换流变压器"、"分离器")
- **性能表征** - 部件特征或性能描述 (如："压力"、"转速"、"温度") 
- **故障状态** - 故障状态描述 (如："漏油"、"断裂"、"变形"、"卡滞")
- **检测工具** - 检测故障的专用仪器 (如："零序互感器"、"保护器"、"漏电测试仪")

#### 核心文件
- `fault_ner_model.py` - 基于BERT的NER模型实现
- `train_ner.py` - 训练脚本
- `deploy_ner.py` - 部署和推理脚本

#### 技术特点
- 基于BERT-base-chinese预训练模型
- BIO标注方案
- 支持字符级别的实体识别
- 自动处理SPO数据格式转换
- 支持启发式实体类型推断

### 2. 关系抽取模块 (`src/relation_extraction/`)

#### 支持的关系类型
- **部件故障** - 部件单元 → 故障状态
- **性能故障** - 性能表征 → 故障状态  
- **检测工具** - 检测工具 → 性能表征
- **组成** - 部件单元 → 部件单元

#### 核心文件
- `fault_relation_model.py` - 基于BERT的关系分类模型
- `train_relation.py` - 训练脚本
- `deploy_relation.py` - 部署和推理脚本

#### 技术特点
- 基于BERT的关系分类
- 自动生成正负样本
- 支持实体对的关系预测
- 置信度评估
- 平衡正负样本比例

### 3. 完整流水线 (`src/pipeline.py`)

#### 功能特性
- 端到端信息抽取
- 集成实体识别和关系抽取
- 支持多种输出格式
- 批量处理能力
- 竞赛格式输出

#### 运行模式
- **API模式** - RESTful Web服务
- **CLI模式** - 命令行处理
- **提交模式** - 生成竞赛提交文件

### 4. 示例和文档

#### 文档
- `src/README.md` - 详细使用指南
- `FAULT_EXTRACTION_SUMMARY.md` - 项目总结
- API文档和示例

#### 示例代码
- `src/example.py` - 完整使用示例
- 示例数据和演示脚本

## 数据格式支持

### 输入格式 (训练数据)
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

### 输出格式 (预测结果)
```json
{
    "ID": "AT0001",
    "text": "发动机盖后部随着车速抖动",
    "entities": [
        {
            "text": "发动机盖",
            "type": "部件单元",
            "start": 0,
            "end": 4
        }
    ],
    "relations": [
        {
            "h": {"name": "发动机盖", "type": "部件单元", "pos": [0, 4]},
            "t": {"name": "抖动", "type": "故障状态", "pos": [9, 11]},
            "relation": "部件故障",
            "confidence": 0.95
        }
    ]
}
```

## 使用方法

### 1. 训练模型

```bash
# 训练实体识别模型
cd src/entity_extraction
python train_ner.py --train_data train.jsonl --output_dir ner_model

# 训练关系抽取模型  
cd ../relation_extraction
python train_relation.py --train_data train.jsonl --output_dir relation_model
```

### 2. 部署服务

```bash
# 启动完整流水线API服务
cd src
python pipeline.py \
    --ner_model ./ner_model \
    --relation_model ./relation_model \
    --mode api \
    --port 5002
```

### 3. 批量处理

```bash
# 处理测试文件并生成提交结果
python pipeline.py \
    --ner_model ./ner_model \
    --relation_model ./relation_model \
    --mode submission \
    --input_file test_data.jsonl \
    --output_file submission.jsonl
```

### 4. API调用示例

```bash
# 信息抽取
curl -X POST http://localhost:5002/extract \
    -H "Content-Type: application/json" \
    -d '{"text": "燃油泵损坏后，燃油将不能正常喷入发动机气缸"}'

# SPO格式输出
curl -X POST http://localhost:5002/extract_spo \
    -H "Content-Type: application/json" \
    -d '{"text": "燃油泵损坏后，燃油将不能正常喷入发动机气缸"}'
```

## 技术架构

### 模型架构
- **预训练模型**: BERT-base-chinese
- **实体识别**: Token级别分类 (BIO标注)
- **关系抽取**: 句子级别分类
- **优化器**: AdamW
- **学习率调度**: Linear warmup

### 性能优化
- GPU自动检测和使用
- 批处理支持
- 内存高效的数据加载
- 模型状态保存和恢复

### 工程特性
- 模块化设计
- 配置化参数
- 详细日志记录
- 异常处理和容错
- 多种部署模式

## 评估指标

### 实体识别
- 精确率 (Precision)
- 召回率 (Recall) 
- F1分数 (F1-Score)
- 支持类别

### 关系抽取
- 准确率 (Accuracy)
- 加权F1分数
- 混淆矩阵
- 置信度分布

## 扩展性

### 支持的扩展
1. **新实体类型**: 修改标签映射即可
2. **新关系类型**: 更新关系词典
3. **其他预训练模型**: 更换model_name参数
4. **多语言支持**: 使用对应语言的预训练模型

### 性能调优
1. **超参数优化**: 学习率、批次大小、序列长度
2. **数据增强**: 同义词替换、随机遮盖等
3. **模型集成**: 多模型投票或平均
4. **领域适应**: 在目标域数据上继续预训练

## 部署建议

### 生产环境
- 使用GPU服务器提高推理速度
- 配置负载均衡处理高并发
- 设置监控和告警机制
- 定期更新模型版本

### 容器化部署
```dockerfile
FROM python:3.8
RUN pip install torch transformers sklearn loguru flask
COPY src/ /app/
WORKDIR /app
CMD ["python", "pipeline.py", "--mode", "api"]
```

### 性能监控
- API响应时间
- 模型预测准确率
- 系统资源使用率
- 错误率和异常日志

## 总结

本系统为工业制造领域故障诊断提供了完整的信息抽取解决方案，具备以下优势：

1. **专业性强** - 针对故障诊断领域设计
2. **功能完整** - 包含实体识别和关系抽取
3. **易于使用** - 提供多种部署和调用方式
4. **扩展性好** - 支持新实体和关系类型
5. **工程化** - 具备生产级别的稳定性和性能

系统可直接用于故障知识图谱构建、智能检修和实时诊断等应用场景。