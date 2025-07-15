# NER 预测问题修复报告

## 问题描述

原始的 `deploy_ner.py` 预测结果存在以下问题：
- 实体边界划分错误
- 实体分类不准确
- 过度分割，产生很多无意义的短实体

例如原始错误预测：
```
- 障现象:车 (FAULT_STATE) at position 1-6
- 速到 (PERFORMANCE) at position 6-8
- 100迈以上发 (FAULT_STATE) at position 8-15
```

## 根本原因分析

### 1. 字符到Token映射错误

**问题**：原始代码中 `char_to_token` 映射逻辑有误
```python
# 错误的映射方式
for _ in range(len(sub_tokens)):
    char_to_token.append(len(tokens) - 1)
```

**问题分析**：这会将一个字符的所有sub-token都映射到最后一个token位置，导致位置错乱。

### 2. 训练和预测不一致

**问题**：训练时(`train_ner.py`)和预测时(`deploy_ner.py`)的tokenization方式不一致。

- 训练时：按字符逐个处理，正确处理sub-token标签对齐
- 预测时：字符映射逻辑错误，没有考虑[CLS]token偏移

### 3. 实体提取逻辑缺陷

**问题**：`_extract_entities` 方法没有正确处理：
- Token索引偏移（[CLS]token）
- 实体类型一致性检查
- 边界条件处理

## 修复方案

### 1. 修复字符到Token映射

```python
# 修复后的映射方式
for i, char in enumerate(text):
    char_start_token = len(tokens)  # 记录起始位置
    sub_tokens = self.tokenizer.tokenize(char)
    if not sub_tokens:
        sub_tokens = ['[UNK]']
    tokens.extend(sub_tokens)
    char_to_token.append(char_start_token)  # 使用起始位置
```

### 2. 统一训练和预测的处理方式

- 按字符逐个tokenization
- 保持与训练时完全一致的处理流程
- 正确处理padding和截断

### 3. 改进实体提取逻辑

```python
def _extract_entities(self, text: str, pred_labels: List[int], char_to_token: List[int]) -> List[Dict]:
    # 正确处理[CLS]token偏移
    token_idx = char_to_token[i] + 1  # +1 因为第一个token是[CLS]
    
    # 添加类型一致性检查
    if entity_type == current_entity['type']:
        current_entity['name'] += char
        current_entity['end_pos'] = i + 1
    else:
        # 类型不匹配，结束当前实体
        entities.append(current_entity)
        current_entity = None
```

### 4. 修复模型加载逻辑

```python
# 正确获取标签数量
num_labels = len(self.label2id)  # 而不是从checkpoint读取错误的值
```

## 修复效果预期

修复后的NER预测应该能够：

1. **正确的实体边界**：完整识别词汇和短语，而不是过度分割
2. **准确的实体分类**：正确区分COMPONENT、PERFORMANCE、FAULT_STATE、DETECTION_TOOL
3. **连贯的实体识别**：相邻的相同类型token能正确合并为完整实体

例如期望的正确预测：
```
- 故障现象 (FAULT_STATE) at position 0-4
- 车速 (PERFORMANCE) at position 5-7
- 发动机盖 (COMPONENT) at position 12-15
- 抖动 (FAULT_STATE) at position 19-21
- 发动机盖锁 (COMPONENT) at position 35-39
- 发动机盖铰链 (COMPONENT) at position 41-46
- 松旷 (FAULT_STATE) at position 47-49
```

## 测试建议

1. 使用提供的测试代码验证tokenization逻辑
2. 对比修复前后的预测结果
3. 在更多样本上测试模型性能
4. 检查实体边界和分类的准确性

## 注意事项

- 需要确保有训练好的模型文件才能测试预测功能
- 建议在验证集上重新评估模型性能
- 如果问题仍然存在，可能需要重新训练模型