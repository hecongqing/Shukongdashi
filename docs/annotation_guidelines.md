# 装备制造故障知识图谱 —— 标注指南

版本：v0.1  
目标受众：标注员 / 质检员

---

## 1. 任务概述

在给定的维修手册、工单、论坛帖子中，对**实体**和**关系**进行标注，用于后续信息抽取模型训练。

- **实体**：8 类（Equipment, Component, Symptom, FaultCode, Cause, Action, Parameter, Material）
- **关系**：12 类（见下表）

> 建议：一句话 ≤ 3 个三元组；如句子过长，可拆分。

## 2. 实体类型

| 名称 | 描述 | 示例 |
|------|------|------|
| Equipment | 整机、产线 | “CNC 6136 车床” |
| Component | 子部件 / 机构 | “主轴电机”、“滚珠丝杠” |
| Symptom | 故障现象 / 表现 | “振动异常”、“温升过高” |
| FaultCode | 报警/错误码 | “E60”、“ALM-012” |
| Cause | 根本原因 | “轴承磨损”、“冷却液不足” |
| Action | 维修/处置措施 | “更换轴承”、“调整间隙” |
| Parameter | 过程/诊断参数 | “振动值 4.5 mm/s”、“温度 85 ℃” |
| Material | 润滑油、清洗剂等 | “46# 润滑油” |

## 3. 关系类型

| 关系名称 | 方向 (Head → Tail) | 解释 | 例子 |
|-----------|-------------------|-------|------|
| has_component | Equipment → Component | 设备包含部件 | 车床 has_component 主轴 |
| part_of | Component → Equipment | 部件属于设备 | 主轴 part_of 车床 |
| exhibits_symptom | Equipment/Component → Symptom | 出现故障现象 | 主轴 exhibits_symptom 振动异常 |
| has_fault_code | Equipment/Component → FaultCode | 产生报警码 | 车床 has_fault_code E60 |
| caused_by | Symptom/FaultCode → Cause | 由…导致 | 振动异常 caused_by 轴承磨损 |
| leads_to | Cause → Symptom | 原因产生现象 | 轴承磨损 leads_to 振动异常 |
| resolved_by | Cause/Fault → Action | 通过…解决 | 轴承磨损 resolved_by 更换轴承 |
| action_on | Action → Component | 维修作用对象 | 更换轴承 action_on 主轴 |
| affects_parameter | Cause/Fault → Parameter | 影响了参数 | 轴承磨损 affects_parameter 振动值 |
| parameter_of | Parameter → Equipment | 参数属于对象 | 振动值 parameter_of 主轴 |
| uses_material | Component → Material | 运行/维修所需材料 | 主轴 uses_material 润滑油 |
| related_to | 任意 → 任意 | 其他非上述语义的相关 | 冷却液 related_to 温度 |

## 4. 标注规范

1. **实体边界**：选择最小、完整的技术名词，不含冗余形容词。  
   - ✅ “主轴电机”  ❌ “整台主轴电机装置”
2. **别名/缩写**：按上下文原文标注，不统一替换。  
3. **数字参数**：数值+单位整体标注为 Parameter。  
4. **跨句关系**：暂不标注，后期由模型推理。  
5. **歧义处理**：无法确定时可标注为 *相关* (related_to)。

## 5. 工具使用

### 5.1 Doccano

1. 访问 `http://localhost:8001`，使用 admin / doccano123 登录。  
2. 在 *Projects* → *Create Project*：
   - Type: *Sequence labeling with relation*  
   - Name: `Equipment Fault KG`  
3. 在 *Labels* 导入 `configs/ontology_schema.json` 中的 entities 和 relations：

```bash
python scripts/annotation/setup_doccano.py --url http://localhost:8001 --username admin --password doccano123 \
      --schema configs/ontology_schema.json
```

### 5.2 Label-Studio

1. 访问 `http://localhost:8080`，使用 admin / ls123456 登录。  
2. New Project → Import XML template `configs/labelstudio_template.xml` (已预置)。

## 6. 质检流程

- **Self-QA**：标注完一批后自查实体类别、关系方向。  
- **Peer-Review**：每 200 句互审一次，纠错率 < 5%。  
- **QA Sheet**：记录常见误标案例，定期同步。

---

如有疑问，请在飞书群 #KG-标注 咨询。