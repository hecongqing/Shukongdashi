#!/bin/bash

# 设备故障知识图谱 - 启动脚本
# 用于快速启动实体抽取和关系抽取系统

set -e

echo "=========================================="
echo "设备故障知识图谱 - 实体抽取与关系抽取系统"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import torch, transformers, sklearn, flask" 2>/dev/null || {
    echo "安装依赖..."
    pip3 install -r requirements.txt
}

# 创建必要的目录
echo "创建目录结构..."
mkdir -p models/ner_models models/relation_models data/processed

# 检查模型文件
NER_MODEL="./models/ner_models/best_ner_model.pth"
RELATION_MODEL="./models/relation_models/best_relation_model.pth"

if [ ! -f "$NER_MODEL" ] || [ ! -f "$RELATION_MODEL" ]; then
    echo "模型文件不存在，开始训练模型..."
    echo "注意: 训练过程可能需要较长时间，请耐心等待"
    python3 train_models.py
fi

# 启动API服务
echo "启动API服务..."
echo "服务地址: http://localhost:5000"
echo "演示页面: http://localhost:5000/demo"
echo "按 Ctrl+C 停止服务"
echo ""

python3 deploy_models.py \
    --ner_model_path "$NER_MODEL" \
    --relation_model_path "$RELATION_MODEL" \
    --port 5000