#!/bin/bash

# 设备故障知识图谱信息抽取系统启动脚本

echo "=========================================="
echo "设备故障知识图谱信息抽取系统"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "Python版本: $python_version"

# 创建必要的目录
echo "创建目录结构..."
mkdir -p data models/ner_model models/re_model results logs

# 检查依赖
echo "检查依赖..."
python3 -c "import torch, transformers, fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖..."
    pip install -r requirements_training.txt
fi

# 显示菜单
echo ""
echo "请选择要执行的操作:"
echo "1. 运行演示 (推荐新手)"
echo "2. 创建示例数据"
echo "3. 训练模型"
echo "4. 启动API服务"
echo "5. 运行系统测试"
echo "6. 查看使用示例"
echo "7. 退出"
echo ""

read -p "请输入选项 (1-7): " choice

case $choice in
    1)
        echo "运行演示..."
        python3 run_demo.py
        ;;
    2)
        echo "创建示例数据..."
        python3 train_models.py --create_sample
        ;;
    3)
        echo "训练模型..."
        echo "请确保您有训练数据文件 (data/train.json)"
        read -p "是否继续? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python3 train_models.py --train_file data/train.json
        fi
        ;;
    4)
        echo "启动API服务..."
        echo "服务将在 http://localhost:8000 启动"
        echo "API文档: http://localhost:8000/docs"
        echo "按 Ctrl+C 停止服务"
        python3 deploy.py
        ;;
    5)
        echo "运行系统测试..."
        python3 test_system.py
        ;;
    6)
        echo "查看使用示例..."
        python3 example_usage.py
        ;;
    7)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "操作完成！"
echo "更多信息请查看 README_TRAINING.md"