#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装备制造故障知识图谱项目启动脚本
提供完整的项目运行流程
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def run_command(command: str, cwd: str = None, check: bool = True) -> bool:
    """运行命令"""
    print(f"执行命令: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return False

def check_dependencies() -> bool:
    """检查依赖"""
    print("检查项目依赖...")
    
    required_packages = [
        "torch", "transformers", "neo4j", "fastapi", "uvicorn",
        "pandas", "numpy", "scikit-learn", "jieba", "loguru"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {missing_packages}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("依赖检查通过")
    return True

def setup_environment():
    """设置环境"""
    print("设置项目环境...")
    
    # 创建必要的目录
    directories = [
        "data", "logs", "models", "temp",
        "data/pdfs", "data/csv", "data/json"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("环境设置完成")

def start_neo4j():
    """启动Neo4j数据库"""
    print("启动Neo4j数据库...")
    
    # 检查Neo4j是否已运行
    if run_command("docker ps | grep neo4j", check=False):
        print("Neo4j已在运行")
        return True
    
    # 启动Neo4j容器
    neo4j_command = """
    docker run -d \
        --name neo4j \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        -e NEO4J_PLUGINS='["apoc"]' \
        neo4j:latest
    """
    
    if run_command(neo4j_command):
        print("Neo4j启动成功")
        return True
    else:
        print("Neo4j启动失败")
        return False

def start_mysql():
    """启动MySQL数据库"""
    print("启动MySQL数据库...")
    
    # 检查MySQL是否已运行
    if run_command("docker ps | grep mysql", check=False):
        print("MySQL已在运行")
        return True
    
    # 启动MySQL容器
    mysql_command = """
    docker run -d \
        --name mysql \
        -p 3306:3306 \
        -e MYSQL_ROOT_PASSWORD=password \
        -e MYSQL_DATABASE=equipment_fault \
        mysql:8.0
    """
    
    if run_command(mysql_command):
        print("MySQL启动成功")
        return True
    else:
        print("MySQL启动失败")
        return False

def run_data_collection():
    """运行数据采集"""
    print("开始数据采集...")
    
    # 运行网络爬虫
    if run_command("python 01_data_collection/web_scraping/main.py"):
        print("网络数据采集完成")
    else:
        print("网络数据采集失败")
    
    # 运行PDF处理
    if run_command("python 01_data_collection/pdf_processing/process_pdf.py"):
        print("PDF数据处理完成")
    else:
        print("PDF数据处理失败")

def run_information_extraction():
    """运行信息抽取"""
    print("开始信息抽取...")
    
    # 运行实体抽取训练
    if run_command("python 03_information_extraction/entity_extraction/train_entity_model.py"):
        print("实体抽取模型训练完成")
    else:
        print("实体抽取模型训练失败")
    
    # 运行大模型抽取
    if run_command("python 03_information_extraction/llm_extraction/llm_extractor.py"):
        print("大模型信息抽取完成")
    else:
        print("大模型信息抽取失败")

def build_knowledge_graph():
    """构建知识图谱"""
    print("开始构建知识图谱...")
    
    if run_command("python 04_knowledge_graph/neo4j_construction/build_graph.py"):
        print("知识图谱构建完成")
        return True
    else:
        print("知识图谱构建失败")
        return False

def start_llm_service():
    """启动大模型服务"""
    print("启动大模型服务...")
    
    # 设置环境变量
    os.environ["MODEL_NAME"] = "THUDM/chatglm2-6b"
    os.environ["DEVICE"] = "auto"
    os.environ["HOST"] = "0.0.0.0"
    os.environ["PORT"] = "8000"
    
    # 启动服务
    llm_command = "python 07_llm_deployment/model_serving/app.py"
    
    if run_command(llm_command, check=False):
        print("大模型服务启动成功")
        return True
    else:
        print("大模型服务启动失败")
        return False

def start_qa_system():
    """启动问答系统"""
    print("启动问答系统...")
    
    # 这里可以启动问答系统的Web界面
    print("问答系统启动完成")
    return True

def run_full_pipeline():
    """运行完整流程"""
    print("=" * 50)
    print("装备制造故障知识图谱项目 - 完整流程")
    print("=" * 50)
    
    # 1. 检查依赖
    if not check_dependencies():
        return False
    
    # 2. 设置环境
    setup_environment()
    
    # 3. 启动数据库
    if not start_neo4j():
        print("Neo4j启动失败，请检查Docker环境")
        return False
    
    if not start_mysql():
        print("MySQL启动失败，请检查Docker环境")
        return False
    
    # 4. 数据采集
    run_data_collection()
    
    # 5. 信息抽取
    run_information_extraction()
    
    # 6. 构建知识图谱
    if not build_knowledge_graph():
        print("知识图谱构建失败")
        return False
    
    # 7. 启动大模型服务
    if not start_llm_service():
        print("大模型服务启动失败")
        return False
    
    # 8. 启动问答系统
    start_qa_system()
    
    print("=" * 50)
    print("项目启动完成！")
    print("访问地址:")
    print("  - Neo4j浏览器: http://localhost:7474")
    print("  - 大模型API: http://localhost:8000")
    print("  - API文档: http://localhost:8000/docs")
    print("=" * 50)
    
    return True

def run_step(step: str):
    """运行指定步骤"""
    steps = {
        "deps": check_dependencies,
        "env": setup_environment,
        "neo4j": start_neo4j,
        "mysql": start_mysql,
        "collect": run_data_collection,
        "extract": run_information_extraction,
        "graph": build_knowledge_graph,
        "llm": start_llm_service,
        "qa": start_qa_system
    }
    
    if step in steps:
        print(f"运行步骤: {step}")
        steps[step]()
    else:
        print(f"未知步骤: {step}")
        print("可用步骤:", list(steps.keys()))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="装备制造故障知识图谱项目")
    parser.add_argument(
        "--mode", 
        choices=["full", "step"], 
        default="full",
        help="运行模式: full(完整流程) 或 step(单步运行)"
    )
    parser.add_argument(
        "--step",
        help="单步运行时指定的步骤"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="仅检查依赖"
    )
    parser.add_argument(
        "--setup-env",
        action="store_true",
        help="仅设置环境"
    )
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
    elif args.setup_env:
        setup_environment()
    elif args.mode == "full":
        run_full_pipeline()
    elif args.mode == "step":
        if args.step:
            run_step(args.step)
        else:
            print("单步运行模式需要指定 --step 参数")
            print("可用步骤: deps, env, neo4j, mysql, collect, extract, graph, llm, qa")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()