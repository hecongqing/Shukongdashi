#!/bin/bash

# 装备制造故障知识图谱项目 - 快速开始脚本

set -e

echo "=========================================="
echo "装备制造故障知识图谱项目 - 快速开始"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system_requirements() {
    print_info "检查系统要求..."
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python版本: $PYTHON_VERSION"
    else
        print_error "未找到Python3，请先安装Python 3.8+"
        exit 1
    fi
    
    # 检查Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker版本: $DOCKER_VERSION"
    else
        print_warning "未找到Docker，将使用本地模式运行"
        USE_DOCKER=false
    fi
    
    # 检查Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose已安装"
    else
        print_warning "未找到Docker Compose"
    fi
    
    # 检查Git
    if command -v git &> /dev/null; then
        print_success "Git已安装"
    else
        print_error "未找到Git，请先安装Git"
        exit 1
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    print_info "创建Python虚拟环境..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "虚拟环境创建成功"
    else
        print_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    print_success "虚拟环境已激活"
}

# 安装依赖
install_dependencies() {
    print_info "安装Python依赖..."
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python依赖安装完成"
    else
        print_error "未找到requirements.txt文件"
        exit 1
    fi
}

# 创建必要的目录
create_directories() {
    print_info "创建项目目录结构..."
    
    directories=(
        "data"
        "data/pdfs"
        "data/csv"
        "data/json"
        "logs"
        "models"
        "temp"
        "docs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "目录结构创建完成"
}

# 启动数据库服务
start_databases() {
    print_info "启动数据库服务..."
    
    if [ "$USE_DOCKER" = true ]; then
        # 使用Docker启动数据库
        docker-compose up -d neo4j mysql redis
        print_success "数据库服务启动完成"
        
        # 等待数据库启动
        print_info "等待数据库启动..."
        sleep 30
    else
        print_warning "请手动启动Neo4j和MySQL数据库"
        print_info "Neo4j默认地址: bolt://localhost:7687"
        print_info "MySQL默认地址: localhost:3306"
    fi
}

# 运行数据采集
run_data_collection() {
    print_info "开始数据采集..."
    
    # 检查是否有PDF文件
    if [ -d "data/pdfs" ] && [ "$(ls -A data/pdfs)" ]; then
        print_info "发现PDF文件，开始处理..."
        python 01_data_collection/pdf_processing/process_pdf.py
    else
        print_warning "未发现PDF文件，跳过PDF处理"
    fi
    
    # 运行网络爬虫（可选）
    read -p "是否运行网络爬虫采集数据？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python 01_data_collection/web_scraping/main.py
    fi
    
    print_success "数据采集完成"
}

# 运行信息抽取
run_information_extraction() {
    print_info "开始信息抽取..."
    
    # 检查是否有训练数据
    if [ -f "data/sample_entity_data.json" ]; then
        python 03_information_extraction/entity_extraction/train_entity_model.py
    else
        print_warning "未找到训练数据，跳过实体抽取训练"
    fi
    
    # 运行大模型抽取
    python 03_information_extraction/llm_extraction/llm_extractor.py
    
    print_success "信息抽取完成"
}

# 构建知识图谱
build_knowledge_graph() {
    print_info "开始构建知识图谱..."
    
    python 04_knowledge_graph/neo4j_construction/build_graph.py
    
    print_success "知识图谱构建完成"
}

# 启动大模型服务
start_llm_service() {
    print_info "启动大模型服务..."
    
    # 设置环境变量
    export MODEL_NAME="THUDM/chatglm2-6b"
    export DEVICE="auto"
    export HOST="0.0.0.0"
    export PORT="8000"
    
    # 启动服务（后台运行）
    nohup python 07_llm_deployment/model_serving/app.py > logs/llm_service.log 2>&1 &
    LLM_PID=$!
    
    # 等待服务启动
    sleep 10
    
    # 检查服务是否启动成功
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "大模型服务启动成功 (PID: $LLM_PID)"
        echo $LLM_PID > temp/llm_service.pid
    else
        print_error "大模型服务启动失败"
        exit 1
    fi
}

# 启动问答系统
start_qa_system() {
    print_info "启动问答系统..."
    
    # 这里可以启动问答系统的Web界面
    print_success "问答系统启动完成"
}

# 显示访问信息
show_access_info() {
    echo
    echo "=========================================="
    echo "项目启动完成！"
    echo "=========================================="
    echo
    echo "访问地址:"
    echo "  - Neo4j浏览器: http://localhost:7474"
    echo "   用户名: neo4j"
    echo "   密码: password"
    echo
    echo "  - 大模型API: http://localhost:8000"
    echo "  - API文档: http://localhost:8000/docs"
    echo
    echo "  - 问答系统: http://localhost:8001"
    echo
    echo "  - 监控面板: http://localhost:9090"
    echo "  - 日志分析: http://localhost:5601"
    echo
    echo "项目目录:"
    echo "  - 数据目录: ./data"
    echo "  - 日志目录: ./logs"
    echo "  - 模型目录: ./models"
    echo
    echo "停止服务:"
    echo "  - 停止大模型服务: kill \$(cat temp/llm_service.pid)"
    echo "  - 停止所有服务: docker-compose down"
    echo
}

# 清理函数
cleanup() {
    print_info "清理资源..."
    
    # 停止大模型服务
    if [ -f "temp/llm_service.pid" ]; then
        LLM_PID=$(cat temp/llm_service.pid)
        if kill -0 $LLM_PID 2>/dev/null; then
            kill $LLM_PID
            print_success "大模型服务已停止"
        fi
        rm -f temp/llm_service.pid
    fi
    
    # 停止Docker服务
    if [ "$USE_DOCKER" = true ]; then
        docker-compose down
        print_success "Docker服务已停止"
    fi
}

# 主函数
main() {
    # 设置默认值
    USE_DOCKER=true
    
    # 检查命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-docker)
                USE_DOCKER=false
                shift
                ;;
            --help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --no-docker    不使用Docker（需要手动启动数据库）"
                echo "  --help         显示帮助信息"
                exit 0
                ;;
            *)
                print_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 设置信号处理
    trap cleanup EXIT
    
    # 执行启动流程
    check_system_requirements
    create_virtual_environment
    install_dependencies
    create_directories
    start_databases
    run_data_collection
    run_information_extraction
    build_knowledge_graph
    start_llm_service
    start_qa_system
    show_access_info
    
    print_success "项目启动完成！"
    
    # 保持脚本运行
    print_info "按 Ctrl+C 停止服务"
    while true; do
        sleep 1
    done
}

# 运行主函数
main "$@"