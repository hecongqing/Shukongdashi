#!/usr/bin/env python3
"""
知识图谱项目启动脚本
提供多种启动方式的便捷入口
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

def start_frontend():
    """启动前端界面"""
    print("🚀 启动知识图谱前端界面...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py", 
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

def start_services():
    """启动Docker服务"""
    print("🐳 启动Docker服务...")
    os.chdir(PROJECT_ROOT)
    subprocess.run(["docker-compose", "up", "-d"])
    print("✅ 服务启动完成！")
    print("📊 Neo4j Browser: http://localhost:7474")
    print("🌐 Streamlit App: http://localhost:8501")

def stop_services():
    """停止Docker服务"""
    print("🛑 停止Docker服务...")
    os.chdir(PROJECT_ROOT)
    subprocess.run(["docker-compose", "down"])
    print("✅ 服务已停止！")

def install_dependencies():
    """安装依赖"""
    print("📦 安装项目依赖...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ 依赖安装完成！")

def run_tests():
    """运行测试"""
    print("🧪 运行项目测试...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])

def create_venv():
    """创建虚拟环境"""
    print("🐍 创建Python虚拟环境...")
    os.chdir(PROJECT_ROOT)
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # 提示激活虚拟环境
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Linux/macOS
        activate_cmd = "source venv/bin/activate"
    
    print(f"✅ 虚拟环境创建完成！")
    print(f"💡 激活虚拟环境: {activate_cmd}")
    print(f"💡 然后运行: python start.py --install")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="知识图谱项目启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python start.py --frontend          # 启动前端界面
  python start.py --services          # 启动Docker服务
  python start.py --install           # 安装依赖
  python start.py --test              # 运行测试
  python start.py --venv              # 创建虚拟环境
  python start.py --stop              # 停止服务
        """
    )
    
    parser.add_argument("--frontend", action="store_true", 
                       help="启动Streamlit前端界面")
    parser.add_argument("--services", action="store_true", 
                       help="启动Docker服务")
    parser.add_argument("--stop", action="store_true", 
                       help="停止Docker服务")
    parser.add_argument("--install", action="store_true", 
                       help="安装项目依赖")
    parser.add_argument("--test", action="store_true", 
                       help="运行项目测试")
    parser.add_argument("--venv", action="store_true", 
                       help="创建Python虚拟环境")
    
    args = parser.parse_args()
    
    # 检查参数
    if not any(vars(args).values()):
        parser.print_help()
        print("\n🎯 快速开始:")
        print("1. python start.py --venv      # 创建虚拟环境")
        print("2. python start.py --install   # 安装依赖")
        print("3. python start.py --services  # 启动服务")
        print("4. python start.py --frontend  # 启动界面")
        return
    
    # 执行对应操作
    try:
        if args.venv:
            create_venv()
        elif args.install:
            install_dependencies()
        elif args.services:
            start_services()
        elif args.stop:
            stop_services()
        elif args.test:
            run_tests()
        elif args.frontend:
            start_frontend()
    except KeyboardInterrupt:
        print("\n⏹️  操作已取消")
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()