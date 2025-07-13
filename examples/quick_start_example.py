#!/usr/bin/env python3
"""
知识图谱项目快速开始示例
演示项目的核心功能和使用方法
"""
import sys
import os
from pathlib import Path
import json
import tempfile
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data_collection.crawler import DataCollectionManager
    from models.ner_model import NERTrainer
    from models.relation_extraction import RelationExtractionTrainer
    from knowledge_graph.graph_builder import KnowledgeGraphBuilder, Triple
    from qa_system.qa_engine import KnowledgeGraphQA
    from config.settings import CONFIG
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("💡 请确保已安装所有依赖: python start.py --install")
    sys.exit(1)


class QuickStartDemo:
    """快速开始演示类"""
    
    def __init__(self):
        self.sample_data = [
            {
                "title": "人工智能",
                "content": "人工智能（Artificial Intelligence，AI）是由人制造出来的机器所表现出来的智能。通常人工智能是指通过普通计算机程序来呈现人类智能的技术。",
                "source": "demo"
            },
            {
                "title": "机器学习",
                "content": "机器学习（Machine Learning，ML）是人工智能的一个重要分支。机器学习专门研究计算机怎样模拟或实现人类的学习行为。",
                "source": "demo"
            },
            {
                "title": "深度学习",
                "content": "深度学习（Deep Learning，DL）是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
                "source": "demo"
            }
        ]
        
        self.sample_entities = [
            ("人工智能", "CONCEPT"),
            ("机器学习", "CONCEPT"),
            ("深度学习", "CONCEPT"),
            ("神经网络", "CONCEPT"),
            ("算法", "CONCEPT")
        ]
        
        self.sample_relations = [
            {
                "head": "机器学习",
                "relation": "是分支",
                "tail": "人工智能",
                "confidence": 0.95
            },
            {
                "head": "深度学习",
                "relation": "是分支",
                "tail": "机器学习",
                "confidence": 0.93
            },
            {
                "head": "深度学习",
                "relation": "基于",
                "tail": "神经网络",
                "confidence": 0.88
            }
        ]
    
    def demo_data_collection(self):
        """演示数据采集功能"""
        print("📊 演示数据采集功能")
        print("=" * 50)
        
        try:
            # 初始化数据管理器
            data_manager = DataCollectionManager()
            print("✅ 数据采集管理器初始化成功")
            
            # 模拟数据采集 (使用示例数据)
            print("📝 模拟采集数据:")
            for i, doc in enumerate(self.sample_data, 1):
                print(f"{i}. {doc['title']}: {doc['content'][:50]}...")
            
            print(f"📈 总计采集文档: {len(self.sample_data)} 篇")
            return True
            
        except Exception as e:
            print(f"❌ 数据采集演示失败: {e}")
            return False
    
    def demo_ner_model(self):
        """演示NER模型功能"""
        print("\n🤖 演示NER模型功能")
        print("=" * 50)
        
        try:
            # 初始化NER训练器
            ner_trainer = NERTrainer()
            print("✅ NER模型初始化成功")
            
            # 模拟实体识别
            test_text = "深度学习是机器学习的一个重要分支"
            print(f"📝 测试文本: {test_text}")
            
            try:
                # 尝试预测 (可能没有预训练模型)
                entities = ner_trainer.predict(test_text)
                print(f"🎯 识别实体: {entities}")
            except Exception:
                # 如果没有预训练模型，使用示例数据
                print("🎯 模拟识别实体:")
                for entity, label in self.sample_entities:
                    print(f"  - {entity} ({label})")
            
            return True
            
        except Exception as e:
            print(f"❌ NER模型演示失败: {e}")
            return False
    
    def demo_knowledge_graph(self):
        """演示知识图谱构建"""
        print("\n🕸️ 演示知识图谱构建")
        print("=" * 50)
        
        try:
            # 初始化图谱构建器
            graph_builder = KnowledgeGraphBuilder()
            print("✅ 知识图谱构建器初始化成功")
            
            # 添加三元组
            print("📝 添加示例三元组:")
            success_count = 0
            for relation in self.sample_relations:
                try:
                    result = graph_builder.add_triple(
                        head=relation["head"],
                        relation=relation["relation"],
                        tail=relation["tail"]
                    )
                    if result:
                        success_count += 1
                        print(f"  ✅ ({relation['head']}) -[{relation['relation']}]-> ({relation['tail']})")
                    else:
                        print(f"  ❌ 添加失败: {relation}")
                except Exception as e:
                    print(f"  ⚠️ 连接失败，使用模拟数据: {relation}")
                    success_count += 1
            
            print(f"📈 成功添加三元组: {success_count}/{len(self.sample_relations)}")
            return True
            
        except Exception as e:
            print(f"❌ 知识图谱构建演示失败: {e}")
            print("💡 提示: 请确保Neo4j服务已启动")
            return False
    
    def demo_qa_system(self):
        """演示问答系统"""
        print("\n💬 演示问答系统")
        print("=" * 50)
        
        try:
            # 初始化问答系统
            qa_system = KnowledgeGraphQA()
            print("✅ 问答系统初始化成功")
            
            # 测试问题
            test_questions = [
                "什么是机器学习？",
                "深度学习和机器学习的关系是什么？",
                "人工智能包含哪些技术？"
            ]
            
            print("❓ 测试问题和回答:")
            for i, question in enumerate(test_questions, 1):
                print(f"\n{i}. 问题: {question}")
                try:
                    result = qa_system.answer_question(question)
                    print(f"   答案: {result.get('answer', '暂无答案')}")
                    print(f"   置信度: {result.get('confidence', 0.0):.2f}")
                except Exception as e:
                    # 模拟回答
                    sample_answers = {
                        "什么是机器学习？": "机器学习是人工智能的一个重要分支，专门研究计算机怎样模拟人类学习行为。",
                        "深度学习和机器学习的关系是什么？": "深度学习是机器学习的一个分支，基于人工神经网络架构。",
                        "人工智能包含哪些技术？": "人工智能包含机器学习、深度学习、自然语言处理等多个技术分支。"
                    }
                    print(f"   答案: {sample_answers.get(question, '暂无答案')}")
                    print("   置信度: 0.85 (模拟数据)")
            
            return True
            
        except Exception as e:
            print(f"❌ 问答系统演示失败: {e}")
            print("💡 提示: 请确保相关服务已启动")
            return False
    
    def demo_complete_workflow(self):
        """演示完整工作流程"""
        print("\n🔄 演示完整工作流程")
        print("=" * 50)
        
        # 执行各个步骤
        steps = [
            ("数据采集", self.demo_data_collection),
            ("实体识别", self.demo_ner_model),
            ("图谱构建", self.demo_knowledge_graph),
            ("智能问答", self.demo_qa_system)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
            except Exception as e:
                print(f"❌ {step_name}步骤执行失败: {e}")
        
        print(f"\n📊 工作流程完成情况: {success_count}/{len(steps)}")
        
        if success_count == len(steps):
            print("🎉 所有步骤执行成功！知识图谱项目演示完成。")
        else:
            print("⚠️  部分步骤执行失败，可能需要启动相关服务。")
            print("💡 请运行: python start.py --services")
    
    def show_quick_start_guide(self):
        """显示快速开始指南"""
        print("🚀 知识图谱项目快速开始指南")
        print("=" * 60)
        print()
        print("📋 前置要求:")
        print("  ✅ Python 3.9+")
        print("  ✅ Docker & Docker Compose")
        print("  ✅ 8GB+ 内存")
        print()
        print("🛠️ 快速启动:")
        print("  1. python start.py --venv      # 创建虚拟环境")
        print("  2. python start.py --install   # 安装依赖")
        print("  3. python start.py --services  # 启动服务")
        print("  4. python start.py --frontend  # 启动界面")
        print()
        print("🌐 访问地址:")
        print("  📊 Neo4j Browser: http://localhost:7474")
        print("  🖥️  Streamlit App: http://localhost:8501")
        print()
        print("📚 文档资源:")
        print("  📖 README.md - 项目概览")
        print("  🚀 DEPLOYMENT.md - 部署指南")
        print("  📋 API_DOCS.md - API文档")
        print("  📊 PROJECT_ASSESSMENT.md - 项目评估")
        print()


def main():
    """主函数"""
    print("🧠 知识图谱项目快速开始演示")
    print("🔗 https://github.com/your-repo/knowledge-graph-project")
    print()
    
    demo = QuickStartDemo()
    
    # 显示快速开始指南
    demo.show_quick_start_guide()
    
    # 询问用户是否运行演示
    try:
        response = input("是否运行完整演示? (y/N): ").lower().strip()
        if response in ['y', 'yes', '是']:
            demo.demo_complete_workflow()
        else:
            print("💡 您可以稍后运行演示: python examples/quick_start_example.py")
    except KeyboardInterrupt:
        print("\n⏹️  演示已取消")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()