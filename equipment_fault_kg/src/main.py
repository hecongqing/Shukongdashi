"""
装备制造故障知识图谱构建项目 - 主程序

整合所有模块，提供完整的知识图谱构建和问答功能
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# 导入各个模块
from data_collection import WebCrawler, PDFExtractor, DataProcessor
from entity_extraction import NERModel
from llm_deployment import ModelLoader
from neo4j_qa import GraphManager


class EquipmentFaultKG:
    """装备制造故障知识图谱主类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # 初始化各个模块
        self.data_processor = DataProcessor(self.config['entity_extraction'])
        self.ner_model = NERModel(self.config['entity_extraction'])
        self.graph_manager = GraphManager(self.config['database']['neo4j'])
        
        # 大模型加载器（可选）
        self.llm_loader = None
        if self.config.get('llm'):
            self.llm_loader = ModelLoader(self.config['llm'])
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """设置日志"""
        log_config = self.config['logging']
        logger.add(
            log_config['file'],
            level=log_config['level'],
            format=log_config['format'],
            rotation="10 MB",
            retention="7 days"
        )
    
    def collect_data(self):
        """数据采集阶段"""
        logger.info("开始数据采集...")
        
        # 网络爬虫
        crawler = WebCrawler(self.config['data_collection'])
        
        # 示例：爬取故障案例（需要替换为实际的数据源）
        fault_cases = []
        for source in self.config['data_collection']['sources']:
            if source['type'] == 'web':
                cases = crawler.crawl_fault_cases(source['url'], max_pages=10)
                fault_cases.extend(cases)
        
        # PDF提取
        pdf_extractor = PDFExtractor()
        pdf_files = list(Path("data/raw").glob("*.pdf"))
        for pdf_file in pdf_files:
            pdf_data = pdf_extractor.extract_fault_manual(str(pdf_file))
            fault_cases.extend(pdf_data)
        
        logger.info(f"数据采集完成，共获取 {len(fault_cases)} 条数据")
        return fault_cases
    
    def process_data(self, raw_data: list):
        """数据处理阶段"""
        logger.info("开始数据处理...")
        
        # 批量处理数据
        processed_data = self.data_processor.process_batch(raw_data)
        
        # 保存处理后的数据
        self.data_processor.save_processed_data(
            processed_data, 
            "data/processed/processed_fault_cases.json"
        )
        
        # 生成统计信息
        stats = self.data_processor.generate_statistics(processed_data)
        logger.info(f"数据处理完成，统计信息: {stats}")
        
        return processed_data
    
    def extract_entities(self, processed_data: list):
        """实体抽取阶段"""
        logger.info("开始实体抽取...")
        
        all_entities = []
        all_relations = []
        
        for item in processed_data:
            text = f"{item['title']} {item['content']}"
            
            # 使用NER模型抽取实体
            entities = self.ner_model.predict(text)
            all_entities.extend(entities)
            
            # 使用大模型抽取关系（如果可用）
            if self.llm_loader:
                relations = self.llm_loader.extract_relations(text)
                all_relations.extend(relations.get('relations', []))
        
        logger.info(f"实体抽取完成，共抽取 {len(all_entities)} 个实体和 {len(all_relations)} 个关系")
        
        return all_entities, all_relations
    
    def build_knowledge_graph(self, entities: list, relations: list):
        """构建知识图谱"""
        logger.info("开始构建知识图谱...")
        
        # 构建Neo4j知识图谱
        self.graph_manager.build_knowledge_graph(entities, relations)
        
        # 获取统计信息
        stats = self.graph_manager.get_statistics()
        logger.info(f"知识图谱构建完成，统计信息: {stats}")
    
    def setup_qa_system(self):
        """设置问答系统"""
        logger.info("设置问答系统...")
        
        # 加载大模型（如果配置了）
        if self.llm_loader:
            try:
                self.llm_loader.load_model()
                logger.info("大模型加载成功")
            except Exception as e:
                logger.warning(f"大模型加载失败: {e}")
    
    def answer_question(self, question: str) -> str:
        """回答问题"""
        # 简单的问答逻辑
        if "故障" in question and "装备" in question:
            # 查询装备故障
            equipment_name = question.split("装备")[1].split("故障")[0] if "装备" in question and "故障" in question else ""
            if equipment_name:
                results = self.graph_manager.query_equipment_faults(equipment_name)
                if results:
                    answer = f"装备 {equipment_name} 的故障信息：\n"
                    for result in results:
                        answer += f"- 故障：{result['fault']}\n"
                        answer += f"  描述：{result['description']}\n"
                    return answer
        
        elif "原因" in question:
            # 查询故障原因
            fault_name = question.split("故障")[1].split("原因")[0] if "故障" in question and "原因" in question else ""
            if fault_name:
                results = self.graph_manager.query_fault_causes(fault_name)
                if results:
                    answer = f"故障 {fault_name} 的可能原因：\n"
                    for result in results:
                        answer += f"- {result['cause']}\n"
                        answer += f"  描述：{result['description']}\n"
                    return answer
        
        elif "解决" in question or "方案" in question:
            # 查询解决方案
            fault_name = question.split("故障")[1].split("解决")[0] if "故障" in question and "解决" in question else ""
            if fault_name:
                results = self.graph_manager.query_fault_solutions(fault_name)
                if results:
                    answer = f"故障 {fault_name} 的解决方案：\n"
                    for result in results:
                        answer += f"- {result['solution']}\n"
                        answer += f"  描述：{result['description']}\n"
                        answer += f"  难度：{result['difficulty']}\n"
                    return answer
        
        # 使用大模型回答（如果可用）
        if self.llm_loader:
            return self.llm_loader.answer_question(question)
        
        return "抱歉，我无法回答这个问题。请尝试询问装备故障、故障原因或解决方案相关的问题。"
    
    def run_pipeline(self):
        """运行完整的知识图谱构建流程"""
        logger.info("开始运行知识图谱构建流程...")
        
        try:
            # 1. 数据采集
            raw_data = self.collect_data()
            
            # 2. 数据处理
            processed_data = self.process_data(raw_data)
            
            # 3. 实体抽取
            entities, relations = self.extract_entities(processed_data)
            
            # 4. 构建知识图谱
            self.build_knowledge_graph(entities, relations)
            
            # 5. 设置问答系统
            self.setup_qa_system()
            
            logger.info("知识图谱构建流程完成！")
            
        except Exception as e:
            logger.error(f"流程执行失败: {e}")
            raise
    
    def interactive_qa(self):
        """交互式问答"""
        logger.info("启动交互式问答系统...")
        print("欢迎使用装备制造故障知识图谱问答系统！")
        print("您可以询问以下类型的问题：")
        print("- 某装备的故障信息")
        print("- 某故障的原因")
        print("- 某故障的解决方案")
        print("输入 'quit' 退出系统")
        
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用，再见！")
                    break
                
                if not question:
                    continue
                
                # 回答问题
                answer = self.answer_question(question)
                print(f"\n回答: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"发生错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="装备制造故障知识图谱构建系统")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=["build", "qa", "interactive"], default="interactive", 
                       help="运行模式：build(构建图谱), qa(问答), interactive(交互式)")
    parser.add_argument("--question", help="要回答的问题（qa模式使用）")
    
    args = parser.parse_args()
    
    # 创建项目实例
    kg_system = EquipmentFaultKG(args.config)
    
    if args.mode == "build":
        # 构建知识图谱
        kg_system.run_pipeline()
        
    elif args.mode == "qa":
        # 问答模式
        if args.question:
            answer = kg_system.answer_question(args.question)
            print(f"问题: {args.question}")
            print(f"回答: {answer}")
        else:
            print("请使用 --question 参数指定要回答的问题")
            
    elif args.mode == "interactive":
        # 交互式问答
        kg_system.interactive_qa()


if __name__ == "__main__":
    main()