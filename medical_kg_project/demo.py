"""
医疗知识图谱快速演示脚本
"""
import json
from src.kg_builder.neo4j_manager import MedicalGraphBuilder, GraphConfig, MedicalGraphAnalyzer
from src.models.llm_extractor import MedicalLLMExtractor
from src.qa_system.medical_qa import MedicalQASystem
from loguru import logger


def create_sample_data():
    """创建示例数据"""
    nodes = [
        # 疾病节点
        {"type": "disease", "properties": {"name": "糖尿病", "icd10": "E11", "description": "一种慢性代谢性疾病，特征是血糖水平持续偏高"}},
        {"type": "disease", "properties": {"name": "高血压", "icd10": "I10", "description": "动脉血压持续升高的慢性疾病"}},
        {"type": "disease", "properties": {"name": "冠心病", "icd10": "I25", "description": "冠状动脉粥样硬化性心脏病"}},
        
        # 症状节点
        {"type": "symptom", "properties": {"name": "多饮", "description": "饮水量明显增加"}},
        {"type": "symptom", "properties": {"name": "多尿", "description": "排尿次数和尿量增加"}},
        {"type": "symptom", "properties": {"name": "多食", "description": "食欲亢进，进食量增加"}},
        {"type": "symptom", "properties": {"name": "头晕", "description": "头部眩晕感"}},
        {"type": "symptom", "properties": {"name": "胸闷", "description": "胸部压迫感或不适"}},
        {"type": "symptom", "properties": {"name": "心悸", "description": "心跳感觉异常"}},
        
        # 药物节点
        {"type": "drug", "properties": {"name": "二甲双胍", "type": "口服降糖药", "usage": "口服，每日2-3次"}},
        {"type": "drug", "properties": {"name": "胰岛素", "type": "激素类药物", "usage": "皮下注射"}},
        {"type": "drug", "properties": {"name": "阿司匹林", "type": "抗血小板药物", "usage": "口服，每日1次"}},
        {"type": "drug", "properties": {"name": "硝苯地平", "type": "钙通道阻滞剂", "usage": "口服，每日1-2次"}},
        
        # 检查项目节点
        {"type": "examination", "properties": {"name": "血糖检查", "description": "检测血液中葡萄糖浓度"}},
        {"type": "examination", "properties": {"name": "血压测量", "description": "测量动脉血压"}},
        {"type": "examination", "properties": {"name": "心电图", "description": "记录心脏电活动"}},
        {"type": "examination", "properties": {"name": "冠脉造影", "description": "冠状动脉造影检查"}},
        
        # 治疗方法节点
        {"type": "treatment", "properties": {"name": "饮食控制", "description": "通过合理饮食控制病情"}},
        {"type": "treatment", "properties": {"name": "运动疗法", "description": "通过适当运动改善病情"}},
        {"type": "treatment", "properties": {"name": "药物治疗", "description": "使用药物控制病情"}},
        
        # 科室节点
        {"type": "department", "properties": {"name": "内分泌科", "description": "治疗内分泌系统疾病"}},
        {"type": "department", "properties": {"name": "心内科", "description": "治疗心血管系统疾病"}},
    ]
    
    relationships = [
        # 糖尿病相关关系
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "symptom", "target_name": "多饮", "type": "has_symptom", "properties": {"probability": 0.8}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "symptom", "target_name": "多尿", "type": "has_symptom", "properties": {"probability": 0.8}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "symptom", "target_name": "多食", "type": "has_symptom", "properties": {"probability": 0.7}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "drug", "target_name": "二甲双胍", "type": "treated_by", "properties": {"effectiveness": "高"}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "drug", "target_name": "胰岛素", "type": "treated_by", "properties": {"effectiveness": "高"}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "examination", "target_name": "血糖检查", "type": "examined_by", "properties": {"frequency": "常规"}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "treatment", "target_name": "饮食控制", "type": "treated_by", "properties": {"importance": "重要"}},
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "department", "target_name": "内分泌科", "type": "belongs_to", "properties": {}},
        
        # 高血压相关关系
        {"source_type": "disease", "source_name": "高血压", "target_type": "symptom", "target_name": "头晕", "type": "has_symptom", "properties": {"probability": 0.6}},
        {"source_type": "disease", "source_name": "高血压", "target_type": "symptom", "target_name": "心悸", "type": "has_symptom", "properties": {"probability": 0.5}},
        {"source_type": "disease", "source_name": "高血压", "target_type": "drug", "target_name": "硝苯地平", "type": "treated_by", "properties": {"effectiveness": "高"}},
        {"source_type": "disease", "source_name": "高血压", "target_type": "examination", "target_name": "血压测量", "type": "examined_by", "properties": {"frequency": "常规"}},
        {"source_type": "disease", "source_name": "高血压", "target_type": "department", "target_name": "心内科", "type": "belongs_to", "properties": {}},
        
        # 冠心病相关关系
        {"source_type": "disease", "source_name": "冠心病", "target_type": "symptom", "target_name": "胸闷", "type": "has_symptom", "properties": {"probability": 0.8}},
        {"source_type": "disease", "source_name": "冠心病", "target_type": "symptom", "target_name": "心悸", "type": "has_symptom", "properties": {"probability": 0.7}},
        {"source_type": "disease", "source_name": "冠心病", "target_type": "drug", "target_name": "阿司匹林", "type": "treated_by", "properties": {"effectiveness": "高"}},
        {"source_type": "disease", "source_name": "冠心病", "target_type": "examination", "target_name": "心电图", "type": "examined_by", "properties": {"frequency": "常规"}},
        {"source_type": "disease", "source_name": "冠心病", "target_type": "examination", "target_name": "冠脉造影", "type": "examined_by", "properties": {"frequency": "必要时"}},
        {"source_type": "disease", "source_name": "冠心病", "target_type": "department", "target_name": "心内科", "type": "belongs_to", "properties": {}},
        
        # 并发症关系
        {"source_type": "disease", "source_name": "糖尿病", "target_type": "disease", "target_name": "冠心病", "type": "complication_of", "properties": {"risk": "中等"}},
        {"source_type": "disease", "source_name": "高血压", "target_type": "disease", "target_name": "冠心病", "type": "complication_of", "properties": {"risk": "高"}},
    ]
    
    return {"nodes": nodes, "relationships": relationships}


def demo_knowledge_extraction():
    """演示知识抽取"""
    print("\n=== 知识抽取演示 ===")
    
    # 初始化抽取器（这里使用模拟，实际使用需要加载模型）
    print("注意：实际使用需要下载并加载大模型，这里仅作演示")
    
    # 示例文本
    test_text = """
    患者王某，男，55岁，因"反复胸闷、胸痛2年，加重1周"入院。
    患者2年前开始出现活动后胸闷、胸痛，休息后可缓解。
    既往有高血压病史10年，糖尿病病史5年。
    入院后完善心电图、心脏彩超等检查，诊断为冠心病、不稳定型心绞痛。
    给予阿司匹林、氯吡格雷双抗治疗，美托洛尔控制心率。
    """
    
    print(f"\n输入文本：\n{test_text}")
    
    # 模拟抽取结果
    print("\n抽取结果：")
    print("实体：")
    print("  - 王某 (人物)")
    print("  - 55岁 (年龄)")
    print("  - 胸闷 (症状)")
    print("  - 胸痛 (症状)")
    print("  - 高血压 (疾病)")
    print("  - 糖尿病 (疾病)")
    print("  - 冠心病 (疾病)")
    print("  - 心电图 (检查)")
    print("  - 阿司匹林 (药物)")
    
    print("\n关系：")
    print("  - 冠心病 --症状表现--> 胸闷")
    print("  - 冠心病 --症状表现--> 胸痛")
    print("  - 冠心病 --治疗药物--> 阿司匹林")
    print("  - 冠心病 --检查方法--> 心电图")


def demo_knowledge_graph_building():
    """演示知识图谱构建"""
    print("\n=== 知识图谱构建演示 ===")
    
    # 创建配置
    config = GraphConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="medicalkg123"
    )
    
    try:
        # 创建图谱构建器
        with MedicalGraphBuilder(config) as builder:
            # 获取示例数据
            data = create_sample_data()
            
            # 批量创建节点
            print("\n创建节点...")
            builder.batch_create_nodes(data['nodes'])
            
            # 批量创建关系
            print("创建关系...")
            builder.batch_create_relationships(data['relationships'])
            
            # 获取统计信息
            stats = builder.get_statistics()
            print("\n图谱统计信息：")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # 演示查询
            print("\n执行示例查询...")
            
            # 查询糖尿病的症状
            result = builder.query(
                "MATCH (d:Disease {name: '糖尿病'})-[:HAS_SYMPTOM]->(s:Symptom) "
                "RETURN s.name as symptom"
            )
            print("\n糖尿病的症状：")
            for r in result:
                print(f"  - {r['symptom']}")
            
            # 分析器演示
            analyzer = MedicalGraphAnalyzer(builder)
            
            # 查找相似疾病
            similar = analyzer.find_similar_diseases('高血压', limit=5)
            if similar:
                print("\n与高血压相似的疾病：")
                for s in similar:
                    print(f"  - {s['disease']} (共同症状数: {s['common_symptoms']})")
            
    except Exception as e:
        print(f"\n注意：Neo4j连接失败 - {e}")
        print("请确保Neo4j服务已启动，或运行: docker-compose up -d neo4j")


def demo_qa_system():
    """演示问答系统"""
    print("\n=== 问答系统演示 ===")
    
    # 测试问题
    test_questions = [
        "糖尿病有什么症状？",
        "头晕可能是什么病？",
        "高血压怎么治疗？",
        "冠心病需要做什么检查？",
        "阿司匹林是什么药？",
    ]
    
    print("\n模拟问答（实际使用需要连接Neo4j）：")
    
    for question in test_questions:
        print(f"\n问：{question}")
        
        # 模拟回答
        if "糖尿病" in question and "症状" in question:
            print("答：糖尿病的常见症状包括：多饮、多尿、多食。")
        elif "头晕" in question:
            print("答：出现头晕可能与以下疾病有关：高血压、颈椎病、贫血等。建议及时就医进行详细检查。")
        elif "高血压" in question and "治疗" in question:
            print("答：高血压的治疗方法包括：药物治疗（如硝苯地平）、生活方式改善（低盐饮食、适量运动）。具体治疗方案需要医生根据病情制定。")
        elif "冠心病" in question and "检查" in question:
            print("答：冠心病需要进行的检查包括：心电图、心脏彩超、冠脉造影等。")
        elif "阿司匹林" in question:
            print("答：阿司匹林是一种抗血小板药物，主要用于预防血栓形成，常用于冠心病、脑梗塞等疾病的治疗。")
        
        print("\n提示：以上信息仅供参考，具体诊疗请咨询专业医生。")


def demo_web_interface():
    """演示Web界面"""
    print("\n=== Web界面演示 ===")
    print("\n启动Web界面的步骤：")
    print("1. 启动所有服务：docker-compose up -d")
    print("2. 访问以下地址：")
    print("   - Neo4j界面: http://localhost:7474")
    print("   - Label Studio: http://localhost:8080")
    print("   - API文档: http://localhost:8000/docs")
    print("   - Web问答界面: http://localhost:8501")
    print("\n3. 使用curl测试API：")
    print('   curl -X POST "http://localhost:8000/qa" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"question": "糖尿病有什么症状？"}\'')


def main():
    """主函数"""
    print("=" * 60)
    print("医疗健康知识图谱实战项目演示")
    print("=" * 60)
    
    while True:
        print("\n请选择演示内容：")
        print("1. 知识抽取演示")
        print("2. 知识图谱构建演示")
        print("3. 问答系统演示")
        print("4. Web界面说明")
        print("5. 显示项目结构")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-5): ").strip()
        
        if choice == '0':
            print("\n感谢使用，再见！")
            break
        elif choice == '1':
            demo_knowledge_extraction()
        elif choice == '2':
            demo_knowledge_graph_building()
        elif choice == '3':
            demo_qa_system()
        elif choice == '4':
            demo_web_interface()
        elif choice == '5':
            print("\n项目结构：")
            print("""
medical_kg_project/
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── annotations/          # 标注数据
├── models/                   # 模型文件
├── src/                      # 源代码
│   ├── data_collection/      # 数据采集
│   │   └── medical_spider.py
│   ├── models/              # 模型训练
│   │   └── llm_extractor.py
│   ├── kg_builder/          # 图谱构建
│   │   └── neo4j_manager.py
│   └── qa_system/           # 问答系统
│       └── medical_qa.py
├── docker-compose.yml        # Docker配置
├── requirements.txt          # Python依赖
├── README.md                # 项目说明
└── demo.py                  # 演示脚本
            """)
        else:
            print("\n无效选项，请重新选择！")


if __name__ == "__main__":
    main()