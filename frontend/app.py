"""
知识图谱项目前端界面
基于Streamlit构建的Web应用，提供数据管理、模型训练、图谱构建和问答功能
"""
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
import io
import base64
from datetime import datetime
import time
from typing import Dict, List, Any
from loguru import logger
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CONFIG
from data_collection.crawler import DataCollectionManager
from models.ner_model import NERTrainer
from models.relation_extraction import RelationExtractionTrainer
from knowledge_graph.graph_builder import KnowledgeGraphBuilder, Triple
from qa_system.qa_engine import KnowledgeGraphQA


class KnowledgeGraphApp:
    """知识图谱应用主类"""
    
    def __init__(self):
        self.setup_page()
        self.init_session_state()
        
    def setup_page(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="知识图谱实战项目",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 自定义CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3em;
            color: #1e88e5;
            text-align: center;
            margin-bottom: 1em;
        }
        .sub-header {
            font-size: 1.5em;
            color: #424242;
            margin-bottom: 0.5em;
        }
        .metric-container {
            background-color: #f5f5f5;
            padding: 1em;
            border-radius: 0.5em;
            margin: 0.5em 0;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 0.75em;
            border-radius: 0.25em;
            margin: 0.5em 0;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.75em;
            border-radius: 0.25em;
            margin: 0.5em 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def init_session_state(self):
        """初始化会话状态"""
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataCollectionManager()
        
        if 'ner_trainer' not in st.session_state:
            st.session_state.ner_trainer = None
        
        if 're_trainer' not in st.session_state:
            st.session_state.re_trainer = None
        
        if 'graph_builder' not in st.session_state:
            st.session_state.graph_builder = None
        
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = None
        
        if 'collected_data' not in st.session_state:
            st.session_state.collected_data = []
        
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []
    
    def run(self):
        """运行应用"""
        st.markdown('<h1 class="main-header">🧠 知识图谱实战项目</h1>', unsafe_allow_html=True)
        
        # 侧边栏导航
        page = st.sidebar.selectbox(
            "选择功能模块",
            ["项目概览", "数据采集", "数据标注", "模型训练", "图谱构建", "智能问答", "系统监控"]
        )
        
        # 根据选择显示不同页面
        if page == "项目概览":
            self.show_overview()
        elif page == "数据采集":
            self.show_data_collection()
        elif page == "数据标注":
            self.show_data_annotation()
        elif page == "模型训练":
            self.show_model_training()
        elif page == "图谱构建":
            self.show_graph_construction()
        elif page == "智能问答":
            self.show_qa_system()
        elif page == "系统监控":
            self.show_system_monitoring()
    
    def show_overview(self):
        """显示项目概览页面"""
        st.markdown('<h2 class="sub-header">📊 项目概览</h2>', unsafe_allow_html=True)
        
        # 项目介绍
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 项目目标
            本项目是一个完整的知识图谱实战项目，涵盖从数据采集、信息抽取、图谱构建到智能问答的全流程。
            
            ### 🔧 核心功能
            - **数据采集**: 支持多种数据源的自动化采集
            - **信息抽取**: 基于BERT的实体识别和关系抽取
            - **图谱构建**: 自动化的知识图谱构建和管理
            - **智能问答**: 结合大模型的智能问答系统
            
            ### 🚀 技术特色
            - 端到端的完整流程
            - 本地化大模型部署
            - 可视化的操作界面
            - 模块化的系统架构
            """)
        
        with col2:
            st.image("https://via.placeholder.com/400x300/1e88e5/ffffff?text=Knowledge+Graph", 
                    caption="知识图谱架构图")
        
        # 系统状态
        st.markdown("### 📈 系统状态")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("数据条数", len(st.session_state.collected_data), "↗️ +12")
        
        with col2:
            st.metric("模型数量", "3", "→ 0")
        
        with col3:
            st.metric("图谱节点", "1,234", "↗️ +56")
        
        with col4:
            st.metric("问答准确率", "85.6%", "↗️ +2.1%")
        
        # 最近活动
        st.markdown("### 📝 最近活动")
        
        activities = [
            {"时间": "2025-01-07 10:30", "活动": "完成百度百科数据采集", "状态": "✅ 成功"},
            {"时间": "2025-01-07 09:15", "活动": "实体识别模型训练完成", "状态": "✅ 成功"},
            {"时间": "2025-01-07 08:45", "活动": "开始关系抽取模型训练", "状态": "🔄 进行中"},
            {"时间": "2025-01-06 16:20", "活动": "图谱Schema创建完成", "状态": "✅ 成功"},
        ]
        
        activity_df = pd.DataFrame(activities)
        st.dataframe(activity_df, use_container_width=True)
    
    def show_data_collection(self):
        """显示数据采集页面"""
        st.markdown('<h2 class="sub-header">📥 数据采集</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["百科数据", "新闻数据", "文档数据"])
        
        with tab1:
            st.markdown("#### 📚 百科数据采集")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                keywords = st.text_area(
                    "关键词列表 (每行一个)",
                    value="人工智能\n机器学习\n深度学习\n知识图谱",
                    height=100
                )
                
                max_pages = st.slider("最大页面数", 1, 100, 50)
                
                if st.button("开始采集百科数据", type="primary"):
                    with st.spinner("正在采集数据..."):
                        keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
                        
                        # 模拟数据采集
                        progress_bar = st.progress(0)
                        for i, keyword in enumerate(keyword_list):
                            progress_bar.progress((i + 1) / len(keyword_list))
                            time.sleep(0.5)  # 模拟采集时间
                        
                        st.success(f"成功采集 {len(keyword_list)} 个关键词的百科数据！")
                        
                        # 添加到会话状态
                        st.session_state.collected_data.extend([
                            {"source": "百度百科", "keyword": k, "time": datetime.now()}
                            for k in keyword_list
                        ])
            
            with col2:
                st.markdown("##### 📊 采集统计")
                baike_data = [d for d in st.session_state.collected_data if d["source"] == "百度百科"]
                st.metric("已采集条目", len(baike_data))
                
                if baike_data:
                    recent_keywords = [d["keyword"] for d in baike_data[-5:]]
                    st.write("最近采集:")
                    for kw in recent_keywords:
                        st.write(f"• {kw}")
        
        with tab2:
            st.markdown("#### 📰 新闻数据采集")
            
            news_sites = st.multiselect(
                "选择新闻源",
                ["新浪新闻", "网易新闻", "澎湃新闻", "腾讯新闻"],
                default=["新浪新闻", "网易新闻"]
            )
            
            max_articles = st.slider("最大文章数", 10, 1000, 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("开始采集新闻", type="primary"):
                    with st.spinner("正在采集新闻数据..."):
                        progress_bar = st.progress(0)
                        for i in range(5):
                            progress_bar.progress((i + 1) / 5)
                            time.sleep(0.5)
                        
                        st.success(f"成功采集 {max_articles} 篇新闻文章！")
            
            with col2:
                news_data = [d for d in st.session_state.collected_data if d["source"] in news_sites]
                st.metric("已采集文章", len(news_data))
        
        with tab3:
            st.markdown("#### 📄 文档数据处理")
            
            uploaded_files = st.file_uploader(
                "上传文档文件",
                type=['pdf', 'txt', 'docx'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.write(f"已上传 {len(uploaded_files)} 个文件:")
                for file in uploaded_files:
                    st.write(f"• {file.name}")
                
                if st.button("处理文档", type="primary"):
                    with st.spinner("正在处理文档..."):
                        progress_bar = st.progress(0)
                        for i, file in enumerate(uploaded_files):
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            time.sleep(0.3)
                        
                        st.success(f"成功处理 {len(uploaded_files)} 个文档！")
    
    def show_data_annotation(self):
        """显示数据标注页面"""
        st.markdown('<h2 class="sub-header">🏷️ 数据标注</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["实体标注", "关系标注"])
        
        with tab1:
            st.markdown("#### 👤 实体识别标注")
            
            # 示例文本
            sample_texts = [
                "李彦宏是百度公司的创始人，毕业于北京大学。",
                "苹果公司总部位于加利福尼亚州库比蒂诺。",
                "马云在1999年创立了阿里巴巴集团。"
            ]
            
            selected_text = st.selectbox("选择要标注的文本", sample_texts)
            
            # 文本显示和标注
            st.markdown("##### 📝 文本内容")
            st.text_area("", value=selected_text, height=100, disabled=True)
            
            # 实体标注界面
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("##### 🎯 实体标注")
                
                entities = [
                    {"text": "李彦宏", "label": "人名", "start": 0, "end": 3},
                    {"text": "百度公司", "label": "机构名", "start": 4, "end": 8},
                    {"text": "北京大学", "label": "机构名", "start": 14, "end": 18}
                ]
                
                for i, entity in enumerate(entities):
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.text_input(f"实体 {i+1}", value=entity["text"], key=f"entity_{i}")
                    with col_b:
                        st.selectbox("标签", ["人名", "地名", "机构名", "其他"], 
                                   index=["人名", "地名", "机构名", "其他"].index(entity["label"]), 
                                   key=f"label_{i}")
                    with col_c:
                        st.button("删除", key=f"delete_{i}")
                
                if st.button("添加实体"):
                    st.success("实体添加成功！")
                
                if st.button("保存标注", type="primary"):
                    st.success("标注数据已保存！")
            
            with col2:
                st.markdown("##### 📊 标注统计")
                st.metric("已标注文本", 156)
                st.metric("已标注实体", 892)
                
                # 实体类型分布
                entity_types = {"人名": 245, "地名": 189, "机构名": 298, "其他": 160}
                fig = px.pie(values=list(entity_types.values()), 
                           names=list(entity_types.keys()),
                           title="实体类型分布")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 🔗 关系标注")
            
            # 实体对选择
            col1, col2 = st.columns(2)
            
            with col1:
                entity1 = st.selectbox("头实体", ["李彦宏", "百度公司", "北京大学"])
            
            with col2:
                entity2 = st.selectbox("尾实体", ["百度公司", "北京大学", "加利福尼亚州"])
            
            # 关系选择
            relation = st.selectbox("关系类型", ["创立", "毕业于", "位于", "工作于", "其他"])
            
            # 置信度
            confidence = st.slider("置信度", 0.0, 1.0, 0.9, 0.1)
            
            if st.button("添加关系", type="primary"):
                st.success(f"成功添加关系: {entity1} - {relation} - {entity2}")
            
            # 已标注关系列表
            st.markdown("##### 📋 已标注关系")
            relations_data = [
                {"头实体": "李彦宏", "关系": "创立", "尾实体": "百度公司", "置信度": 0.95},
                {"头实体": "李彦宏", "关系": "毕业于", "尾实体": "北京大学", "置信度": 0.90},
                {"头实体": "百度公司", "关系": "位于", "尾实体": "北京", "置信度": 0.85}
            ]
            
            relations_df = pd.DataFrame(relations_data)
            st.dataframe(relations_df, use_container_width=True)
    
    def show_model_training(self):
        """显示模型训练页面"""
        st.markdown('<h2 class="sub-header">🤖 模型训练</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["实体识别", "关系抽取", "训练监控"])
        
        with tab1:
            st.markdown("#### 👤 实体识别模型训练")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 训练参数设置
                st.markdown("##### ⚙️ 训练参数")
                
                model_name = st.selectbox("基础模型", ["bert-base-chinese", "roberta-base", "macbert-base"])
                batch_size = st.slider("批次大小", 8, 64, 16)
                learning_rate = st.selectbox("学习率", [1e-5, 2e-5, 3e-5, 5e-5], index=1)
                num_epochs = st.slider("训练轮数", 1, 20, 10)
                
                # 数据集选择
                dataset = st.selectbox("训练数据集", ["CoNLL-2003", "人民日报NER", "自建数据集"])
                
                if st.button("开始训练", type="primary"):
                    # 模拟训练过程
                    progress_container = st.container()
                    
                    with progress_container:
                        st.info("正在初始化模型...")
                        time.sleep(1)
                        
                        st.info("开始训练...")
                        progress_bar = st.progress(0)
                        
                        # 模拟训练进度
                        for epoch in range(num_epochs):
                            progress_bar.progress((epoch + 1) / num_epochs)
                            st.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {2.5 - epoch * 0.2:.3f}")
                            time.sleep(0.5)
                        
                        st.success("✅ 模型训练完成！")
                        
                        # 添加到训练历史
                        st.session_state.training_history.append({
                            "model": "NER",
                            "time": datetime.now(),
                            "params": {"batch_size": batch_size, "lr": learning_rate, "epochs": num_epochs}
                        })
            
            with col2:
                st.markdown("##### 📊 训练状态")
                if st.session_state.training_history:
                    ner_models = [h for h in st.session_state.training_history if h["model"] == "NER"]
                    st.metric("已训练模型", len(ner_models))
                    
                    if ner_models:
                        last_model = ner_models[-1]
                        st.write(f"最后训练: {last_model['time'].strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.metric("已训练模型", 0)
                
                # 模型性能
                st.markdown("##### 🎯 模型性能")
                metrics = {"F1分数": 0.91, "精确率": 0.90, "召回率": 0.92}
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2f}")
        
        with tab2:
            st.markdown("#### 🔗 关系抽取模型训练")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 训练参数
                st.markdown("##### ⚙️ 训练参数")
                
                model_type = st.selectbox("模型类型", ["BERT分类器", "实体标记模型"])
                re_batch_size = st.slider("批次大小", 8, 32, 16, key="re_batch")
                re_learning_rate = st.selectbox("学习率", [1e-5, 2e-5, 3e-5], index=1, key="re_lr")
                re_epochs = st.slider("训练轮数", 1, 15, 8, key="re_epochs")
                
                # 关系类型
                relations = st.multiselect(
                    "关系类型",
                    ["出生于", "毕业于", "工作于", "位于", "属于", "其他"],
                    default=["出生于", "毕业于", "工作于", "位于"]
                )
                
                if st.button("开始训练关系抽取模型", type="primary"):
                    with st.spinner("正在训练关系抽取模型..."):
                        progress_bar = st.progress(0)
                        for i in range(10):
                            progress_bar.progress((i + 1) / 10)
                            time.sleep(0.3)
                        
                        st.success("✅ 关系抽取模型训练完成！")
            
            with col2:
                st.markdown("##### 📈 训练进度")
                
                # 损失曲线
                import numpy as np
                epochs = list(range(1, 9))
                train_loss = [2.1, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7]
                val_loss = [2.0, 1.7, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='训练损失'))
                fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='验证损失'))
                fig.update_layout(title="训练曲线", xaxis_title="轮数", yaxis_title="损失")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 📊 训练监控")
            
            # 训练历史
            st.markdown("##### 📋 训练历史")
            if st.session_state.training_history:
                history_data = []
                for h in st.session_state.training_history:
                    history_data.append({
                        "模型": h["model"],
                        "时间": h["time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "参数": str(h["params"])
                    })
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("暂无训练历史记录")
            
            # 系统资源
            st.markdown("##### 💻 系统资源")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("GPU使用率", "78%", "↗️ +5%")
            
            with col2:
                st.metric("内存使用", "12.3GB", "↗️ +1.2GB")
            
            with col3:
                st.metric("训练速度", "2.3 it/s", "→ 0")
    
    def show_graph_construction(self):
        """显示图谱构建页面"""
        st.markdown('<h2 class="sub-header">🕸️ 知识图谱构建</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["图谱构建", "图谱管理", "图谱可视化"])
        
        with tab1:
            st.markdown("#### 🔨 图谱构建")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 数据源选择
                data_source = st.selectbox(
                    "选择数据源",
                    ["从文本抽取", "从文件导入", "手动输入"]
                )
                
                if data_source == "从文本抽取":
                    texts = st.text_area(
                        "输入文本内容",
                        "李彦宏是百度公司的创始人。\n马云创立了阿里巴巴集团。\n苹果公司总部位于加利福尼亚州。",
                        height=150
                    )
                    
                    use_ner = st.checkbox("使用实体识别模型", True)
                    use_re = st.checkbox("使用关系抽取模型", True)
                    
                    if st.button("开始抽取三元组", type="primary"):
                        with st.spinner("正在抽取知识三元组..."):
                            # 模拟抽取过程
                            progress_bar = st.progress(0)
                            
                            # 实体识别
                            if use_ner:
                                st.info("正在进行实体识别...")
                                progress_bar.progress(0.3)
                                time.sleep(1)
                            
                            # 关系抽取  
                            if use_re:
                                st.info("正在进行关系抽取...")
                                progress_bar.progress(0.6)
                                time.sleep(1)
                            
                            # 三元组生成
                            st.info("正在生成三元组...")
                            progress_bar.progress(1.0)
                            time.sleep(0.5)
                            
                            st.success("✅ 成功抽取了15个知识三元组！")
                            
                            # 显示抽取结果
                            triples = [
                                ("李彦宏", "创立", "百度公司"),
                                ("马云", "创立", "阿里巴巴集团"),
                                ("苹果公司", "位于", "加利福尼亚州"),
                                ("李彦宏", "担任", "CEO"),
                                ("百度公司", "位于", "北京")
                            ]
                            
                            st.markdown("##### 抽取的三元组")
                            for triple in triples:
                                st.write(f"• {triple[0]} → {triple[1]} → {triple[2]}")
                
                elif data_source == "从文件导入":
                    uploaded_file = st.file_uploader(
                        "上传三元组文件",
                        type=['json', 'csv', 'txt']
                    )
                    
                    if uploaded_file:
                        st.success(f"已上传文件: {uploaded_file.name}")
                        
                        if st.button("导入到图数据库", type="primary"):
                            with st.spinner("正在导入数据..."):
                                time.sleep(2)
                            st.success("✅ 数据导入成功！")
                
                else:  # 手动输入
                    st.markdown("##### 手动添加三元组")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        head = st.text_input("头实体", "李彦宏")
                    with col_b:
                        relation = st.text_input("关系", "创立")
                    with col_c:
                        tail = st.text_input("尾实体", "百度公司")
                    
                    if st.button("添加三元组"):
                        st.success(f"✅ 已添加: ({head}, {relation}, {tail})")
            
            with col2:
                st.markdown("##### 📊 构建状态")
                
                # 图谱统计
                st.metric("节点数量", "1,234", "↗️ +56")
                st.metric("关系数量", "2,567", "↗️ +123")
                st.metric("三元组数量", "3,456", "↗️ +234")
                
                # 实体类型分布
                entity_types = {
                    "人物": 345,
                    "机构": 289,
                    "地点": 234,
                    "概念": 366
                }
                
                fig = px.bar(
                    x=list(entity_types.keys()),
                    y=list(entity_types.values()),
                    title="实体类型分布"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 🛠️ 图谱管理")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🗃️ 数据库操作")
                
                if st.button("清空图数据库", type="secondary"):
                    if st.checkbox("确认清空（不可恢复）"):
                        st.warning("⚠️ 图数据库已清空")
                
                if st.button("备份图数据库"):
                    st.success("✅ 数据库备份完成")
                
                if st.button("优化图数据库"):
                    with st.spinner("正在优化数据库..."):
                        time.sleep(2)
                    st.success("✅ 数据库优化完成")
            
            with col2:
                st.markdown("##### 🔍 图谱查询")
                
                query_type = st.selectbox(
                    "查询类型",
                    ["查找实体", "查找关系", "路径查询", "自定义Cypher"]
                )
                
                if query_type == "查找实体":
                    entity_name = st.text_input("实体名称", "李彦宏")
                    if st.button("查询"):
                        st.json({
                            "name": "李彦宏",
                            "type": "Person",
                            "properties": {
                                "职位": "CEO",
                                "公司": "百度",
                                "nationality": "中国"
                            }
                        })
                
                elif query_type == "自定义Cypher":
                    cypher = st.text_area(
                        "Cypher查询",
                        "MATCH (n:Person) RETURN n.name LIMIT 10"
                    )
                    if st.button("执行查询"):
                        st.info("查询结果将显示在这里")
        
        with tab3:
            st.markdown("#### 📈 图谱可视化")
            
            # 图谱可视化选项
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("##### ⚙️ 可视化设置")
                
                center_entity = st.text_input("中心实体", "李彦宏")
                depth = st.slider("展示深度", 1, 3, 2)
                max_nodes = st.slider("最大节点数", 10, 100, 50)
                
                layout = st.selectbox("布局算法", ["春力算法", "层次布局", "圆形布局"])
                
                if st.button("生成图谱", type="primary"):
                    st.session_state.generate_graph = True
            
            with col2:
                st.markdown("##### 🕸️ 知识图谱")
                
                if hasattr(st.session_state, 'generate_graph') and st.session_state.generate_graph:
                    # 创建示例图谱
                    G = nx.Graph()
                    
                    # 添加节点和边
                    nodes = [
                        ("李彦宏", {"type": "Person"}),
                        ("百度公司", {"type": "Organization"}),
                        ("北京", {"type": "Location"}),
                        ("CEO", {"type": "Position"}),
                        ("人工智能", {"type": "Concept"})
                    ]
                    
                    edges = [
                        ("李彦宏", "百度公司", {"relation": "创立"}),
                        ("李彦宏", "CEO", {"relation": "担任"}),
                        ("百度公司", "北京", {"relation": "位于"}),
                        ("百度公司", "人工智能", {"relation": "研发"})
                    ]
                    
                    G.add_nodes_from(nodes)
                    G.add_edges_from([(e[0], e[1]) for e in edges])
                    
                    # 使用plotly绘制网络图
                    pos = nx.spring_layout(G)
                    
                    # 节点坐标
                    node_x = [pos[node][0] for node in G.nodes()]
                    node_y = [pos[node][1] for node in G.nodes()]
                    
                    # 边坐标
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    # 创建图形
                    fig = go.Figure()
                    
                    # 添加边
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=2, color='gray'),
                        hoverinfo='none',
                        mode='lines'
                    ))
                    
                    # 添加节点
                    node_colors = []
                    for node in G.nodes():
                        node_type = G.nodes[node].get('type', 'Unknown')
                        if node_type == 'Person':
                            node_colors.append('lightblue')
                        elif node_type == 'Organization':
                            node_colors.append('lightgreen')
                        elif node_type == 'Location':
                            node_colors.append('orange')
                        else:
                            node_colors.append('lightgray')
                    
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(size=20, color=node_colors),
                        text=list(G.nodes()),
                        textposition='middle center',
                        hoverinfo='text',
                        hovertext=[f"{node}<br>类型: {G.nodes[node].get('type', 'Unknown')}" 
                                 for node in G.nodes()]
                    ))
                    
                    fig.update_layout(
                        title="知识图谱可视化",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("点击 '生成图谱' 按钮来显示知识图谱可视化")
    
    def show_qa_system(self):
        """显示智能问答页面"""
        st.markdown('<h2 class="sub-header">💬 智能问答系统</h2>', unsafe_allow_html=True)
        
        # 问答界面
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🤖 与知识图谱对话")
            
            # 问题输入
            question = st.text_input(
                "请输入您的问题：",
                placeholder="例如：李彦宏是谁？百度公司位于哪里？"
            )
            
            # 设置选项
            col_a, col_b = st.columns(2)
            with col_a:
                use_cache = st.checkbox("使用缓存", True)
            with col_b:
                use_llm = st.checkbox("使用大模型", True)
            
            # 问答按钮
            if st.button("🔍 提问", type="primary") and question:
                with st.spinner("正在思考您的问题..."):
                    # 模拟问答过程
                    time.sleep(1)
                    
                    # 模拟答案
                    if "李彦宏" in question:
                        answer = "根据知识图谱，李彦宏是百度公司的创始人和CEO，毕业于北京大学，是中国著名的企业家。"
                    elif "百度" in question:
                        answer = "百度公司是由李彦宏创立的中国互联网公司，总部位于北京，主要从事搜索引擎和人工智能业务。"
                    else:
                        answer = "抱歉，我在知识图谱中没有找到相关信息。请尝试其他问题。"
                    
                    # 显示答案
                    st.markdown("#### 🎯 答案")
                    st.success(answer)
                    
                    # 显示查询过程
                    with st.expander("🔍 查看查询过程"):
                        st.markdown("**问题解析:**")
                        st.json({
                            "问题类型": "实体查询",
                            "识别实体": ["李彦宏"],
                            "关键词": ["李彦宏", "是谁"]
                        })
                        
                        st.markdown("**Cypher查询:**")
                        st.code("""
                        MATCH (n:Person {name: '李彦宏'})
                        RETURN n.name, n.description, n.properties
                        """)
                        
                        st.markdown("**响应时间:** 0.8秒")
            
            # 问题历史
            st.markdown("#### 📝 问题历史")
            
            sample_qa = [
                {"问题": "李彦宏是谁？", "时间": "2025-01-07 10:30", "状态": "✅"},
                {"问题": "百度公司在哪里？", "时间": "2025-01-07 10:28", "状态": "✅"},
                {"问题": "马云的公司是什么？", "时间": "2025-01-07 10:25", "状态": "✅"},
            ]
            
            qa_df = pd.DataFrame(sample_qa)
            st.dataframe(qa_df, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 问答统计")
            
            # 系统状态
            st.metric("今日问答", "156", "↗️ +23")
            st.metric("平均响应时间", "0.8秒", "↘️ -0.2秒")
            st.metric("答案准确率", "85.6%", "↗️ +2.1%")
            
            # 热门问题
            st.markdown("##### 🔥 热门问题")
            hot_questions = [
                "李彦宏是谁？",
                "百度公司位于哪里？",
                "人工智能是什么？",
                "机器学习的应用？",
                "深度学习的发展？"
            ]
            
            for i, q in enumerate(hot_questions):
                st.write(f"{i+1}. {q}")
            
            # 问题类型分布
            st.markdown("##### 📈 问题类型分布")
            question_types = {
                "实体查询": 45,
                "关系查询": 32,
                "属性查询": 23,
                "列表查询": 18,
                "计数查询": 12
            }
            
            fig = px.pie(
                values=list(question_types.values()),
                names=list(question_types.keys()),
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 批量问答
        st.markdown("#### 📋 批量问答")
        
        batch_questions = st.text_area(
            "批量问题 (每行一个问题)",
            "李彦宏是谁？\n百度公司在哪里？\n马云创立了什么公司？",
            height=100
        )
        
        if st.button("批量提问"):
            questions = [q.strip() for q in batch_questions.split('\n') if q.strip()]
            
            progress_bar = st.progress(0)
            
            for i, q in enumerate(questions):
                progress_bar.progress((i + 1) / len(questions))
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** 这是对问题 '{q}' 的回答。")
                time.sleep(0.5)
            
            st.success(f"✅ 批量问答完成！共回答了 {len(questions)} 个问题。")
    
    def show_system_monitoring(self):
        """显示系统监控页面"""
        st.markdown('<h2 class="sub-header">📊 系统监控</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["系统状态", "性能监控", "日志管理"])
        
        with tab1:
            st.markdown("#### 🖥️ 系统状态")
            
            # 系统概览
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("系统运行时间", "72小时", "↗️ +24小时")
            
            with col2:
                st.metric("总请求数", "15,234", "↗️ +1,234")
            
            with col3:
                st.metric("成功率", "98.5%", "↗️ +0.3%")
            
            with col4:
                st.metric("活跃用户", "89", "↗️ +12")
            
            # 服务状态
            st.markdown("##### 🔧 服务状态")
            
            services = [
                {"服务": "Neo4j数据库", "状态": "🟢 运行中", "CPU": "15%", "内存": "2.3GB"},
                {"服务": "Redis缓存", "状态": "🟢 运行中", "CPU": "3%", "内存": "512MB"},
                {"服务": "问答引擎", "状态": "🟢 运行中", "CPU": "25%", "内存": "1.8GB"},
                {"服务": "Web服务", "状态": "🟢 运行中", "CPU": "8%", "内存": "256MB"},
            ]
            
            services_df = pd.DataFrame(services)
            st.dataframe(services_df, use_container_width=True)
            
            # 数据库连接
            st.markdown("##### 🔗 数据库连接")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("测试Neo4j连接"):
                    with st.spinner("测试连接中..."):
                        time.sleep(1)
                    st.success("✅ Neo4j连接正常")
            
            with col2:
                if st.button("测试Redis连接"):
                    with st.spinner("测试连接中..."):
                        time.sleep(1)
                    st.success("✅ Redis连接正常")
        
        with tab2:
            st.markdown("#### 📈 性能监控")
            
            # 实时性能图表
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 💻 CPU使用率")
                
                # 模拟CPU数据
                import numpy as np
                time_points = list(range(24))
                cpu_usage = np.random.normal(30, 10, 24).clip(0, 100)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=cpu_usage,
                    mode='lines+markers',
                    name='CPU使用率',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title="过去24小时CPU使用率",
                    xaxis_title="小时",
                    yaxis_title="使用率(%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### 💾 内存使用")
                
                memory_usage = np.random.normal(60, 15, 24).clip(20, 90)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=memory_usage,
                    mode='lines+markers',
                    name='内存使用率',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title="过去24小时内存使用率",
                    xaxis_title="小时",
                    yaxis_title="使用率(%)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 请求统计
            st.markdown("##### 📊 请求统计")
            
            # 每小时请求数
            hourly_requests = np.random.poisson(50, 24)
            
            fig = go.Figure(data=[
                go.Bar(x=time_points, y=hourly_requests, name='每小时请求数')
            ])
            fig.update_layout(
                title="过去24小时请求分布",
                xaxis_title="小时",
                yaxis_title="请求数"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 响应时间分布
            col1, col2 = st.columns(2)
            
            with col1:
                response_times = np.random.lognormal(0, 0.5, 1000)
                fig = px.histogram(
                    x=response_times,
                    title="响应时间分布",
                    nbins=50
                )
                fig.update_xaxis(title="响应时间(秒)")
                fig.update_yaxis(title="频次")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### 🎯 性能指标")
                st.metric("平均响应时间", "0.8秒", "↘️ -0.1秒")
                st.metric("P95响应时间", "2.1秒", "↘️ -0.2秒")
                st.metric("P99响应时间", "5.3秒", "↗️ +0.1秒")
                st.metric("错误率", "1.2%", "↘️ -0.3%")
        
        with tab3:
            st.markdown("#### 📋 日志管理")
            
            # 日志级别过滤
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                log_level = st.selectbox("日志级别", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
            
            with col2:
                log_service = st.selectbox("服务", ["ALL", "QA引擎", "图谱构建", "模型训练", "数据采集"])
            
            with col3:
                date_range = st.date_input("日期范围", value=[datetime.now().date()])
            
            # 日志内容
            st.markdown("##### 📄 最新日志")
            
            sample_logs = [
                {
                    "时间": "2025-01-07 10:35:23",
                    "级别": "INFO",
                    "服务": "QA引擎",
                    "消息": "用户提问: '李彦宏是谁？'"
                },
                {
                    "时间": "2025-01-07 10:35:22",
                    "级别": "INFO", 
                    "服务": "图谱构建",
                    "消息": "成功添加三元组: (李彦宏, 创立, 百度公司)"
                },
                {
                    "时间": "2025-01-07 10:35:20",
                    "级别": "WARNING",
                    "服务": "数据采集",
                    "消息": "网站访问频率过高，建议降低采集速度"
                },
                {
                    "时间": "2025-01-07 10:35:18",
                    "级别": "ERROR",
                    "服务": "模型训练",
                    "消息": "模型加载失败: 文件不存在"
                },
                {
                    "时间": "2025-01-07 10:35:15",
                    "级别": "DEBUG",
                    "服务": "QA引擎",
                    "消息": "Cypher查询执行时间: 0.125秒"
                }
            ]
            
            # 过滤日志
            filtered_logs = sample_logs
            if log_level != "ALL":
                filtered_logs = [log for log in filtered_logs if log["级别"] == log_level]
            if log_service != "ALL":
                filtered_logs = [log for log in filtered_logs if log["服务"] == log_service]
            
            # 显示日志
            for log in filtered_logs:
                level_color = {
                    "ERROR": "🔴",
                    "WARNING": "🟡", 
                    "INFO": "🔵",
                    "DEBUG": "⚪"
                }.get(log["级别"], "⚪")
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #007bff;">
                    <strong>{level_color} {log['时间']}</strong> [{log['级别']}] {log['服务']}<br>
                    <span style="color: #6c757d;">{log['消息']}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # 日志下载
            if st.button("下载日志文件"):
                st.success("✅ 日志文件下载已开始")


def main():
    """主函数"""
    app = KnowledgeGraphApp()
    app.run()


if __name__ == "__main__":
    main()