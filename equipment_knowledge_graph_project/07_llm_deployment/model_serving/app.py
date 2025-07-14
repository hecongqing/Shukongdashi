#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地大模型服务应用
提供API接口供信息抽取模块调用
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

# 全局变量
model = None
tokenizer = None

class ChatRequest(BaseModel):
    """聊天请求模型"""
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    model_name: str
    timestamp: str
    tokens_used: int

class ModelInfo(BaseModel):
    """模型信息"""
    model_name: str
    model_type: str
    max_length: int
    device: str
    loaded: bool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    global model, tokenizer
    logger.info("正在加载大模型...")
    
    try:
        model_name = os.getenv("MODEL_NAME", "THUDM/chatglm2-6b")
        device = os.getenv("DEVICE", "auto")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        if hasattr(model, 'half'):
            model = model.half()
        
        model.eval()
        logger.info(f"模型加载完成: {model_name}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        model = None
        tokenizer = None
    
    yield
    
    # 关闭时清理资源
    logger.info("正在清理模型资源...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()

# 创建FastAPI应用
app = FastAPI(
    title="装备制造故障知识图谱大模型服务",
    description="基于本地部署大模型的装备制造故障信息抽取和问答服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LLMService:
    """大模型服务类"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.device = ""
    
    def load_model(self, model_name: str = "THUDM/chatglm2-6b", device: str = "auto"):
        """加载模型"""
        try:
            logger.info(f"开始加载模型: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device
            )
            
            if hasattr(self.model, 'half'):
                self.model = self.model.half()
            
            self.model.eval()
            self.model_name = model_name
            self.device = device
            
            logger.info("模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 2048, 
                         temperature: float = 0.7, top_p: float = 0.9,
                         top_k: int = 40, repetition_penalty: float = 1.1) -> str:
        """生成响应"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载")
        
        try:
            # 使用模型的chat方法
            if hasattr(self.model, 'chat'):
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=[],
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty
                )
                return response
            else:
                # 使用generate方法
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.replace(prompt, "").strip()
                
        except Exception as e:
            logger.error(f"生成响应失败: {e}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """抽取实体"""
        prompt = f"""
请从以下装备制造故障文本中抽取实体，包括设备、组件、故障、品牌、系统、错误代码等。

文本：{text}

请以JSON格式返回结果，格式如下：
{{
    "entities": [
        {{
            "text": "实体文本",
            "type": "实体类型",
            "start": 起始位置,
            "end": 结束位置
        }}
    ]
}}

实体类型说明：
- equipment: 设备名称
- component: 组件名称
- fault: 故障现象
- brand: 品牌名称
- system: 系统名称
- error_code: 错误代码
- solution: 解决方案
"""
        
        try:
            response = self.generate_response(prompt, temperature=0.3)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return {"entities": []}
    
    def extract_relations(self, text: str, entities: list) -> Dict[str, Any]:
        """抽取关系"""
        entity_texts = [f"{e['text']}({e['type']})" for e in entities]
        entities_str = ", ".join(entity_texts)
        
        prompt = f"""
请从以下装备制造故障文本中抽取实体间的关系。

文本：{text}

已识别的实体：{entities_str}

请以JSON格式返回结果，格式如下：
{{
    "relations": [
        {{
            "head": "头实体",
            "tail": "尾实体",
            "relation": "关系类型",
            "confidence": 置信度
        }}
    ]
}}

关系类型说明：
- cause: 导致关系（故障原因导致故障现象）
- contain: 包含关系（设备包含组件）
- belong_to: 属于关系（组件属于设备）
- solve: 解决关系（解决方案解决故障）
- trigger: 触发关系（操作触发故障）
- indicate: 指示关系（错误代码指示故障）
"""
        
        try:
            response = self.generate_response(prompt, temperature=0.3)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return {"relations": []}
    
    def answer_question(self, question: str, context: str = "") -> str:
        """回答问题"""
        if context:
            prompt = f"""
基于以下知识图谱信息回答问题：

知识图谱信息：
{context}

问题：{question}

请提供准确、详细的回答。
"""
        else:
            prompt = f"""
请回答以下装备制造故障相关问题：

问题：{question}

请提供准确、详细的回答。
"""
        
        try:
            response = self.generate_response(prompt, temperature=0.7)
            return response
        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return "抱歉，我无法回答这个问题。"
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1]
            else:
                json_str = response
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.warning(f"解析JSON响应失败: {e}")
            return {}

# 创建服务实例
llm_service = LLMService()

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "装备制造故障知识图谱大模型服务",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info() -> ModelInfo:
    """获取模型信息"""
    return ModelInfo(
        model_name=llm_service.model_name or "未加载",
        model_type="ChatGLM2-6B",
        max_length=2048,
        device=llm_service.device or "未加载",
        loaded=model is not None
    )

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """聊天接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        response = llm_service.generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        # 计算token使用量
        tokens_used = len(tokenizer.encode(request.prompt + response))
        
        return ChatResponse(
            response=response,
            model_name=llm_service.model_name,
            timestamp=datetime.now().isoformat(),
            tokens_used=tokens_used
        )
        
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract/entities")
async def extract_entities(request: ChatRequest) -> Dict[str, Any]:
    """实体抽取接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        result = llm_service.extract_entities(request.prompt)
        return {
            "entities": result.get("entities", []),
            "model_name": llm_service.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"实体抽取接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract/relations")
async def extract_relations(request: ChatRequest) -> Dict[str, Any]:
    """关系抽取接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 这里需要传入实体信息，简化处理
        entities = [{"text": "示例实体", "type": "equipment"}]
        result = llm_service.extract_relations(request.prompt, entities)
        return {
            "relations": result.get("relations", []),
            "model_name": llm_service.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"关系抽取接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa")
async def question_answering(request: ChatRequest) -> Dict[str, Any]:
    """问答接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        answer = llm_service.answer_question(request.prompt)
        return {
            "question": request.prompt,
            "answer": answer,
            "model_name": llm_service.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"问答接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch/extract")
async def batch_extract(request: ChatRequest) -> Dict[str, Any]:
    """批量抽取接口"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 实体抽取
        entities_result = llm_service.extract_entities(request.prompt)
        entities = entities_result.get("entities", [])
        
        # 关系抽取
        relations_result = llm_service.extract_relations(request.prompt, entities)
        relations = relations_result.get("relations", [])
        
        return {
            "text": request.prompt,
            "entities": entities,
            "relations": relations,
            "model_name": llm_service.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"批量抽取接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 配置参数
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    model_name = os.getenv("MODEL_NAME", "THUDM/chatglm2-6b")
    device = os.getenv("DEVICE", "auto")
    
    # 启动服务
    logger.info(f"启动大模型服务: {host}:{port}")
    logger.info(f"模型: {model_name}")
    logger.info(f"设备: {device}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )