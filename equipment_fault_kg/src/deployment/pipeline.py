import torch
import json
import logging
from typing import List, Dict, Tuple
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ..entity_extraction.trainer import NERTrainer
from ..relation_extraction.trainer import RETrainer
from ..entity_extraction.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str

class EntityOutput(BaseModel):
    text: str
    type: str
    start: int
    end: int
    confidence: float

class RelationOutput(BaseModel):
    head: EntityOutput
    tail: EntityOutput
    relation: str
    confidence: float

class PipelineOutput(BaseModel):
    text: str
    entities: List[EntityOutput]
    relations: List[RelationOutput]

class InformationExtractionPipeline:
    """信息抽取管道，整合实体抽取和关系抽取"""
    
    def __init__(self, ner_model_path: str = None, re_model_path: str = None):
        self.ner_trainer = None
        self.re_trainer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        if ner_model_path and os.path.exists(ner_model_path):
            self.load_ner_model(ner_model_path)
        
        if re_model_path and os.path.exists(re_model_path):
            self.load_re_model(re_model_path)
    
    def load_ner_model(self, model_path: str):
        """加载NER模型"""
        try:
            self.ner_trainer = NERTrainer()
            self.ner_trainer.load_model(model_path)
            logger.info(f"NER model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise
    
    def load_re_model(self, model_path: str):
        """加载关系抽取模型"""
        try:
            self.re_trainer = RETrainer()
            self.re_trainer.load_model(model_path)
            logger.info(f"RE model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load RE model: {e}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict]:
        """抽取实体"""
        if self.ner_trainer is None:
            raise ValueError("NER model not loaded")
        
        try:
            entities = self.ner_trainer.predict(text)
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """抽取关系"""
        if self.re_trainer is None:
            raise ValueError("RE model not loaded")
        
        if not entities:
            return []
        
        try:
            relations = self.re_trainer.predict(text, entities)
            return relations
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    def extract(self, text: str) -> Dict:
        """端到端信息抽取"""
        if not text.strip():
            return {
                'text': text,
                'entities': [],
                'relations': []
            }
        
        # 1. 实体抽取
        entities = self.extract_entities(text)
        
        # 2. 关系抽取
        relations = self.extract_relations(text, entities)
        
        # 3. 格式化输出
        result = {
            'text': text,
            'entities': entities,
            'relations': relations
        }
        
        return result
    
    def batch_extract(self, texts: List[str]) -> List[Dict]:
        """批量信息抽取"""
        results = []
        for text in texts:
            result = self.extract(text)
            results.append(result)
        return results

# 创建FastAPI应用
app = FastAPI(
    title="Equipment Fault Knowledge Graph Information Extraction API",
    description="API for extracting entities and relations from equipment fault texts",
    version="1.0.0"
)

# 全局管道实例
pipeline = None

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global pipeline
    
    # 模型路径配置
    ner_model_path = os.getenv("NER_MODEL_PATH", "models/ner_model")
    re_model_path = os.getenv("RE_MODEL_PATH", "models/re_model")
    
    try:
        pipeline = InformationExtractionPipeline(ner_model_path, re_model_path)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise

@app.post("/extract", response_model=PipelineOutput)
async def extract_information(input_data: TextInput):
    """单文本信息抽取"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        result = pipeline.extract(input_data.text)
        return PipelineOutput(**result)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_batch")
async def extract_batch_information(texts: List[str]):
    """批量文本信息抽取"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        results = pipeline.batch_extract(texts)
        return {"results": results}
    except Exception as e:
        logger.error(f"Batch extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "ner_model_loaded": pipeline.ner_trainer is not None if pipeline else False,
        "re_model_loaded": pipeline.re_trainer is not None if pipeline else False
    }

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Equipment Fault Knowledge Graph Information Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract": "Single text extraction",
            "/extract_batch": "Batch text extraction",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()