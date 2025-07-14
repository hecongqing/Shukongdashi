#!/usr/bin/env python3
"""
本地大模型部署脚本
支持多种开源大模型的部署、量化和API服务
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
from datetime import datetime
from loguru import logger
import torch
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.config.settings import get_settings

class LLMDeploymentConfig(BaseModel):
    """大模型部署配置"""
    model_name: str
    model_path: str
    quantization: Optional[str] = None  # int4, int8, fp16
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    port: int = 8001
    host: str = "0.0.0.0"

class LLMDeployment:
    """大模型部署管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.config = None
        self.model = None
        self.tokenizer = None
        self.app = None
        self.deployment_type = None
        
    def check_system_requirements(self) -> Dict[str, Any]:
        """检查系统要求"""
        logger.info("检查系统要求...")
        
        # 检查GPU
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_memory = []
        
        if gpu_available:
            for i in range(gpu_count):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_memory.append(gpu_mem / (1024**3))  # GB
        
        # 检查CPU和内存
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        requirements = {
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_memory,
            "cpu_count": cpu_count,
            "memory_gb": round(memory_gb, 2),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available()
        }
        
        logger.info(f"系统配置: {requirements}")
        return requirements
    
    def recommend_deployment_config(self, model_name: str) -> LLMDeploymentConfig:
        """根据系统配置推荐部署配置"""
        requirements = self.check_system_requirements()
        
        # 模型配置映射
        model_configs = {
            "chatglm3-6b": {
                "min_memory_gb": 12,
                "recommended_memory_gb": 16,
                "quantization_options": ["int4", "int8", "fp16"]
            },
            "baichuan2-7b": {
                "min_memory_gb": 14,
                "recommended_memory_gb": 20,
                "quantization_options": ["int4", "int8", "fp16"]
            },
            "qwen-7b": {
                "min_memory_gb": 14,
                "recommended_memory_gb": 20,
                "quantization_options": ["int4", "int8", "fp16"]
            },
            "llama2-7b": {
                "min_memory_gb": 14,
                "recommended_memory_gb": 20,
                "quantization_options": ["int4", "int8", "fp16"]
            }
        }
        
        model_config = model_configs.get(model_name, {
            "min_memory_gb": 8,
            "recommended_memory_gb": 16,
            "quantization_options": ["int4", "int8"]
        })
        
        # 根据GPU内存选择量化方案
        quantization = "int4"
        if requirements["gpu_available"] and requirements["gpu_memory_gb"]:
            max_gpu_memory = max(requirements["gpu_memory_gb"])
            if max_gpu_memory >= model_config["recommended_memory_gb"]:
                quantization = "fp16"
            elif max_gpu_memory >= model_config["min_memory_gb"]:
                quantization = "int8"
        
        # 确定模型路径
        model_path = f"{self.settings.MODEL_DIR}/llm/{model_name}"
        
        config = LLMDeploymentConfig(
            model_name=model_name,
            model_path=model_path,
            quantization=quantization,
            tensor_parallel_size=min(requirements["gpu_count"], 2),
            gpu_memory_utilization=0.8 if requirements["gpu_available"] else 0.0
        )
        
        logger.info(f"推荐配置: {config}")
        return config
    
    def download_model(self, model_name: str) -> str:
        """下载模型"""
        logger.info(f"下载模型: {model_name}")
        
        # 模型下载映射
        model_urls = {
            "chatglm3-6b": "THUDM/chatglm3-6b",
            "baichuan2-7b": "baichuan-inc/Baichuan2-7B-Chat",
            "qwen-7b": "Qwen/Qwen-7B-Chat",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf"
        }
        
        model_url = model_urls.get(model_name)
        if not model_url:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_path = f"{self.settings.MODEL_DIR}/llm/{model_name}"
        
        # 检查模型是否已存在
        if Path(model_path).exists():
            logger.info(f"模型已存在: {model_path}")
            return model_path
        
        # 创建模型目录
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # 使用huggingface-cli下载
        try:
            cmd = [
                "huggingface-cli",
                "download",
                model_url,
                "--local-dir",
                model_path,
                "--local-dir-use-symlinks",
                "False"
            ]
            
            logger.info(f"执行下载命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"下载失败: {result.stderr}")
            
            logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"下载模型失败: {e}")
            raise
    
    def setup_vllm_deployment(self, config: LLMDeploymentConfig):
        """设置vLLM部署"""
        logger.info("设置vLLM部署...")
        
        try:
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # 配置vLLM参数
            engine_args = AsyncEngineArgs(
                model=config.model_path,
                tokenizer=config.model_path,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                dtype=config.dtype,
                quantization=config.quantization,
                max_model_len=config.max_tokens,
                trust_remote_code=True
            )
            
            # 创建异步引擎
            self.model = AsyncLLMEngine.from_engine_args(engine_args)
            self.deployment_type = "vllm"
            
            logger.info("vLLM部署设置完成")
            
        except ImportError:
            logger.error("vLLM未安装，请先安装: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"vLLM部署设置失败: {e}")
            raise
    
    def setup_transformers_deployment(self, config: LLMDeploymentConfig):
        """设置Transformers部署"""
        logger.info("设置Transformers部署...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=True
            )
            
            # 配置模型加载参数
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if config.tensor_parallel_size > 1 else None,
                "torch_dtype": torch.float16 if config.quantization == "fp16" else torch.float32
            }
            
            # 量化配置
            if config.quantization == "int8":
                model_kwargs["load_in_8bit"] = True
            elif config.quantization == "int4":
                model_kwargs["load_in_4bit"] = True
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                **model_kwargs
            )
            
            self.deployment_type = "transformers"
            logger.info("Transformers部署设置完成")
            
        except Exception as e:
            logger.error(f"Transformers部署设置失败: {e}")
            raise
    
    def setup_llama_cpp_deployment(self, config: LLMDeploymentConfig):
        """设置llama.cpp部署"""
        logger.info("设置llama.cpp部署...")
        
        try:
            from llama_cpp import Llama
            
            # 查找GGML模型文件
            model_file = None
            for ext in [".gguf", ".bin"]:
                potential_file = Path(config.model_path) / f"ggml-model-{config.quantization}{ext}"
                if potential_file.exists():
                    model_file = str(potential_file)
                    break
            
            if not model_file:
                raise FileNotFoundError(f"找不到GGML模型文件在: {config.model_path}")
            
            # 配置llama.cpp参数
            llama_kwargs = {
                "model_path": model_file,
                "n_ctx": config.max_tokens,
                "n_gpu_layers": -1 if torch.cuda.is_available() else 0,
                "verbose": False
            }
            
            # 创建Llama实例
            self.model = Llama(**llama_kwargs)
            self.deployment_type = "llama_cpp"
            
            logger.info("llama.cpp部署设置完成")
            
        except ImportError:
            logger.error("llama-cpp-python未安装，请先安装: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"llama.cpp部署设置失败: {e}")
            raise
    
    async def generate_response_vllm(self, prompt: str, **kwargs) -> str:
        """使用vLLM生成响应"""
        from vllm import SamplingParams
        
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_tokens=kwargs.get("max_tokens", 512),
            stop=kwargs.get("stop", [])
        )
        
        # 生成响应
        results = await self.model.generate(prompt, sampling_params)
        return results[0].outputs[0].text
    
    async def generate_response_transformers(self, prompt: str, **kwargs) -> str:
        """使用Transformers生成响应"""
        # 编码输入
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 生成参数
        generate_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generate_kwargs)
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        response = response[len(prompt):].strip()
        
        return response
    
    async def generate_response_llama_cpp(self, prompt: str, **kwargs) -> str:
        """使用llama.cpp生成响应"""
        # 生成参数
        generate_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stop": kwargs.get("stop", []),
            "echo": False
        }
        
        # 生成响应
        result = self.model(prompt, **generate_kwargs)
        
        return result["choices"][0]["text"]
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        if self.deployment_type == "vllm":
            return await self.generate_response_vllm(prompt, **kwargs)
        elif self.deployment_type == "transformers":
            return await self.generate_response_transformers(prompt, **kwargs)
        elif self.deployment_type == "llama_cpp":
            return await self.generate_response_llama_cpp(prompt, **kwargs)
        else:
            raise ValueError(f"不支持的部署类型: {self.deployment_type}")
    
    def create_api_server(self, config: LLMDeploymentConfig):
        """创建API服务器"""
        logger.info("创建API服务器...")
        
        # 请求和响应模型
        class GenerateRequest(BaseModel):
            prompt: str
            max_tokens: Optional[int] = 512
            temperature: Optional[float] = 0.7
            top_p: Optional[float] = 0.9
            stop: Optional[List[str]] = None
        
        class GenerateResponse(BaseModel):
            text: str
            model: str
            created: int
        
        # 创建FastAPI应用
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info("API服务器启动")
            yield
            logger.info("API服务器关闭")
        
        app = FastAPI(
            title="Local LLM API",
            description="本地大语言模型API服务",
            version="1.0.0",
            lifespan=lifespan
        )
        
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """生成文本"""
            try:
                response_text = await self.generate_response(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop or []
                )
                
                return GenerateResponse(
                    text=response_text,
                    model=config.model_name,
                    created=int(datetime.now().timestamp())
                )
            except Exception as e:
                logger.error(f"生成失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "healthy", "model": config.model_name}
        
        @app.get("/model/info")
        async def model_info():
            """模型信息"""
            return {
                "model_name": config.model_name,
                "model_path": config.model_path,
                "quantization": config.quantization,
                "deployment_type": self.deployment_type
            }
        
        self.app = app
        logger.info("API服务器创建完成")
    
    async def deploy_model(self, config: LLMDeploymentConfig):
        """部署模型"""
        logger.info(f"开始部署模型: {config.model_name}")
        
        self.config = config
        
        # 检查模型路径
        if not Path(config.model_path).exists():
            logger.info("模型不存在，开始下载...")
            self.download_model(config.model_name)
        
        # 根据配置选择部署方式
        deployment_methods = {
            "vllm": self.setup_vllm_deployment,
            "transformers": self.setup_transformers_deployment,
            "llama_cpp": self.setup_llama_cpp_deployment
        }
        
        # 尝试不同的部署方式
        for method_name, method in deployment_methods.items():
            try:
                method(config)
                logger.info(f"使用{method_name}部署成功")
                break
            except Exception as e:
                logger.warning(f"{method_name}部署失败: {e}")
                continue
        else:
            raise Exception("所有部署方式都失败了")
        
        # 创建API服务器
        self.create_api_server(config)
        
        logger.info("模型部署完成")
    
    def run_server(self, config: LLMDeploymentConfig):
        """运行服务器"""
        logger.info(f"启动API服务器: {config.host}:{config.port}")
        
        uvicorn.run(
            self.app,
            host=config.host,
            port=config.port,
            log_level="info"
        )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地大模型部署工具")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--port", type=int, default=8001, help="API端口")
    parser.add_argument("--host", default="0.0.0.0", help="API主机")
    parser.add_argument("--quantization", choices=["int4", "int8", "fp16"], help="量化方案")
    parser.add_argument("--download-only", action="store_true", help="仅下载模型")
    
    args = parser.parse_args()
    
    # 创建部署器
    deployment = LLMDeployment()
    
    # 生成配置
    config = deployment.recommend_deployment_config(args.model)
    
    # 覆盖配置
    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host
    if args.quantization:
        config.quantization = args.quantization
    
    # 仅下载模式
    if args.download_only:
        deployment.download_model(args.model)
        logger.info("模型下载完成")
        return
    
    # 部署并运行
    async def deploy_and_run():
        await deployment.deploy_model(config)
        deployment.run_server(config)
    
    asyncio.run(deploy_and_run())

if __name__ == "__main__":
    main()