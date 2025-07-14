"""
大模型加载器

负责加载和初始化本地大模型，支持量化部署
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, Optional
import os
from loguru import logger


class ModelLoader:
    """大模型加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        self.model_path = config['model_path']
        self.quantization_config = config['quantization']
        self.inference_config = config['inference']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 设置量化参数
            load_kwargs = {}
            if self.quantization_config['load_in_8bit']:
                load_kwargs['load_in_8bit'] = True
                logger.info("使用8位量化加载模型")
            elif self.quantization_config['load_in_4bit']:
                load_kwargs['load_in_4bit'] = True
                logger.info("使用4位量化加载模型")
            
            # 加载模型
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map='auto' if torch.cuda.is_available() else None,
                **load_kwargs
            )
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info(f"模型加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载")
        
        # 合并配置参数
        generation_config = {
            'max_length': self.inference_config['max_length'],
            'temperature': self.inference_config['temperature'],
            'top_p': self.inference_config['top_p'],
            'repetition_penalty': self.inference_config['repetition_penalty'],
            **kwargs
        }
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除输入部分
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return f"生成失败: {str(e)}"
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """使用大模型进行实体抽取"""
        prompt = f"""
请从以下装备故障文本中抽取实体，以JSON格式返回：

文本：{text}

请识别以下类型的实体：
- Equipment（装备）：如数控机床、车床等
- Component（部件）：如主轴、伺服电机等
- Fault（故障）：具体的故障描述
- Cause（原因）：故障原因
- Solution（解决方案）：解决方法
- Symptom（症状）：故障表现
- Material（材料）：相关材料
- Tool（工具）：使用的工具

请以JSON格式返回，格式如下：
{{
    "entities": [
        {{"type": "实体类型", "text": "实体文本", "start": 起始位置, "end": 结束位置}}
    ]
}}
"""
        
        try:
            response = self.generate_response(prompt)
            
            # 尝试解析JSON
            import json
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return {"entities": []}
    
    def extract_relations(self, text: str) -> Dict[str, Any]:
        """使用大模型进行关系抽取"""
        prompt = f"""
请从以下装备故障文本中抽取实体关系，以JSON格式返回：

文本：{text}

请识别以下类型的关系：
- HAS_FAULT：装备-故障关系
- HAS_COMPONENT：装备-部件关系
- CAUSES：原因-故障关系
- SOLVES：解决方案-故障关系
- HAS_SYMPTOM：故障-症状关系
- REQUIRES_TOOL：解决方案-工具关系
- REQUIRES_MATERIAL：解决方案-材料关系

请以JSON格式返回，格式如下：
{{
    "relations": [
        {{"head": "头实体", "relation": "关系类型", "tail": "尾实体"}}
    ]
}}
"""
        
        try:
            response = self.generate_response(prompt)
            
            # 尝试解析JSON
            import json
            result = json.loads(response)
            return result
            
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return {"relations": []}
    
    def answer_question(self, question: str, context: str = "") -> str:
        """回答问题"""
        if context:
            prompt = f"""
基于以下上下文回答问题：

上下文：{context}

问题：{question}

请提供准确、详细的回答：
"""
        else:
            prompt = f"""
请回答以下关于装备制造故障的问题：

问题：{question}

请提供准确、详细的回答：
"""
        
        return self.generate_response(prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is None:
            return {"status": "未加载"}
        
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "quantization": self.quantization_config,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return info