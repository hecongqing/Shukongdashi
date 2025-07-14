#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于大模型的信息抽取模块
使用本地部署的大模型进行实体和关系抽取
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import requests
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMExtractor:
    """基于大模型的信息抽取器"""
    
    def __init__(self, model_name: str = "THUDM/chatglm2-6b", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.api_url = None
        
    def load_model(self):
        """加载本地大模型"""
        try:
            logger.info(f"开始加载模型: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            
            if hasattr(self.model, 'half'):
                self.model = self.model.half()
            
            self.model.eval()
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def setup_api_client(self, api_url: str):
        """设置API客户端"""
        self.api_url = api_url
        logger.info(f"API客户端已设置: {api_url}")
    
    def extract_entities_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用大模型抽取实体"""
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
            if self.model is not None:
                # 使用本地模型
                response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            elif self.api_url:
                # 使用API
                response = self._call_api(prompt)
            else:
                raise ValueError("模型未加载且API未设置")
            
            # 解析响应
            entities = self._parse_entity_response(response, text)
            return entities
            
        except Exception as e:
            logger.error(f"实体抽取失败: {e}")
            return []
    
    def extract_relations_with_llm(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """使用大模型抽取关系"""
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
            if self.model is not None:
                # 使用本地模型
                response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            elif self.api_url:
                # 使用API
                response = self._call_api(prompt)
            else:
                raise ValueError("模型未加载且API未设置")
            
            # 解析响应
            relations = self._parse_relation_response(response)
            return relations
            
        except Exception as e:
            logger.error(f"关系抽取失败: {e}")
            return []
    
    def extract_triples_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """使用大模型抽取三元组"""
        prompt = f"""
请从以下装备制造故障文本中抽取三元组（头实体-关系-尾实体）。

文本：{text}

请以JSON格式返回结果，格式如下：
{{
    "triples": [
        {{
            "head": "头实体",
            "relation": "关系",
            "tail": "尾实体",
            "head_type": "头实体类型",
            "tail_type": "尾实体类型",
            "confidence": 置信度
        }}
    ]
}}

请确保抽取的三元组准确且有意义。
"""
        
        try:
            if self.model is not None:
                # 使用本地模型
                response, _ = self.model.chat(self.tokenizer, prompt, history=[])
            elif self.api_url:
                # 使用API
                response = self._call_api(prompt)
            else:
                raise ValueError("模型未加载且API未设置")
            
            # 解析响应
            triples = self._parse_triple_response(response)
            return triples
            
        except Exception as e:
            logger.error(f"三元组抽取失败: {e}")
            return []
    
    def _call_api(self, prompt: str) -> str:
        """调用API"""
        try:
            payload = {
                "prompt": prompt,
                "max_length": 2048,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json().get("response", "")
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise
    
    def _parse_entity_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """解析实体抽取响应"""
        entities = []
        
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            if "entities" in data:
                for entity in data["entities"]:
                    # 验证实体位置
                    text = entity.get("text", "")
                    start = entity.get("start", 0)
                    end = entity.get("end", len(text))
                    
                    # 检查实体是否在原文中存在
                    if text in original_text:
                        entities.append({
                            "text": text,
                            "type": entity.get("type", "unknown"),
                            "start": start,
                            "end": end
                        })
            
        except Exception as e:
            logger.warning(f"解析实体响应失败: {e}")
            # 尝试使用正则表达式提取
            entities = self._extract_entities_with_regex(response, original_text)
        
        return entities
    
    def _parse_relation_response(self, response: str) -> List[Dict[str, Any]]:
        """解析关系抽取响应"""
        relations = []
        
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            if "relations" in data:
                for relation in data["relations"]:
                    relations.append({
                        "head": relation.get("head", ""),
                        "tail": relation.get("tail", ""),
                        "relation": relation.get("relation", ""),
                        "confidence": relation.get("confidence", 0.5)
                    })
            
        except Exception as e:
            logger.warning(f"解析关系响应失败: {e}")
        
        return relations
    
    def _parse_triple_response(self, response: str) -> List[Dict[str, Any]]:
        """解析三元组抽取响应"""
        triples = []
        
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1]
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            if "triples" in data:
                for triple in data["triples"]:
                    triples.append({
                        "head": triple.get("head", ""),
                        "relation": triple.get("relation", ""),
                        "tail": triple.get("tail", ""),
                        "head_type": triple.get("head_type", ""),
                        "tail_type": triple.get("tail_type", ""),
                        "confidence": triple.get("confidence", 0.5)
                    })
            
        except Exception as e:
            logger.warning(f"解析三元组响应失败: {e}")
        
        return triples
    
    def _extract_entities_with_regex(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """使用正则表达式提取实体（备用方法）"""
        import re
        
        entities = []
        
        # 定义实体模式
        patterns = {
            "equipment": r"设备[：:]\s*([^\n，。]+)",
            "component": r"组件[：:]\s*([^\n，。]+)",
            "fault": r"故障[：:]\s*([^\n，。]+)",
            "brand": r"品牌[：:]\s*([^\n，。]+)",
            "error_code": r"错误代码[：:]\s*([A-Z0-9]+)",
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, response)
            for match in matches:
                text = match.group(1).strip()
                if text in original_text:
                    start = original_text.find(text)
                    entities.append({
                        "text": text,
                        "type": entity_type,
                        "start": start,
                        "end": start + len(text)
                    })
        
        return entities
    
    def batch_extract(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量抽取信息"""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"处理第 {i+1}/{len(texts)} 个文本")
            
            try:
                # 抽取实体
                entities = self.extract_entities_with_llm(text)
                
                # 抽取关系
                relations = self.extract_relations_with_llm(text, entities)
                
                # 抽取三元组
                triples = self.extract_triples_with_llm(text)
                
                result = {
                    "text": text,
                    "entities": entities,
                    "relations": relations,
                    "triples": triples,
                    "processed_at": datetime.now().isoformat()
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理文本失败: {e}")
                results.append({
                    "text": text,
                    "entities": [],
                    "relations": [],
                    "triples": [],
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                })
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """保存抽取结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    # 创建抽取器
    extractor = LLMExtractor()
    
    # 方式1：使用本地模型
    try:
        extractor.load_model()
        logger.info("使用本地模型进行抽取")
    except:
        # 方式2：使用API
        extractor.setup_api_client("http://localhost:8000/api/chat")
        logger.info("使用API进行抽取")
    
    # 测试文本
    test_texts = [
        "数控机床主轴出现异常振动，检查发现主轴轴承磨损严重，需要更换轴承",
        "FANUC系统报警ALM401，伺服电机过载保护，检查发现电机温度过高",
        "西门子数控系统出现E1234报警，检查发现X轴伺服驱动器故障"
    ]
    
    # 批量抽取
    results = extractor.batch_extract(test_texts)
    
    # 保存结果
    extractor.save_results(results, "../data/llm_extraction_results.json")
    
    # 打印结果
    for i, result in enumerate(results):
        print(f"\n文本 {i+1}: {result['text']}")
        print("实体:")
        for entity in result['entities']:
            print(f"  {entity['type']}: {entity['text']}")
        print("关系:")
        for relation in result['relations']:
            print(f"  {relation['head']} --{relation['relation']}--> {relation['tail']}")
        print("三元组:")
        for triple in result['triples']:
            print(f"  ({triple['head']}, {triple['relation']}, {triple['tail']})")

if __name__ == "__main__":
    main()