"""Few-shot extraction using local LLM (Qwen/Baichuan).

需要已下载的模型权重，或使用 HF Hub 自动下载。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.data_collection.cleaning import clean_text

TRIPLE_PATTERN = re.compile(r"<([^:]+):([^>]+)>")  # e.g. <Symptom:振动异常>


class LLMBasedExtractor:
    """调用本地大模型进行 few-shot 信息抽取。"""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen1.5-0.5B-Chat", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # few-shot示例，可根据实际标注数据扩充
        self.shots = [
            (
                "主轴出现振动异常并报警E60，需要更换轴承。",
                "<Component:主轴> <Symptom:振动异常> <FaultCode:E60> <Action:更换轴承>"
            ),
        ]

    def build_prompt(self, text: str) -> str:
        prompt = "您是专业的信息抽取助手，请从句子中找出实体，输出格式为多个<类别:文本>，类别取值见示例。\n"
        for idx, (s, a) in enumerate(self.shots, 1):
            prompt += f"示例{idx}:\n句子:{s}\n结果:{a}\n"
        prompt += f"现在开始处理：\n句子:{text}\n结果:"
        return prompt

    def extract(self, text: str, max_new_tokens: int = 128) -> Dict:
        text = clean_text(text)
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        resp = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        ents: List[Tuple[str, str]] = []
        for typ, val in TRIPLE_PATTERN.findall(resp):
            ents.append((val, typ))
        return {"entities": ents, "relations": []}