"""
基于大模型的医疗信息抽取
支持本地部署的ChatGLM、Qwen等模型
"""
import json
import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dataclasses import dataclass
from loguru import logger
import re


@dataclass
class Entity:
    """实体类"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Relation:
    """关系类"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0


class MedicalLLMExtractor:
    """基于大模型的医疗信息抽取器"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = {
        'chatglm3-6b': 'THUDM/chatglm3-6b',
        'qwen-7b': 'Qwen/Qwen-7B-Chat',
        'baichuan2-7b': 'baichuan-inc/Baichuan2-7B-Chat',
        'glm-4': 'THUDM/glm-4-9b-chat'
    }
    
    # 医疗实体类型
    ENTITY_TYPES = [
        '疾病', '症状', '药物', '检查项目', 
        '治疗方法', '身体部位', '医疗设备', '病原体'
    ]
    
    # 医疗关系类型
    RELATION_TYPES = [
        '症状表现', '治疗药物', '检查方法', '引起原因',
        '发生部位', '并发症', '禁忌症', '适应症'
    ]
    
    def __init__(self, model_name: str = 'chatglm3-6b', device: str = 'auto'):
        """
        初始化抽取器
        
        Args:
            model_name: 模型名称
            device: 设备类型 ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device
        
        # 加载模型和分词器
        logger.info(f"Loading model: {model_name}")
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        # 根据模型类型选择加载方式
        if 'chatglm' in model_name.lower():
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=device if device == 'auto' else None,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=device if device == 'auto' else None,
                torch_dtype=torch.float16
            )
        
        if device != 'auto':
            self.model = self.model.to(device)
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def extract(self, text: str, task: str = 'all') -> Dict[str, any]:
        """
        抽取医疗信息
        
        Args:
            text: 输入文本
            task: 任务类型 ('all', 'entities', 'relations', 'events')
            
        Returns:
            抽取结果字典
        """
        results = {}
        
        if task in ['all', 'entities']:
            entities = self.extract_entities(text)
            results['entities'] = entities
            
        if task in ['all', 'relations']:
            # 如果已经抽取了实体，使用实体结果
            entities = results.get('entities', self.extract_entities(text))
            relations = self.extract_relations(text, entities)
            results['relations'] = relations
            
        if task in ['all', 'events']:
            events = self.extract_events(text)
            results['events'] = events
            
        return results
    
    def extract_entities(self, text: str) -> List[Entity]:
        """抽取实体"""
        prompt = self._build_entity_prompt(text)
        response = self._generate(prompt)
        entities = self._parse_entity_response(response, text)
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """抽取关系"""
        if len(entities) < 2:
            return []
            
        prompt = self._build_relation_prompt(text, entities)
        response = self._generate(prompt)
        relations = self._parse_relation_response(response, entities)
        return relations
    
    def extract_events(self, text: str) -> List[Dict]:
        """抽取医疗事件"""
        prompt = self._build_event_prompt(text)
        response = self._generate(prompt)
        events = self._parse_event_response(response)
        return events
    
    def _build_entity_prompt(self, text: str) -> str:
        """构建实体抽取提示"""
        prompt = f"""你是一个医疗信息抽取专家。请从以下医疗文本中抽取所有的医疗实体。

实体类型包括：{', '.join(self.ENTITY_TYPES)}

请严格按照以下JSON格式输出，确保JSON格式正确：
{{
    "entities": [
        {{"text": "实体文本", "type": "实体类型", "start": 起始位置, "end": 结束位置}}
    ]
}}

注意：
1. start和end是实体在原文中的字符位置（从0开始）
2. 实体类型必须是上述类型之一
3. 不要遗漏任何医疗相关实体

文本：{text}

输出："""
        return prompt
    
    def _build_relation_prompt(self, text: str, entities: List[Entity]) -> str:
        """构建关系抽取提示"""
        entity_list = [f"{e.text}({e.type})" for e in entities]
        
        prompt = f"""你是一个医疗关系抽取专家。请从文本中抽取以下实体之间的医疗关系。

已识别的实体：{', '.join(entity_list)}

关系类型包括：{', '.join(self.RELATION_TYPES)}

请严格按照以下JSON格式输出：
{{
    "relations": [
        {{"head": "头实体", "relation": "关系类型", "tail": "尾实体"}}
    ]
}}

注意：
1. 头实体和尾实体必须是已识别实体中的文本
2. 关系类型必须是上述类型之一
3. 只抽取文本中明确存在的关系

文本：{text}

输出："""
        return prompt
    
    def _build_event_prompt(self, text: str) -> str:
        """构建事件抽取提示"""
        prompt = f"""你是一个医疗事件抽取专家。请从文本中抽取医疗事件。

医疗事件类型包括：
1. 诊断事件：包含诊断时间、诊断结果、诊断方法等
2. 治疗事件：包含治疗时间、治疗方法、用药情况等
3. 检查事件：包含检查时间、检查项目、检查结果等
4. 手术事件：包含手术时间、手术名称、手术结果等

请严格按照以下JSON格式输出：
{{
    "events": [
        {{
            "type": "事件类型",
            "trigger": "触发词",
            "arguments": {{
                "时间": "xxx",
                "地点": "xxx",
                "参与者": "xxx",
                "其他参数": "xxx"
            }}
        }}
    ]
}}

文本：{text}

输出："""
        return prompt
    
    def _generate(self, prompt: str, max_length: int = 2048) -> str:
        """生成模型响应"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        if self.device != 'auto':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if 'chatglm' in self.model_name.lower():
                response, _ = self.model.chat(
                    self.tokenizer, 
                    prompt, 
                    history=[],
                    max_length=max_length
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 提取助手的回复
                response = response.split(prompt)[-1].strip()
        
        return response
    
    def _parse_entity_response(self, response: str, original_text: str) -> List[Entity]:
        """解析实体抽取响应"""
        entities = []
        
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning("No JSON found in response")
                return entities
                
            data = json.loads(json_match.group())
            
            for item in data.get('entities', []):
                # 验证实体是否在原文中
                entity_text = item['text']
                if entity_text in original_text:
                    # 查找实体位置
                    start = original_text.find(entity_text)
                    end = start + len(entity_text)
                    
                    entity = Entity(
                        text=entity_text,
                        type=item['type'],
                        start=start,
                        end=end
                    )
                    entities.append(entity)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing entity response: {e}")
            
        return entities
    
    def _parse_relation_response(self, response: str, entities: List[Entity]) -> List[Relation]:
        """解析关系抽取响应"""
        relations = []
        entity_texts = {e.text for e in entities}
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning("No JSON found in response")
                return relations
                
            data = json.loads(json_match.group())
            
            for item in data.get('relations', []):
                # 验证头尾实体是否在已识别实体中
                if (item['head'] in entity_texts and 
                    item['tail'] in entity_texts and
                    item['relation'] in self.RELATION_TYPES):
                    
                    relation = Relation(
                        head=item['head'],
                        relation=item['relation'],
                        tail=item['tail']
                    )
                    relations.append(relation)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing relation response: {e}")
            
        return relations
    
    def _parse_event_response(self, response: str) -> List[Dict]:
        """解析事件抽取响应"""
        events = []
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                logger.warning("No JSON found in response")
                return events
                
            data = json.loads(json_match.group())
            events = data.get('events', [])
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing event response: {e}")
            
        return events


class FewShotMedicalExtractor(MedicalLLMExtractor):
    """基于少样本学习的医疗信息抽取器"""
    
    def __init__(self, model_name: str = 'chatglm3-6b', device: str = 'auto'):
        super().__init__(model_name, device)
        
        # 少样本示例
        self.entity_examples = [
            {
                "text": "患者因发热、咳嗽3天就诊，诊断为上呼吸道感染，给予阿莫西林治疗。",
                "entities": [
                    {"text": "发热", "type": "症状"},
                    {"text": "咳嗽", "type": "症状"},
                    {"text": "上呼吸道感染", "type": "疾病"},
                    {"text": "阿莫西林", "type": "药物"}
                ]
            },
            {
                "text": "糖尿病患者常见症状包括多饮、多尿、多食，需要使用胰岛素治疗。",
                "entities": [
                    {"text": "糖尿病", "type": "疾病"},
                    {"text": "多饮", "type": "症状"},
                    {"text": "多尿", "type": "症状"},
                    {"text": "多食", "type": "症状"},
                    {"text": "胰岛素", "type": "药物"}
                ]
            }
        ]
        
        self.relation_examples = [
            {
                "text": "糖尿病患者常见症状包括多饮、多尿、多食。",
                "relations": [
                    {"head": "糖尿病", "relation": "症状表现", "tail": "多饮"},
                    {"head": "糖尿病", "relation": "症状表现", "tail": "多尿"},
                    {"head": "糖尿病", "relation": "症状表现", "tail": "多食"}
                ]
            }
        ]
    
    def _build_entity_prompt(self, text: str) -> str:
        """构建包含少样本示例的实体抽取提示"""
        examples_str = ""
        for i, example in enumerate(self.entity_examples, 1):
            examples_str += f"\n示例{i}：\n"
            examples_str += f"文本：{example['text']}\n"
            examples_str += f"输出：{json.dumps({'entities': example['entities']}, ensure_ascii=False, indent=2)}\n"
        
        prompt = f"""你是一个医疗信息抽取专家。请参考以下示例，从医疗文本中抽取实体。

实体类型包括：{', '.join(self.ENTITY_TYPES)}
{examples_str}

现在请处理以下文本：
文本：{text}

输出："""
        return prompt


# 使用示例
if __name__ == "__main__":
    # 初始化抽取器
    extractor = MedicalLLMExtractor(model_name='chatglm3-6b')
    
    # 测试文本
    test_text = """
    患者王某，男，55岁，因"反复胸闷、胸痛2年，加重1周"入院。
    患者2年前开始出现活动后胸闷、胸痛，休息后可缓解。
    1周前症状加重，伴有心悸、出汗。
    既往有高血压病史10年，糖尿病病史5年。
    入院后完善心电图、心脏彩超等检查，诊断为冠心病、不稳定型心绞痛。
    给予阿司匹林、氯吡格雷双抗治疗，辛伐他汀调脂，美托洛尔控制心率。
    """
    
    # 抽取信息
    results = extractor.extract(test_text)
    
    # 打印结果
    print("实体抽取结果：")
    for entity in results['entities']:
        print(f"  {entity.text} ({entity.type}) [{entity.start}:{entity.end}]")
    
    print("\n关系抽取结果：")
    for relation in results['relations']:
        print(f"  {relation.head} --{relation.relation}--> {relation.tail}")