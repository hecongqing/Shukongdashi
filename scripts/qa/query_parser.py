"""Parse natural language question to Cypher using simple heuristics + templates."""
from __future__ import annotations

import re
from typing import Dict

from scripts.ie.rules import extract_entities_rule
from scripts.kg.cypher_templates import TEMPLATES


def classify_question(question: str) -> str | None:
    """Return template key based on keywords."""
    q = question.lower()
    if any(k in q for k in ["原因", "什么导致", "什么原因"]):
        return "故障查询"
    if any(k in q for k in ["维修", "怎么处理", "如何处理", "措施", "修复"]):
        return "维修建议"
    if "参数" in q or "影响" in q:
        return "参数影响"
    return None


def parse(question: str) -> Dict:
    """Return dict with cypher and params or error."""
    template_key = classify_question(question)
    if not template_key:
        return {"error": "无法识别问题类型"}

    entities = extract_entities_rule(question)
    ent_map = {typ: text for text, typ in entities}

    if template_key == "故障查询":
        if "Equipment" not in ent_map or "Symptom" not in ent_map:
            return {"error": "缺少设备或故障现象实体"}
        cypher = TEMPLATES[template_key]
        params = {"equipment": ent_map["Equipment"], "symptom": ent_map["Symptom"]}
    elif template_key == "维修建议":
        if "Equipment" not in ent_map or "Cause" not in ent_map:
            return {"error": "缺少设备或原因实体"}
        cypher = TEMPLATES[template_key]
        params = {"equipment": ent_map["Equipment"], "cause": ent_map["Cause"]}
    elif template_key == "参数影响":
        if "Cause" not in ent_map:
            return {"error": "缺少原因实体"}
        cypher = TEMPLATES[template_key]
        params = {"cause": ent_map["Cause"]}
    else:
        return {"error": "未支持的问题类型"}

    return {"cypher": cypher, "params": params, "template": template_key}