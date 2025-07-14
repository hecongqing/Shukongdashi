"""Domain dictionaries & regex patterns for rule-based extraction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from scripts.data_collection.cleaning import clean_text

# 简易词典，可根据业务持续扩充 --------------------------------------------------
DICT_DIR = Path(__file__).parent / "dicts"
DICT_DIR.mkdir(exist_ok=True)

# default minimal sets; can be loaded from external txt
EQUIPMENT = ["机床", "加工中心", "机器人", "发动机"]
COMPONENT = ["主轴", "伺服电机", "丝杠", "轴承"]
SYMPTOM = ["振动异常", "温升过高", "卡滞", "异响"]
ACTIONS = ["更换", "调整", "校准", "清洗"]
FAULT_CODE_PATTERN = re.compile(r"\b[EA][0-9]{2,3}\b")  # E60, A123

# ----------------------------------------------------------------------
# Rule-based extractors
# ----------------------------------------------------------------------

def load_dict(name: str) -> List[str]:
    path = DICT_DIR / f"{name}.txt"
    if path.exists():
        return [line.strip() for line in path.open(encoding="utf-8") if line.strip()]
    return []

# Update lists if custom dict exists
EQUIPMENT += load_dict("equipment")
COMPONENT += load_dict("component")
SYMPTOM += load_dict("symptom")
ACTIONS += load_dict("action")

# compiled regexes for entity detection
EQUIP_RE = re.compile("|".join(map(re.escape, sorted(set(EQUIPMENT), key=len, reverse=True))))
COMP_RE = re.compile("|".join(map(re.escape, sorted(set(COMPONENT), key=len, reverse=True))))
SYMP_RE = re.compile("|".join(map(re.escape, sorted(set(SYMPTOM), key=len, reverse=True))))
ACT_RE = re.compile("|".join(map(re.escape, sorted(set(ACTIONS), key=len, reverse=True))))


Entity = Tuple[str, str]  # (text, type)


def extract_entities_rule(text: str) -> List[Entity]:
    """Find entities using dictionaries & regex."""
    text = clean_text(text)
    ents: List[Entity] = []

    for m in EQUIP_RE.finditer(text):
        ents.append((m.group(), "Equipment"))
    for m in COMP_RE.finditer(text):
        ents.append((m.group(), "Component"))
    for m in SYMP_RE.finditer(text):
        ents.append((m.group(), "Symptom"))
    for m in ACT_RE.finditer(text):
        ents.append((m.group(), "Action"))
    for m in FAULT_CODE_PATTERN.finditer(text):
        ents.append((m.group(), "FaultCode"))

    return ents