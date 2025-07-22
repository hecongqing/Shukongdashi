import re
from typing import Dict, List, Tuple

# To reuse the existing CNN model and NER utilities that are already shipped with the original project
try:
    from Shukongdashi.toolkit.NER import get_NE  # type: ignore
    from Shukongdashi.toolkit.pre_load import cnn_model  # type: ignore
except ModuleNotFoundError:
    # If the original modules are absent (e.g. when demonstrating standalone),
    # we keep the import optional and fall back to simple rule-based heuristics
    get_NE = None  # type: ignore
    cnn_model = None  # type: ignore


__all__ = [
    "extract_entities",
    "classify_sentences",
    "parse_fault_text",
]

def _simple_tokenize(text: str) -> List[str]:
    """Very light tokenizer for fallback mode."""
    # Split on punctuation and whitespace
    tokens = re.split(r"[，。.,;；\s]+", text)
    return [t for t in tokens if t]

def _fallback_extract(tokens: List[str]) -> List[Tuple[str, str]]:
    """A very naive rule-based extractor used when the heavyweight model is unavailable."""
    # For demo purpose, we mark any token that contains a digit as a fault code,
    # Chinese manufacturing domain often embeds parts like "电机", "模块", "组件" etc.
    # We use a small keyword list as an illustrative heuristic.
    PART_HINTS = ["电机", "轴", "模块", "电源", "油泵", "线路", "电路", "刀库", "伺服", "驱动", "主轴", "继电器"]

    res: List[Tuple[str, str]] = []
    for tok in tokens:
        if re.search(r"\d", tok):
            res.append((tok, "FaultCode"))
        elif any(h in tok for h in PART_HINTS):
            res.append((tok, "FaultPart"))
        else:
            res.append((tok, "Phenomenon"))
    return res

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entities from raw fault description text.

    Returns a list of (entity, label) tuples. The implementation first tries the
    pre-trained NER from the original project. If that is not available (e.g. in
    a minimal demo environment), it falls back to a simple rule-based method so
    that the pipeline can still run.
    """
    if get_NE is not None:
        # Use the trained THULAC based NER, its return format is [[word, label], …]
        result = get_NE(text)
        out: List[Tuple[str, str]] = []
        for word, label in result:
            if label == 0:
                continue  # skip non-entity tokens
            out.append((word, str(label)))
        return out

    # Fallback mode
    tokens = _simple_tokenize(text)
    return _fallback_extract(tokens)

def classify_sentences(sentences: List[str]) -> List[str]:
    """Classify each sentence into one of the predefined five labels.

    If the CNN classification model from the original project is available we
    call it; otherwise we return a default value so that the downstream logic
    can still work in demo mode.
    """
    if cnn_model is not None:
        return [cnn_model.predict(s) for s in sentences]  # type: ignore
    # Fallback: label all as "故障现象"
    return ["故障现象" for _ in sentences]

def parse_fault_text(text: str) -> Dict[str, List[str]]:
    """Pipeline that turns raw user input into a structured dict expected by the QA module.

    The function performs sentence segmentation, classification and entity
    extraction, returning a dictionary like::

        {
            "operations": [...],
            "phenomena": [...],
            "fault_codes": [...],
        }
    """
    # 1) sentence segmentation – very rough, split on punctuation
    sentences = [s for s in re.split(r"[。！!？?;；]", text) if s.strip()]

    # 2) classification
    labels = classify_sentences(sentences)

    operations: List[str] = []
    phenomena: List[str] = []
    parts: List[str] = []
    fault_codes: List[str] = []

    # 3) entity extraction per sentence
    for sent, label in zip(sentences, labels):
        ents = extract_entities(sent)
        for ent, ent_label in ents:
            if ent_label == "FaultCode" or "故障代码" in ent_label:
                fault_codes.append(ent)
            elif ent_label == "FaultPart" or "故障部位" in ent_label:
                parts.append(ent)
            elif label == "用户操作" or label == "操作" or label == "操作步骤":
                operations.append(ent)
            else:
                phenomena.append(ent)

    return {
        "operations": list(set(operations)),
        "phenomena": list(set(phenomena)),
        "fault_codes": list(set(fault_codes)),
        "parts": list(set(parts)),
    }