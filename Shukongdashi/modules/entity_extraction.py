
"""High-level entity extraction API.

This file wraps the low-level implementation that lives in
`Shukongdashi.toolkit.NER`, exposing a single `extract` function that can be
used programmatically or imported inside Django views / notebooks.

Usage
-----
>>> from Shukongdashi.modules.entity_extraction import extract
>>> text = "自动换刀时刀链运转不到位，刀库停止运转"
>>> extract(text)
[('自动换刀', 'Xianxiang'), ('刀链', 'GuzhangBuwei'), ...]

The return value is a list of 2-tuples (entity_text, entity_type).  The
*entity_type* is mapped to a human-readable string via the helper mapping at the
bottom of this module.
"""

from typing import List, Tuple

from Shukongdashi.toolkit.NER import get_NE

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# NOTE: Keep a lightweight mapping here so that consumers do not have to rely on
# the numeric / English labels returned by the original implementation.
_LABEL_MAP = {
    1: "人物(Person)",
    2: "地点(Location)",
    3: "机构(Organization)",
    4: "政治经济名词(Politics/Economy)",
    5: "动物学名词(Zoology)",
    6: "植物学名词(Botany)",
    7: "化学名词(Chemistry)",
    8: "季节气候(Season/Climate)",
    9: "动植物产品(Product)",
    10: "动植物疾病(Disease)",
    11: "自然灾害(Disaster)",
    12: "营养成分(Nutrition)",
    13: "生物学名词(Biology)",
    14: "农机具(Equipment)",
    15: "农业技术术语(AgriTech)",
    16: "其它实体(Other)",
    # pass-through for english tags – keep as-is for now
    "np": "人物(Person)",
    "ns": "地点(Location)",
    "ni": "机构(Organization)",
    "nz": "专业名词(FieldTerm)",
    "i": "习语(Idiom)",
    "id": "习语(Idiom)",
    "j": "简称(Abbreviation)",
    "x": "其它(Other)",
    "t": "时间(Time)",
}


def extract(text: str) -> List[Tuple[str, str]]:
    """Return all recognised entities inside *text*.

    Parameters
    ----------
    text: str
        The input sentence / paragraph.

    Returns
    -------
    List[Tuple[str, str]]
        A list of `(entity, entity_type)` tuples.  The *entity_type* is a human
        readable string as defined in `_LABEL_MAP`.  Un-recognised tokens are
        filtered out.
    """
    raw_result = get_NE(text)
    entities: List[Tuple[str, str]] = []
    for entity, label in raw_result:
        if not label:  # label == 0 in original code => non-entity
            continue
        entities.append((entity, _LABEL_MAP.get(label, str(label))))
    return entities


__all__ = ["extract"]