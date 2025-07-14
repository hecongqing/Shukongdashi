"""High-level rule-based extraction for entities & simple relations."""
from __future__ import annotations

from typing import Dict, List

from scripts.ie.rules import extract_entities_rule, Entity

# Placeholder simple relation logic

def extract_relations_simple(entities: List[Entity]) -> List[Dict]:
    """Very naive heuristic: if both Equipment and Component in sentence, link has_component."""
    eqs = [e for e in entities if e[1] == "Equipment"]
    comps = [e for e in entities if e[1] == "Component"]
    rels: List[Dict] = []
    for h in eqs:
        for t in comps:
            rels.append({"head": h[0], "tail": t[0], "type": "has_component"})
    return rels


def extract(text: str) -> Dict:
    """Return dict {'entities': [...], 'relations': [...]}"""
    ents = extract_entities_rule(text)
    rels = extract_relations_simple(ents)
    return {"entities": ents, "relations": rels}