"""Batch import JSONL extractions into Neo4j.

Expect each line: {"entities": [[text, type], ...], "relations": [{"head": ..., "tail": ..., "type": ...}]}"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.kg.neo4j_utils import Neo4jConnector


def collect_entities_relations(path: Path):
    entities: Dict[str, List[str]] = defaultdict(list)
    relations: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # rel_type -> (head, tail, labels?)

    with path.open(encoding="utf-8") as fr:
        for line in fr:
            data = json.loads(line)
            for text, typ in data.get("entities", []):
                entities[typ].append(text)
            for rel in data.get("relations", []):
                relations[rel["type"]].append((rel["head"], rel["tail"], rel.get("type")))
    return entities, relations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="extracted jsonl file path")
    args = parser.parse_args()

    path = Path(args.jsonl)
    ents, rels = collect_entities_relations(path)

    conn = Neo4jConnector()
    conn.create_constraints()

    for label, names in ents.items():
        conn.merge_entities(label, names)
    # map of relation type to (head_label, tail_label) can be hard-coded for now
    type_map = {
        "has_component": ("Equipment", "Component"),
        "part_of": ("Component", "Equipment"),
        "exhibits_symptom": ("Component", "Symptom"),
        "has_fault_code": ("Component", "FaultCode"),
        "caused_by": ("Symptom", "Cause"),
        "leads_to": ("Cause", "Symptom"),
        "resolved_by": ("Cause", "Action"),
        "action_on": ("Action", "Component"),
        "affects_parameter": ("Cause", "Parameter"),
        "parameter_of": ("Parameter", "Component"),
        "uses_material": ("Component", "Material"),
    }
    for rel_type, pairs in rels.items():
        if rel_type not in type_map:
            continue
        head_label, tail_label = type_map[rel_type]
        conn.merge_relation(head_label, tail_label, rel_type, [(h, t)[:2] for h, t, _ in pairs])

    conn.close()
    print("✅ Import finished")


if __name__ == "__main__":
    main()