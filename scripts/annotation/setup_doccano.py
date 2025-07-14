"""Bootstrap a Doccano project with ontology.

Usage:
    python setup_doccano.py --url http://localhost:8001 --username admin --password xxx \
        --schema configs/ontology_schema.json --project "Equipment Fault KG"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from doccano_client.client import DoccanoClient  # type: ignore


def load_schema(path: str | Path) -> tuple[List[Dict], List[Dict]]:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    return meta["entities"], meta["relations"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--schema", required=True, help="ontology_schema.json path")
    parser.add_argument("--project", default="Equipment Fault KG")
    args = parser.parse_args()

    entities, relations = load_schema(args.schema)

    client = DoccanoClient(base_url=args.url)
    client.login(args.username, args.password)

    # Create project
    project = client.post("/v1/projects", {
        "name": args.project,
        "description": "Equipment fault KG annotation project",
        "project_type": "SequenceLabelingWithRelation",
    })
    project_id = project["id"]
    print("Project id:", project_id)

    # Add labels (entities)
    for ent in entities:
        client.post(f"/v1/projects/{project_id}/labels", {
            "text": ent["name"],
            "background_color": ent["color"],
            "suffix_key": str(ent["id"]),
            "description": ent.get("description", ""),
        })

    # Add relations
    for rel in relations:
        client.post(f"/v1/projects/{project_id}/relations", {
            "type": rel["name"],
            "color": rel["color"],
            "description": rel.get("description", ""),
        })

    print("✅ Ontology imported successfully")


if __name__ == "__main__":
    main()