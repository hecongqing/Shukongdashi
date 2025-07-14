"""Neo4j utility helpers."""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Iterable, Tuple

from neo4j import GraphDatabase, Driver

CFG_PATH = Path(__file__).parent.parent.parent / "configs/neo4j_config.yaml"


class Neo4jConnector:
    def __init__(self, cfg_path: str | Path = CFG_PATH):
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        self.driver: Driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))

    # ----------------------- schema -------------------------
    def create_constraints(self):
        cqls = [
            "CREATE CONSTRAINT equipment_name IF NOT EXISTS FOR (n:Equipment) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (n:Component) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT fault_code_name IF NOT EXISTS FOR (n:FaultCode) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT cause_name IF NOT EXISTS FOR (n:Cause) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT action_name IF NOT EXISTS FOR (n:Action) REQUIRE n.name IS UNIQUE"
        ]
        with self.driver.session() as session:
            for cql in cqls:
                session.run(cql)

    # ----------------------- import -------------------------
    def merge_entities(self, label: str, names: Iterable[str]):
        query = f"""
        UNWIND $rows AS name
        MERGE (n:{label} {{name: name}})
        """
        with self.driver.session() as session:
            session.run(query, rows=list(names))

    def merge_relation(self, head_label: str, tail_label: str, rel_type: str, pairs: Iterable[Tuple[str, str]]):
        query = f"""
        UNWIND $rows AS row
        MATCH (h:{head_label} {{name: row.head}})
        MATCH (t:{tail_label} {{name: row.tail}})
        MERGE (h)-[:{rel_type}]->(t)
        """
        rows = [{"head": h, "tail": t} for h, t in pairs]
        with self.driver.session() as session:
            session.run(query, rows=rows)

    def close(self):
        self.driver.close()