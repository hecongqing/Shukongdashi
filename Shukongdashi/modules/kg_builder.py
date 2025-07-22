
"""Lightweight builder / updater for the Neo4j backed knowledge graph.

This wrapper depends on the `Neo4j` helper in `Shukongdashi.Model.neo_models` and
provides two high-level helpers:

1. `add_entities(entities)` – ensure the *entity* nodes exist.
2. `add_relations(relations)` – ensure the `(head)-[:REL]->(tail)` relation
   exists.  *relations* is expected to be an iterable of
   `(head_entity, relation_type, tail_entity)` tuples – exactly the output of
   `modules.relation_extraction.extract`.

Both functions are **idempotent** – running them multiple times will *not*
create duplicates, making them safe for educational live demos.
"""
from __future__ import annotations

from typing import Iterable, Tuple

from Shukongdashi.Model.neo_models import Neo4j

neo = Neo4j()
neo.connectDB()

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def add_entities(entities: Iterable[Tuple[str, str]]) -> None:
    """Insert *entities* into Neo4j.

    Parameters
    ----------
    entities: Iterable[Tuple[str, str]]
        Each tuple is `(entity_text, label)` where *label* should be a valid
        Neo4j node label (e.g. `Xianxiang`, `Yuanyin`, ...).
    """
    for entity, label in entities:
        neo.insertNode(entity, label)


def add_relations(relations: Iterable[Tuple[str, str, str]]) -> None:
    """Insert *relations* into Neo4j, creating nodes lazily if needed."""
    for head, rel_type, tail in relations:
        # Guess labels if not present – for demo purpose we fall back to a generic
        # label that already exists in the sample DB.
        label1 = label2 = "Describe"
        neo.insertRelation(head, rel_type, tail, label1, label2)


__all__ = ["add_entities", "add_relations", "neo"]