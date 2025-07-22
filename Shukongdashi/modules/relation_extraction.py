
"""High-level relation extraction API.

This is a lightweight façade on top of the legacy implementation scattered
around `Shukongdashi.demo` *and* `Shukongdashi.test_my`.

The goal is to expose a **single call** that takes in raw text and returns a
list of relations `(head_entity, relation_type, tail_entity)` so that the output
can be directly ingested by the `kg_builder` module.

NOTE
----
The original codebase mixes data-base operations with NLP logic.  Refactoring it
properly would require a larger effort.  For the purpose of a teaching / demo
project we opt for an incremental approach:

1. Keep the heavy algorithms untouched.
2. Import the fastest-to-use helper – right now the *rule-based* splitter in
   `question_fenxi.py` – and post-process its result.
3. If the underlying implementation changes, only *this* thin wrapper needs to
   be updated.
"""
from __future__ import annotations

from typing import List, Tuple

# The legacy helper we rely on for now.
try:
    from Shukongdashi.demo.question_fenxi import analyse_sentence  # type: ignore
except ImportError:  # pragma: no cover – Safe-guard for CI
    analyse_sentence = None  # type: ignore

Relation = Tuple[str, str, str]


def extract(text: str) -> List[Relation]:
    """Extract semantic relations from *text*.

    If the optimised rule-based extractor is not available, the function will
    fall back to an **empty list** so that downstream code does not fail.
    """
    if analyse_sentence is None:
        # De-graded behaviour for environments where the heavy dependencies are
        # not compiled / downloaded.
        return []

    # The original `analyse_sentence` returns a dict keyed by *relation_type*
    # with values being a list of (head, tail) tuples.
    legacy_output = analyse_sentence(text)

    relations: List[Relation] = []
    for rel_type, pairs in legacy_output.items():
        for head, tail in pairs:
            relations.append((head, rel_type, tail))
    return relations


__all__ = ["extract", "Relation"]