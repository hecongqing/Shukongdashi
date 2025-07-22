
"""Question-Answering (QA) façade built on top of the Neo4j knowledge graph.

The original implementation in `Shukongdashi.demo.question_wenda` mixes HTTP
handling, Chinese regex rules and graph traversal logic – this wrapper isolates
ONLY the last part so that the algorithm can be reused in notebooks, unit tests
or new web frameworks.

The public API is a single `answer(question: str) -> list[str]` function.
"""
from __future__ import annotations

import re
from typing import List

from Shukongdashi.demo.question_wenda import huida, pattern  # re-use proven logic

# Build an index mapping each regex group in *pattern* to its corresponding type
_PATTERN_INDEX: list[tuple[re.Pattern, int]] = []
for i, group in enumerate(pattern):
    for raw in group:
        _PATTERN_INDEX.append((re.compile(raw), i))


def _classify(question: str) -> tuple[int, str]:
    """Return `(type_id, subject)` where *subject* is the question focus word."""
    pos = -1
    q_type = -1
    for regex, idx in _PATTERN_INDEX:
        m = regex.search(question)
        if m:
            pos = m.span()[0]
            q_type = idx
            break
    return q_type, question[:pos] if pos != -1 else question


def answer(question: str) -> List[str]:
    """Return a list of answers for *question* using the underlying KG."""
    q_type, subject = _classify(question)
    if q_type == -1:
        return []  # Unsupported yet.
    return huida(q_type, subject)


__all__ = ["answer"]