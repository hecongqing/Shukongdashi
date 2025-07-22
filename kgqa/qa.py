from typing import Dict, List

# ------------------------------------------------------------------
# We want the demo to run even without Neo4j / py2neo installed. If import
# fails we create a *very* small stub that mimics the handful of APIs we use
# so that the rest of the code can still execute (albeit returning empty
# results).
# ------------------------------------------------------------------
try:
    from py2neo import Graph  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – demo convenience
    class _DummyCursor(list):
        def data(self):  # py2neo Cursor.data() returns list[dict]
            return []

        def evaluate(self):
            return None

    class Graph:  # type: ignore
        def __init__(self, *_, **__):
            pass

        def run(self, *_, **__):
            return _DummyCursor()

    print("[kgqa] py2neo not available – running in stub mode. All KG queries will return empty results.")

from .extraction import parse_fault_text

__all__ = ["KnowledgeGraphQA"]


def _deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for i in items:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out

class KnowledgeGraphQA:
    """Light-weight question answering module built on top of a Neo4j graph.

    The logic in this demonstration version is deliberately simple so that it
    can be understood quickly in an interview / teaching scenario. For a fully
    fledged production system you would typically incorporate a ranking model
    and more sophisticated inference.
    """

    def __init__(self, uri: str = "http://localhost:7474", user: str = "neo4j", password: str = "password"):
        self.graph = Graph(uri, auth=(user, password))

    # ------------------------------------------------------------------
    # Core helpers – Cypher queries
    # ------------------------------------------------------------------
    def _search_by_fault_code(self, codes: List[str]):
        res = []
        for code in codes:
            data = self.graph.run(
                """
                MATCH (f:FaultCode {title: $code})-[:FJ]->(p:Phenomenon)<-[:CY]-(c:Cause)
                OPTIONAL MATCH (c)-[:FS]->(s:Solution)
                RETURN f.title as fault_code, p.title as phenomenon, c.title as cause, collect(s.title) as solutions
                """,
                code=code,
            ).data()
            res.extend(data)
        return res

    def _search_by_phenomenon(self, phenomena: List[str]):
        res = []
        for phe in phenomena:
            data = self.graph.run(
                """
                MATCH (p:Phenomenon {title: $phe})<-[:CY]-(c:Cause)
                OPTIONAL MATCH (c)-[:FS]->(s:Solution)
                RETURN p.title as phenomenon, c.title as cause, collect(s.title) as solutions
                """,
                phe=phe,
            ).data()
            res.extend(data)
        return res

    # Helper to retrieve additional phenomena that share the same root cause – this is used
    # for "unrevealed" fault inference so that the frontend can ask the user for
    # confirmation and thus improve reliability.
    def _infer_related_phenomena(self, causes: List[str], already_found: List[str]):
        """Given a list of cause names, return a *deduplicated* list of other phenomena
        that are linked to **any** of the causes while excluding the ones we already
        know are present in the user description.

        Cypher pattern utilised::

            (c:Cause)-[:CY]->(p:Phenomenon)
        """
        if not causes:
            return []

        # gather related phenomena for every cause individually – this keeps Cypher
        # simple and avoids injection problems
        related: List[str] = []
        for cause in causes:
            rows = self.graph.run(
                """
                MATCH (c:Cause {title: $cause})-[:CY]->(p:Phenomenon)
                RETURN p.title AS phenomenon
                """,
                cause=cause,
            ).data()
            related.extend([r["phenomenon"] for r in rows])

        # remove already reported ones and keep order
        return [p for p in _deduplicate_keep_order(related) if p not in already_found]

    # Fuzzy search as a very light-weight synonym / alias mechanism ----------------
    def _fuzzy_match_phenomenon(self, raw: str) -> str:
        """Try to map an *input* phenomenon string to the canonical one in KG.

        If an exact node title exists we simply return it. Otherwise we do a
        case-insensitive CONTAINS match and return the *first* hit. In a more
        sophisticated system you would use embedding similarity or a dedicated
        alias table; for a demo substring match works reasonably well.
        """
        # exact
        hit = self.graph.run("MATCH (p:Phenomenon {title: $t}) RETURN p.title AS t", t=raw).evaluate()
        if hit:
            return hit

        # substring – note LIMIT 1 for performance
        hit = self.graph.run(
            """
            MATCH (p:Phenomenon)
            WHERE toLower(p.title) CONTAINS toLower($t)
            RETURN p.title AS t LIMIT 1
            """,
            t=raw,
        ).evaluate()
        return hit or raw  # fall back to raw string if nothing helps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def answer(self, raw_question: str) -> Dict:
        """Main entry point used by external callers."""
        parsed = parse_fault_text(raw_question)

        # Normalise user-supplied phenomena so that we can have a higher hit-rate
        if parsed["phenomena"]:
            parsed["phenomena"] = _deduplicate_keep_order([
                self._fuzzy_match_phenomenon(p) for p in parsed["phenomena"]
            ])

        aggregates: List[Dict] = []

        # 1) exact match on fault codes (highest precision)
        if parsed["fault_codes"]:
            aggregates.extend(self._search_by_fault_code(parsed["fault_codes"]))

        # 2) fallback to phenomenon search
        if not aggregates and parsed["phenomena"]:
            aggregates.extend(self._search_by_phenomenon(parsed["phenomena"]))

        # ------------------------------------------------------------------
        # Inference of related / hidden phenomena
        # ------------------------------------------------------------------
        causes_in_results = [item.get("cause") for item in aggregates if item.get("cause")]
        related_ph = self._infer_related_phenomena(causes_in_results, parsed["phenomena"])

        # Post-processing – flatten duplicates
        unique_seen = set()
        cleaned: List[Dict] = []
        for item in aggregates:
            key = (item.get("phenomenon"), item.get("cause"))
            if key not in unique_seen:
                unique_seen.add(key)
                item["solutions"] = _deduplicate_keep_order(item.get("solutions", []))
                cleaned.append(item)

        return {
            "query_parse": parsed,
            "answers": cleaned or "未在图谱中检索到直接答案，请尝试修改描述或在线检索。",
            "related_phenomena": related_ph,
        }