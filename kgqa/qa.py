from typing import Dict, List

from py2neo import Graph

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def answer(self, raw_question: str) -> Dict:
        """Main entry point used by external callers."""
        parsed = parse_fault_text(raw_question)

        aggregates: List[Dict] = []

        # 1) exact match on fault codes (highest precision)
        if parsed["fault_codes"]:
            aggregates.extend(self._search_by_fault_code(parsed["fault_codes"]))

        # 2) fallback to phenomenon search
        if not aggregates and parsed["phenomena"]:
            aggregates.extend(self._search_by_phenomenon(parsed["phenomena"]))

        # NOTE: For a demo we skip operation-based reasoning.

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
        }