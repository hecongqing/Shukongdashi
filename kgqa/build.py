from typing import Iterable, List, Tuple

from py2neo import Graph

__all__ = ["KnowledgeGraphBuilder"]

class KnowledgeGraphBuilder:
    """Utility class that wraps around py2neo to bulk load triples into Neo4j.

    Parameters
    ----------
    uri : str
        The bolt / http uri, e.g. ``bolt://localhost:7687`` or ``http://localhost:7474``.
    user : str
        Neo4j username.
    password : str
        Neo4j password.
    """

    def __init__(self, uri: str = "http://localhost:7474", user: str = "neo4j", password: str = "password"):
        self.graph = Graph(uri, auth=(user, password))

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _merge_node(self, label: str, title: str):
        self.graph.run("MERGE (_:%s {title: $title})" % label, title=title)

    def _merge_relation(self, head: str, tail: str, rel_type: str, head_label: str, tail_label: str):
        cypher = (
            "MATCH (h:%s {title: $head}), (t:%s {title: $tail}) "
            "MERGE (h)-[:%s {type: $rel_type}]->(t)" % (head_label, tail_label, head_label[0] + tail_label[0])
        )
        self.graph.run(cypher, head=head, tail=tail, rel_type=rel_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_triples(self, triples: Iterable[Tuple[str, str, str, str, str]]):
        """Load a list/iterable of triples into the graph.

        Each triple item is expected to be ``(head, relation, tail, head_label, tail_label)``
        """
        for h, r, t, h_l, t_l in triples:
            self._merge_node(h_l, h)
            self._merge_node(t_l, t)
            self._merge_relation(h, t, r, h_l, t_l)

    # Convenience loader ------------------------------------------------
    def load_from_dataframe(self, df, head_col="head", rel_col="relation", tail_col="tail", head_label_col="head_label", tail_label_col="tail_label"):
        """Load triples from a pandas DataFrame."""
        triples = df[[head_col, rel_col, tail_col, head_label_col, tail_label_col]].itertuples(index=False, name=None)
        self.load_triples(triples)