from typing import Any, Dict, List

from py2neo import Graph

__all__ = ["graph_to_visjs"]


def graph_to_visjs(graph: Graph, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """Export a subgraph into a JSON structure consumable by Vis.js or Neovis.js.

    Parameters
    ----------
    graph : Graph
        Active py2neo Graph connection.
    limit : int, optional
        Maximum number of relationships to fetch.
    """
    query = (
        "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT %d" % limit
    )
    data = graph.run(query).data()

    nodes_seen = {}
    edges = []

    for row in data:
        n = row["n"]
        m = row["m"]
        r = row["r"]

        for node in [n, m]:
            node_id = node.identity
            if node_id not in nodes_seen:
                nodes_seen[node_id] = {
                    "id": node_id,
                    "label": node["title"],
                    "group": list(node.labels)[0] if node.labels else "Entity",
                }

        edges.append({
            "from": n.identity,
            "to": m.identity,
            "label": r["type"] if "type" in r else "rel",
        })

    return {
        "nodes": list(nodes_seen.values()),
        "edges": edges,
    }