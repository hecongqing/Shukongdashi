"""Common Cypher query templates for QA layer."""
from __future__ import annotations

TEMPLATES = {
    "故障查询": (
        "MATCH (e:Equipment)-[:exhibits_symptom]->(s:Symptom) "
        "WHERE e.name=$equipment AND s.name=$symptom "
        "OPTIONAL MATCH (s)-[:caused_by]->(c:Cause) "
        "RETURN e.name AS 设备, s.name AS 故障现象, c.name AS 可能原因"
    ),
    "维修建议": (
        "MATCH (c:Cause)<-[:caused_by]-(:Symptom)<-[:exhibits_symptom]-(:Component)<-[:has_component]-(e:Equipment) "
        "WHERE e.name=$equipment AND c.name=$cause "
        "OPTIONAL MATCH (c)-[:resolved_by]->(a:Action) "
        "RETURN a.name AS 维修措施"
    ),
    "参数影响": (
        "MATCH (c:Cause)-[:affects_parameter]->(p:Parameter) "
        "WHERE c.name=$cause RETURN p.name AS 受影响参数"
    ),
}