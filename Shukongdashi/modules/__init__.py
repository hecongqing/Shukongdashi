
"""Teaching-friendly modular wrappers for the Shukongdashi project.

This package regroup the core capabilities of the platform into clear, easy-to-demo
sub-modules:

1. entity_extraction  – 从文本中识别实体
2. relation_extraction – 抽取实体之间的语义关系
3. kg_builder          – 构建 / 更新 Neo4j 知识图谱
4. qa                  – 基于知识图谱的问答

These wrappers keep the original heavy implementation untouched while providing a
simple and stable API surface that can be reused in live coding sessions or
technical interviews.
"""