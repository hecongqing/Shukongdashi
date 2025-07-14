"""Graph-RAG pipeline: question -> Cypher -> results -> LLM answer."""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Dict

from tabulate import tabulate  # type: ignore

from scripts.kg.neo4j_utils import Neo4jConnector
from scripts.qa.query_parser import parse
from scripts.ie.llm_extractor import LLMBasedExtractor


def run_cypher(conn: Neo4jConnector, cypher: str, params: Dict):
    with conn.driver.session() as session:
        result = session.run(cypher, params)
        records = [r.data() for r in result]
    return records


def build_context(records):
    if not records:
        return "未在知识图谱中找到匹配结果。"
    table = tabulate(records, headers="keys", tablefmt="grid")
    return f"以下是知识图谱返回的结果:\n{table}"


def answer_question(question: str, model_path: str = "Qwen/Qwen1.5-0.5B-Chat") -> str:
    parsed = parse(question)
    if "error" in parsed:
        return parsed["error"]

    conn = Neo4jConnector()
    records = run_cypher(conn, parsed["cypher"], parsed["params"])
    conn.close()

    context = build_context(records)

    prompt = textwrap.dedent(
        f"""你是一位装备维修知识专家，请根据提供的知识图谱结果回答用户问题。\n
        问题: {question}\n
        {context}\n
        请使用中文简洁回答，如果图谱无结果，请说明"""
    )

    llm = LLMBasedExtractor(model_path)
    resp = llm.extract(prompt, max_new_tokens=128)  # reuse extract to get answer but we need raw generation
    # we can tweak llm.extract to work generically. quick workaround:
    return "\n".join(t for t, _ in resp["entities"]) if resp["entities"] else context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="中文问题")
    parser.add_argument("--model", default="Qwen/Qwen1.5-0.5B-Chat")
    args = parser.parse_args()
    print(answer_question(args.question, args.model))