from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .qa import KnowledgeGraphQA

app = FastAPI(title="Equipment Fault Diagnosis QA", version="0.1.0")

kgqa = KnowledgeGraphQA()

class QueryIn(BaseModel):
    question: str

class QueryOut(BaseModel):
    query_parse: Dict[str, Any]
    answers: Any
    related_phenomena: Any

@app.post("/qa", response_model=QueryOut)
def qa_endpoint(in_payload: QueryIn):
    if not in_payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = kgqa.answer(in_payload.question)
    return result