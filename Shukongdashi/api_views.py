
"""Additional HTTP endpoints exposing the new teaching-friendly modules.

These endpoints are *stateless* and return JSON.  They can coexist with the
original URLs without breaking backward compatibility.

Routes
------
GET /api/entities?text=...
GET /api/relations?text=...
GET /api/qa?question=...
"""
import json
from typing import Any

from django.http import HttpRequest, HttpResponse, JsonResponse

from Shukongdashi.modules import entity_extraction, relation_extraction, qa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response(data: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(
        json.dumps({"code": status, "data": data}, ensure_ascii=False),
        content_type="application/json;charset=utf-8",
        status=status,
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def entities_endpoint(request: HttpRequest) -> HttpResponse:
    text = request.GET.get("text", "")
    if not text:
        return _response({"error": "Missing 'text' parameter"}, status=400)
    return _response(entity_extraction.extract(text))


def relations_endpoint(request: HttpRequest) -> HttpResponse:
    text = request.GET.get("text", "")
    if not text:
        return _response({"error": "Missing 'text' parameter"}, status=400)
    return _response(relation_extraction.extract(text))


def qa_endpoint(request: HttpRequest) -> HttpResponse:
    question = request.GET.get("question", "")
    if not question:
        return _response({"error": "Missing 'question' parameter"}, status=400)
    return _response(qa.answer(question))