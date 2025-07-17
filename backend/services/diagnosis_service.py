from __future__ import annotations

"""Diagnosis service – orchestrates the end-to-end workflow described in the diagram.

步骤：
1. 使用 NLPService 将用户输入分句、分类、抽取简单实体。
2. 聚焦分类为 SYMPTOM 的句子，调用 KBService 搜相似 *标准现象*。
3. 基于最佳匹配的标准现象，获取故障原因 & 解决方案。
4. 组装并返回诊断结果字典；后续可以替换为 Pydantic 模型。
"""

from typing import Dict, Any, List, Tuple
import uuid
import logging

from backend.services.nlp_service import NLPService
from backend.services.kb_service import KBService

logger = logging.getLogger(__name__)


class DiagnosisService:
    """High-level service that provides `diagnose()` callable."""

    def __init__(self):
        self._nlp = NLPService()
        self._kb = KBService()
        logger.debug("DiagnosisService initialised")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def diagnose(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one-shot diagnosis.

        Parameters
        ----------
        request : dict – expected keys:
            • fault_description : str  (mandatory)
            • equipment_brand / equipment_model / etc. (optional)
        Returns
        -------
        dict – result with causes / solutions / scores.
        """
        description: str = request.get("fault_description", "").strip()
        if not description:
            raise ValueError("fault_description is required for diagnosis")

        # 1) NLP processing -------------------------------------------------
        sentences: List[str] = self._nlp.sentence_split(description)
        logger.debug("%d sentence(s) detected", len(sentences))

        symptom_sents: List[str] = [
            s for s in sentences if self._nlp.classify_sentence(s) == "SYMPTOM"
        ]
        if not symptom_sents:
            symptom_sents = sentences  # fallback – use all sentences

        # Extract entities (currently not used further but returned for debug)
        entities = [entity for s in sentences for entity in self._nlp.ner_extract(s)]

        # 2) KB search ------------------------------------------------------
        best_matches: List[Tuple[str, float]] = []
        for sent in symptom_sents:
            best_matches.extend(self._kb.search_standard_symptom(sent, topk=1))

        best_matches.sort(key=lambda x: x[1], reverse=True)
        matched_symptom, score = (best_matches[0] if best_matches else (None, 0.0))

        # 3) Fetch causes & solutions --------------------------------------
        kb_info = self._kb.fetch_causes_solutions(matched_symptom) if matched_symptom else {}

        # 4) Assemble result -----------------------------------------------
        result = {
            "diagnosis_id": uuid.uuid4().hex,
            "matched_symptom": matched_symptom,
            "similarity_score": score,
            "extracted_entities": [e._asdict() for e in entities],
            "fault_causes": kb_info.get("causes", []),
            "repair_solutions": kb_info.get("solutions", []),
        }
        return result