from __future__ import annotations

"""NLP service module
A very light-weight implementation that covers the steps shown in the architecture diagram:
 1) sentence segmentation
 2) sentence classification (equipment info / user operation / fault symptom)
 3) simple entity extraction (currently rule-based, can be swapped for real NER later)

This module purposefully keeps external dependencies minimal so it can run even if transformers
or other heavy libraries are not yet installed. Once a proper model is ready, simply replace the
methods `classify_sentence` and `ner_extract` with model inference code.
"""

from typing import List, NamedTuple
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple entity representation used inside the service. In later iterations it
# can be replaced by the Pydantic models located in `backend.models.entity_models`.
# ---------------------------------------------------------------------------

class Entity(NamedTuple):
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# NLPService – public API
# ---------------------------------------------------------------------------

class NLPService:
    """Light-weight text processing helper.

    The class offers three capabilities that map to the left半 of the diagram:
    1. sentence_split – segmentation
    2. classify_sentence – determine sentence role
    3. ner_extract – rule / model based entity extraction
    """

    EQUIPMENT_PATTERN = re.compile(r"(FANUC|SIEMENS|KUKA|\d+[TD])", re.IGNORECASE)
    OPERATION_PATTERN = re.compile(r"(按下|点击|启动|停止|更换|调整)")
    ALARM_CODE_PATTERN = re.compile(r"[A-Z]{2,}\d{3,}")

    def __init__(self) -> None:
        # Heavy model loading can be placed here later. Keep fast for now.
        logger.debug("NLPService initialised (rule-based mode)")

    # ---------------------------------------------------------------------
    # Basic utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def sentence_split(text: str) -> List[str]:
        """Split text into sentences respecting Chinese punctuation."""
        # Remove duplicate newlines, then split.
        text = text.replace("\r", "").strip()
        sentences = re.split(r"[。！？!?\n]+", text)
        # Filter empty pieces and normalise whitespace.
        return [s.strip() for s in sentences if s.strip()]

    # ---------------------------------------------------------------------
    # Sentence classification
    # ---------------------------------------------------------------------
    def classify_sentence(self, sentence: str) -> str:
        """Very naïve rule-based classification.

        Returns one of:
        • "EQUIPMENT"
        • "OPERATION"
        • "SYMPTOM"  (default)
        """
        if self.EQUIPMENT_PATTERN.search(sentence):
            return "EQUIPMENT"
        if self.OPERATION_PATTERN.search(sentence):
            return "OPERATION"
        return "SYMPTOM"

    # ---------------------------------------------------------------------
    # Entity extraction (placeholder)
    # ---------------------------------------------------------------------
    def ner_extract(self, sentence: str) -> List[Entity]:
        """Extract ALARM_CODE & other simple mentions from a sentence.

        For now only extracts alarm codes; extend with real NER later.
        """
        entities: List[Entity] = []
        for m in self.ALARM_CODE_PATTERN.finditer(sentence):
            entities.append(Entity(
                text=m.group(),
                entity_type="ALARM_CODE",
                start_pos=m.start(),
                end_pos=m.end(),
                confidence=0.9,
            ))
        return entities

    # ---------------------------------------------------------------------
    # Batch helpers – convenience wrappers
    # ---------------------------------------------------------------------
    def batch_classify(self, sentences: List[str]) -> List[str]:
        return [self.classify_sentence(s) for s in sentences]

    def batch_ner(self, sentences: List[str]) -> List[List[Entity]]:
        return [self.ner_extract(s) for s in sentences]