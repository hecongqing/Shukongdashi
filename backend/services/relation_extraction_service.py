import re
from typing import List, Dict, Optional
from loguru import logger

from backend.config.settings import get_settings
from backend.models.relation_models import RelationResult, RelationType
from backend.utils.text_utils import TextPreprocessor

class RelationExtractor:
    """关系抽取器（当前实现主要基于规则，后续可接入深度学习模型）"""

    def __init__(self):
        self.settings = get_settings()
        self.text_preprocessor = TextPreprocessor()
        # 未来可加载深度学习模型: 自行在 settings.RELATION_MODEL_PATH 配置
        self._load_model()

        # 简单规则模式
        self.patterns: Dict[RelationType, List[str]] = {
            RelationType.CAUSES: [r"(?P<src>.+?)(?:导致|引起|造成)(?P<tgt>.+?)"],
            RelationType.REPAIRS: [r"(?P<src>.+?)(?:解决|修复|处理)(?P<tgt>.+?)"],
            RelationType.CONTAINS: [r"(?P<src>.+?)(?:包含|包括)(?P<tgt>.+?)"],
            RelationType.BELONGS_TO: [r"(?P<src>.+?)(?:属于|隶属)(?P<tgt>.+?)"],
        }

    def _load_model(self):
        """如果存在深度学习模型，此处进行加载。目前占位实现。"""
        model_path = self.settings.RELATION_MODEL_PATH
        # 仅记录日志，不做实际加载
        logger.info(f"[RelationExtractor] 使用规则抽取。若需模型，请放置至 {model_path}")

    def _apply_patterns(self, text: str) -> List[RelationResult]:
        """应用正则模式抽取关系"""
        relations: List[RelationResult] = []
        for rel_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    src = match.group("src").strip()
                    tgt = match.group("tgt").strip()
                    if not src or not tgt or len(src) > 40 or len(tgt) > 40:
                        # 简单过滤噪声
                        continue
                    relations.append(
                        RelationResult(
                            source_entity=src,
                            target_entity=tgt,
                            relation_type=rel_type,
                            confidence=0.8,
                        )
                    )
        return relations

    def extract_relations_rule(self, text: str) -> List[RelationResult]:
        """基于规则的关系抽取"""
        return self._apply_patterns(text)

    def extract_relations(self, text: str) -> List[RelationResult]:
        """统一入口：目前直接调用规则抽取，可并入模型抽取结果"""
        processed_text = self.text_preprocessor.clean_text(text)
        relations = self.extract_relations_rule(processed_text)
        # TODO: 合并模型与规则结果（若未来有模型）
        return relations

    # 预留 API
    def update_patterns(self, new_patterns: Dict[RelationType, List[str]]):
        """动态更新/扩展规则"""
        for rel_type, pats in new_patterns.items():
            if rel_type in self.patterns:
                self.patterns[rel_type].extend(pats)
            else:
                self.patterns[rel_type] = pats

class RelationExtractionService:
    """关系抽取服务，封装异步接口以便 FastAPI 调用"""

    def __init__(self):
        self.extractor = RelationExtractor()

    async def extract_relations(self, text: str) -> List[RelationResult]:
        return self.extractor.extract_relations(text)

    async def batch_extract_relations(self, texts: List[str]) -> List[List[RelationResult]]:
        results: List[List[RelationResult]] = []
        for txt in texts:
            results.append(self.extractor.extract_relations(txt))
        return results

    async def update_relation_rules(self, rules: Dict[RelationType, List[str]]):
        self.extractor.update_patterns(rules)