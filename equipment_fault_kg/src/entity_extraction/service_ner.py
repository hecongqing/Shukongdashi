from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from .ner_model import NERModel

# ------------------------------------------------------------------------
# Model bootstrap
# ------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["entity_extraction"]

ner_mgr = NERModel(cfg)
model_ckpt = PROJECT_ROOT / "models" / "best_ner_model.pth"
if model_ckpt.exists():
    ner_mgr.load_model(str(model_ckpt))
else:
    logger.warning("Model checkpoint not found. Make sure to train the model first.")

# ------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------
app = FastAPI(title="Equipment Fault – Entity Extraction Service", version="1.0.0")


class TextRequest(BaseModel):
    text: str


@app.post("/extract", summary="抽取实体", tags=["NER"])
def extract_entities(req: TextRequest):
    """Return list of extracted entities for the given text."""
    entities = ner_mgr.predict(req.text)
    return {"entities": entities}