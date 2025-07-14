from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from .train_relation import RELATION_LABELS, LABEL2ID, ID2LABEL, SEPARATOR_TOKEN

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
MODEL_DIR = PROJECT_ROOT / "models" / "relation_classifier"

if not MODEL_DIR.exists():
    logger.warning("Relation classifier not found. Make sure to train the model first!")

model_name = "bert-base-chinese"
try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        model_name = yaml.safe_load(f)["relation_extraction"]["model"].get("name", model_name)
except Exception as e:
    logger.error(f"Failed reading config: {e}")

# load tokenizer always from base model (to avoid missing) & model weights from saved dir if present
_tokenizer = BertTokenizer.from_pretrained(model_name)
_model_path = MODEL_DIR if MODEL_DIR.exists() else model_name
_model = BertForSequenceClassification.from_pretrained(
    _model_path,
    num_labels=len(RELATION_LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
_model.eval()

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="Equipment Fault – Relation Extraction Service", version="1.0.0")


class RelationRequest(BaseModel):
    text: str
    subject: str
    object: str
    top_k: Optional[int] = 1


@app.post("/predict", tags=["Relation"], summary="预测关系类型")
def predict_rel(req: RelationRequest):
    # build input sequence: subj [SEP] obj [SEP] context
    input_text = f"{req.subject} {SEPARATOR_TOKEN} {req.object} {SEPARATOR_TOKEN} {req.text}"
    inputs = _tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    # return top_k predictions
    topk = req.top_k or 1
    values, indices = torch.topk(probs, k=topk)
    results = [
        {"relation": ID2LABEL[idx.item()], "confidence": float(values[i])}
        for i, idx in enumerate(indices)
    ]
    return {"predictions": results}