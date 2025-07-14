"""OCR utilities using PaddleOCR.

提供 PDF、图片的 OCR 提取能力，兼容扫描件手册、维修工单等。
依赖：paddleocr>=2.7.0, fitz(PyMuPDF)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from paddleocr import PaddleOCR  # type: ignore

LOGGER = logging.getLogger(__name__)


class OCRExtractor:
    """封装 PaddleOCR 调用，仅初始化一次。"""

    def __init__(self, lang: str = "ch") -> None:
        # use_angle_cls=True 对倾斜文本更友好
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    # ------------------------------------------------------------------
    # Image & PDF helpers
    # ------------------------------------------------------------------
    def ocr_image(self, img_path: str | Path) -> str:
        """Run OCR for one image file."""
        result = self.ocr.ocr(str(img_path), cls=True)
        texts: List[str] = []
        for line in result[0]:
            texts.append(line[1][0])  # line format: ((box), (text, score))
        return "\n".join(texts)

    def ocr_pdf(self, pdf_path: str | Path, dpi: int = 200) -> str:
        """Convert each page to pixmap then OCR."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        page_texts: List[str] = []
        for page in doc:  # type: ignore[assignment]
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            result = self.ocr.ocr(img_bytes, cls=True)
            for line in result[0]:
                page_texts.append(line[1][0])
        return "\n".join(page_texts)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def batch_ocr_pdf(self, pdf_dir: str | Path, suffix: str = "_ocr.txt") -> None:
        """遍历目录中的 PDF，生成 OCR 文本。

        如果同名 txt 已存在则跳过。
        """
        pdf_dir = Path(pdf_dir)
        for pdf_file in pdf_dir.glob("*.pdf"):
            out_file = pdf_file.with_suffix(suffix)
            if out_file.exists():
                continue
            LOGGER.info("OCR %s", pdf_file.name)
            try:
                txt = self.ocr_pdf(pdf_file)
                out_file.write_text(txt, encoding="utf-8")
            except Exception as exc:
                LOGGER.error("Failed OCR %s: %s", pdf_file, exc)