"""Equipment manuals PDF fetcher & extractor.

下载装备制造相关 PDF 手册/标准文件并抽取纯文本，供后续信息抽取使用。
后续可在 sample_urls 中填入待抓取的公开链接。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import requests


LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


class ManualFetcher:
    """下载 PDF 并抽取文本。"""

    def __init__(self, download_dir: str | Path = "data/raw/manuals") -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 网络请求
    # ---------------------------------------------------------------------
    @staticmethod
    def _sanitize_filename(url: str) -> str:
        # 去掉查询参数等
        name = url.split("/")[-1]
        name = re.sub(r"[?&#].*$", "", name)
        return name or "manual.pdf"

    def download_pdfs(self, urls: List[str]) -> None:
        """批量下载 PDF（若已存在则跳过）。"""
        for url in urls:
            filename = self._sanitize_filename(url)
            dest = self.download_dir / filename
            if dest.exists():
                LOGGER.info("Skip existing %s", dest.name)
                continue

            LOGGER.info("Downloading %s", url)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                LOGGER.info("Saved to %s (%.2f KB)", dest, dest.stat().st_size / 1024)
            except Exception as exc:
                LOGGER.error("Failed to download %s: %s", url, exc)

    # ------------------------------------------------------------------
    # 文本抽取
    # ------------------------------------------------------------------
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """使用 PyMuPDF 抽取整本 PDF 文本。"""
        doc = fitz.open(pdf_path)
        texts: list[str] = []
        for page in doc:  # type: ignore[assignment]
            texts.append(page.get_text())
        return "\n".join(texts)

    def batch_extract(self) -> None:
        """遍历 download_dir 下所有 PDF，抽取文本保存为同名 .txt。"""
        for pdf_file in self.download_dir.glob("*.pdf"):
            txt_file = pdf_file.with_suffix(".txt")
            if txt_file.exists():
                continue
            LOGGER.info("Extracting %s", pdf_file.name)
            try:
                txt = self.extract_text_from_pdf(pdf_file)
                txt_file.write_text(txt, encoding="utf-8")
            except Exception as exc:
                LOGGER.error("Failed to extract %s: %s", pdf_file, exc)


if __name__ == "__main__":
    # TODO: 将公开 PDF 链接放到这里或通过命令行参数传入。
    sample_urls: list[str] = [
        # "https://example.com/path/to/equipment_manual.pdf",
    ]

    fetcher = ManualFetcher()
    if sample_urls:
        fetcher.download_pdfs(sample_urls)
    fetcher.batch_extract()