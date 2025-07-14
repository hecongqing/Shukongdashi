"""Web scraping utilities for maintenance forums and technical docs."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup  # type: ignore

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class ForumScraper:
    """通用论坛分页爬虫（适配基于 Discuz / 简单分页参数的网站）。"""

    def __init__(
        self,
        base_url: str,
        page_param: str = "page",
        headers: Optional[dict[str, str]] = None,
        delay: float = 1.0,
    ) -> None:
        self.base_url = base_url
        self.page_param = page_param
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        self.delay = delay

    def fetch_page(self, page: int) -> str:
        url = re.sub(rf"[?&]{self.page_param}=\d+", "", self.base_url)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{self.page_param}={page}"
        LOGGER.debug("GET %s", url)
        resp = requests.get(url, headers=self.headers, timeout=20)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def parse_posts(html: str) -> List[dict[str, str]]:
        """简单示例：提取标题 + 内容。需根据站点自行修改。"""
        soup = BeautifulSoup(html, "lxml")
        posts: List[dict[str, str]] = []
        for item in soup.select("div.post"):
            title = item.select_one("h2")
            content = item.select_one("div.content")
            if not title or not content:
                continue
            posts.append({"title": title.get_text(strip=True), "content": content.get_text("\n", strip=True)})
        return posts

    def crawl(self, pages: int = 5, output: str | Path = "data/raw/forum_posts.jsonl") -> None:
        outfile = Path(output)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with outfile.open("a", encoding="utf-8") as fw:
            for p in range(1, pages + 1):
                LOGGER.info("Crawling page %d", p)
                try:
                    html = self.fetch_page(p)
                    posts = self.parse_posts(html)
                    for post in posts:
                        fw.write(json.dumps(post, ensure_ascii=False) + "\n")
                except Exception as exc:
                    LOGGER.error("Failed page %d: %s", p, exc)
                time.sleep(self.delay)


class TechDocScraper:
    """下载公开技术 HTML 文档或 PDF。"""

    def __init__(self, save_dir: str | Path = "data/raw/tech_docs") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> Path | None:
        filename = url.split("/")[-1]
        dest = self.save_dir / filename
        if dest.exists():
            LOGGER.info("Skip existing %s", filename)
            return dest
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            LOGGER.info("Downloaded %s", filename)
            return dest
        except Exception as exc:
            LOGGER.error("Failed to download %s: %s", url, exc)
            return None