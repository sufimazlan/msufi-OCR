"""PaddleOCR-VL: layout detection plus VL recognition -> Markdown with HTML tables.

Runs in the WSL2 venv on the GPU PC (docs/setup-gpu.md). The paddleocr API has moved
between releases; scripts/gpu_check.py exercises this wrapper against the installed one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..schemas import IngestedDoc
from .base import free_paddle_gpu


class PaddleVLEngine:
    name = "paddle_vl"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL  # type: ignore

            self._pipeline = PaddleOCRVL(**self.kwargs)
        return self._pipeline

    def parse(self, doc: IngestedDoc) -> list[str]:
        pipeline = self._load()
        pages = []
        for img in doc.page_images:
            res = next(iter(pipeline.predict(str(img))))
            pages.append(_markdown_of(res))
        return pages

    def close(self) -> None:
        self._pipeline = None
        free_paddle_gpu()


def _markdown_of(res) -> str:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict) and md.get("markdown_texts"):
        return md["markdown_texts"]
    if isinstance(md, str):
        return md
    # Fallback: let the result write itself out, then read it back.
    with tempfile.TemporaryDirectory() as tmp:
        res.save_to_markdown(save_path=tmp)
        files = sorted(Path(tmp).rglob("*.md"))
        return "\n\n".join(f.read_text(encoding="utf-8") for f in files)
