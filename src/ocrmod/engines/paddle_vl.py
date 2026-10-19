"""PaddleOCR-VL: layout detection plus VL recognition -> Markdown with HTML tables.

Runs in the WSL2 venv on the GPU PC (docs/setup-gpu.md). The paddleocr API has moved
between releases; scripts/gpu_check.py exercises this wrapper against the installed one.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from ..schemas import IngestedDoc
from .base import free_paddle_gpu, paddle_memory_mb, reset_paddle_peak

log = logging.getLogger(__name__)


class PaddleVLEngine:
    name = "paddle_vl"

    def __init__(self, predict_kwargs: dict | None = None, **kwargs):
        self.kwargs = kwargs
        self.predict_kwargs = predict_kwargs or {}
        self._pipeline = None
        self.memory: dict[str, float | None] = {}

    def _load(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL  # type: ignore

            self._pipeline = PaddleOCRVL(**self.kwargs)
            # Loading leaves temporary copies of the weights on the GPU (paddlex reads the
            # file onto the GPU, then converts it); hand that memory back.
            free_paddle_gpu()
            self.memory["load_peak_mb"] = paddle_memory_mb()[1]
            self.memory["after_load_mb"] = paddle_memory_mb()[0]
            reset_paddle_peak()
        return self._pipeline

    def parse(self, doc: IngestedDoc) -> list[str]:
        pipeline = self._load()
        pages = []
        for n, img in enumerate(doc.page_images, start=1):
            t0 = time.time()
            res = next(iter(pipeline.predict(str(img), **self.predict_kwargs)))
            pages.append(_markdown_of(res))
            log.warning("paddle_vl page %d/%d: %.1fs", n, len(doc.page_images), time.time() - t0)
        self.memory["reading_peak_mb"] = paddle_memory_mb()[1]
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
