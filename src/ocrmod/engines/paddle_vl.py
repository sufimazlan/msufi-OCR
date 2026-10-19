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

    def __init__(self, predict_kwargs: dict | None = None, cpu_weight_conversion: bool = True,
                 **kwargs):
        self.kwargs = kwargs
        self.predict_kwargs = predict_kwargs or {}
        self.cpu_weight_conversion = cpu_weight_conversion
        self._pipeline = None
        self.memory: dict[str, float | None] = {}

    def _load(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL  # type: ignore

            if self.cpu_weight_conversion:
                patch_weight_conversion()
            # Earlier Paddle models (orientation, PP-OCR) leave freed memory in Paddle's
            # reserve, which counts against the memory cap; hand it back before loading.
            free_paddle_gpu()
            self.memory["before_load_mb"] = paddle_memory_mb()[0]
            reset_paddle_peak()
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


def patch_weight_conversion() -> bool:
    """Convert PaddleOCR-VL's weights in CPU RAM instead of on the GPU.

    paddlex reads the weights file onto the GPU, then set_hf_state_dict() glues pieces
    together (paddle.concat) there, briefly holding several copies: a 5.9GB load peak for a
    1.9GB model. Moving the state dict to the CPU first makes the concat run in RAM; the
    original method already moves each converted tensor to the model's device itself
    (`state_dict[key].to(v1.place)`). Returns False if the class is not where we expect.
    """
    try:
        import paddle  # type: ignore
        from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import (  # type: ignore
            _paddleocr_vl as modeling,
        )
    except Exception as exc:
        log.warning("CPU weight conversion not applied (%s)", exc)
        return False
    cls = modeling.PaddleOCRVLForConditionalGeneration
    if getattr(cls, "_ocrmod_cpu_conversion", False):
        return True
    original = cls.set_hf_state_dict

    def set_hf_state_dict(self, state_dict, *args, **kwargs):
        for key in list(state_dict):
            value = state_dict[key]
            if isinstance(value, paddle.Tensor) and not value.place.is_cpu_place():
                state_dict[key] = value.cpu()
        value = None
        free_paddle_gpu()
        return original(self, state_dict, *args, **kwargs)

    cls.set_hf_state_dict = set_hf_state_dict
    cls._ocrmod_cpu_conversion = True
    return True


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
