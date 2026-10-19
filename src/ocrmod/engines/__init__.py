"""Engine factory: builds engines from settings by name."""

from __future__ import annotations

from ..config import Settings


def make_layout_engine(name: str, settings: Settings):
    if name == "paddle_vl":
        from .paddle_vl import PaddleVLEngine

        cfg = settings["paddle_vl"]
        return PaddleVLEngine(predict_kwargs=cfg.get("predict_kwargs", {}),
                              cpu_weight_conversion=cfg.get("cpu_weight_conversion", True),
                              **cfg["kwargs"])
    if name == "glm_ocr":
        from .glm_ocr import GLMOCREngine

        o = settings["ollama"]
        return GLMOCREngine(o["url"], o["glm_ocr_model"], o["glm_ocr_prompt"], o["timeout_seconds"])
    if name == "pdftext":
        from .pdftext import PdfTextEngine

        return PdfTextEngine()
    raise ValueError(f"Unknown layout engine {name!r}")


def make_line_engine(name: str, settings: Settings):
    if name == "none":
        return None
    if name == "ppocr":
        from .ppocr import PPOCREngine

        return PPOCREngine(**settings["ppocr"]["kwargs"])
    if name == "pdftext":
        from .pdftext import PdfTextEngine

        return PdfTextEngine()
    raise ValueError(f"Unknown line engine {name!r}")
