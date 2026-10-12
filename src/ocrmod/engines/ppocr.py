"""PP-OCRv5: the second engine. Gives text, confidence and a box per line."""

from __future__ import annotations

from ..schemas import IngestedDoc, OCRLine
from .base import free_paddle_gpu, result_get


class PPOCREngine:
    name = "ppocr"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._ocr = None

    def _load(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore

            self._ocr = PaddleOCR(**self.kwargs)
        return self._ocr

    def lines(self, doc: IngestedDoc) -> list[OCRLine]:
        ocr = self._load()
        out: list[OCRLine] = []
        for page, img in enumerate(doc.page_images):
            res = next(iter(ocr.predict(str(img))))
            texts = result_get(res, "rec_texts") or []
            scores = result_get(res, "rec_scores") or []
            polys = result_get(res, "rec_polys")
            if polys is None:
                polys = result_get(res, "dt_polys") or []
            for text, score, poly in zip(texts, scores, polys):
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                out.append(OCRLine(text=str(text), conf=float(score),
                                   bbox=(min(xs), min(ys), max(xs), max(ys)), page=page))
        return out

    def close(self) -> None:
        self._ocr = None
        free_paddle_gpu()
