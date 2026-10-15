"""Engine interfaces. Each engine loads its model lazily and frees it in close(),
so only one model needs to sit in the 8GB of VRAM at a time."""

from __future__ import annotations

from typing import Protocol

from ..schemas import IngestedDoc, OCRLine


class LayoutEngine(Protocol):
    """Page images -> Markdown (tables as HTML), one string per page."""

    name: str

    def parse(self, doc: IngestedDoc) -> list[str]: ...

    def close(self) -> None: ...


class LineEngine(Protocol):
    """Page images -> text lines with a confidence score and bounding box."""

    name: str

    def lines(self, doc: IngestedDoc) -> list[OCRLine]: ...

    def close(self) -> None: ...


def free_paddle_gpu() -> None:
    import gc

    gc.collect()  # drop the model objects first, or empty_cache has nothing to release
    try:
        import paddle  # type: ignore

        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def result_get(res, key: str):
    """PaddleOCR results behave like dicts in some versions and expose .json in others."""
    try:
        return res[key]
    except (KeyError, TypeError):
        data = getattr(res, "json", None) or {}
        return data.get("res", data).get(key)
