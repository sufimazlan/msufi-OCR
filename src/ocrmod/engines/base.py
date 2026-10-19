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


def paddle_memory_mb() -> tuple[float | None, float | None]:
    """(currently allocated, peak allocated) by Paddle on the GPU, in MB."""
    try:
        import paddle  # type: ignore

        cuda = paddle.device.cuda
        return cuda.memory_allocated() / 2**20, cuda.max_memory_allocated() / 2**20
    except Exception:
        return None, None


def reset_paddle_peak() -> None:
    try:
        import paddle  # type: ignore

        paddle.device.cuda.reset_max_memory_allocated()
    except Exception:
        pass
