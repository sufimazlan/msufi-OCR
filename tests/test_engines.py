import sys
import types

from ocrmod.engines.paddle_vl import PaddleVLEngine
from ocrmod.schemas import IngestedDoc


class _Res:
    markdown = {"markdown_texts": "# INV-1\n<table><tr><td>1</td></tr></table>"}


def test_paddle_vl_wrapper_with_stand_in_module(monkeypatch, tmp_path):
    created, calls = [], []

    class FakeVL:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def predict(self, path, **kw):
            calls.append(kw)
            return iter([_Res()])

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCRVL=FakeVL))
    eng = PaddleVLEngine(predict_kwargs={"use_queues": False}, foo=1)
    doc = IngestedDoc(pdf_path=tmp_path / "x.pdf", doc_id="x",
                      page_images=[tmp_path / "p1.png", tmp_path / "p2.png"], dpi=250)
    pages = eng.parse(doc)
    assert pages == [_Res.markdown["markdown_texts"]] * 2
    assert created == [{"foo": 1}]  # loaded once, kwargs passed through
    assert calls == [{"use_queues": False}] * 2
    assert set(eng.memory) == {"load_peak_mb", "after_load_mb", "reading_peak_mb"}
    eng.close()
    assert eng._pipeline is None
