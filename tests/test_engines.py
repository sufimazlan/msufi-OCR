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


def test_patch_weight_conversion_moves_state_dict_to_cpu(monkeypatch):
    from ocrmod.engines import paddle_vl

    class Place:
        def __init__(self, cpu):
            self.cpu = cpu

        def is_cpu_place(self):
            return self.cpu

    class Tensor:
        def __init__(self, name, cpu=False):
            self.name, self.place = name, Place(cpu)

        def cpu(self):
            return Tensor(self.name, cpu=True)

    seen = {}

    class Model:
        def set_hf_state_dict(self, state_dict):
            seen.update({k: v.place.is_cpu_place() if isinstance(v, Tensor) else v
                         for k, v in state_dict.items()})
            return "loaded"

    fake_paddle = types.SimpleNamespace(Tensor=Tensor)
    modeling = types.SimpleNamespace(PaddleOCRVLForConditionalGeneration=Model)
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    for name in ("paddlex", "paddlex.inference", "paddlex.inference.models",
                 "paddlex.inference.models.doc_vlm", "paddlex.inference.models.doc_vlm.modeling"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    pkg = types.ModuleType("paddlex.inference.models.doc_vlm.modeling.paddleocr_vl")
    pkg._paddleocr_vl = modeling
    monkeypatch.setitem(sys.modules, pkg.__name__, pkg)
    monkeypatch.setattr(paddle_vl, "free_paddle_gpu", lambda: None)

    assert paddle_vl.patch_weight_conversion() is True
    assert paddle_vl.patch_weight_conversion() is True  # idempotent
    out = Model().set_hf_state_dict({"a": Tensor("a"), "b": Tensor("b", cpu=True), "n": 3})
    assert out == "loaded"
    assert seen == {"a": True, "b": True, "n": 3}  # GPU tensor moved to CPU, rest untouched
