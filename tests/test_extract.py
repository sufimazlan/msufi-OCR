import json
import math

from ocrmod.extract import EXTRACTION_SCHEMA, field_token_probs, leaf_spans


def test_schema_is_self_contained():
    text = json.dumps(EXTRACTION_SCHEMA)
    assert "$ref" not in text and "$defs" not in text
    assert "line_items" in EXTRACTION_SCHEMA["properties"]


def test_leaf_spans_paths():
    content = '{"total": 894.24, "vendor_name": "A \\"B\\" Sdn Bhd", "line_items": [{"amount": 5}], "po_number": null}'
    spans = leaf_spans(content)
    assert content[slice(*spans["total"])] == "894.24"
    assert content[slice(*spans["vendor_name"])] == '"A \\"B\\" Sdn Bhd"'
    assert content[slice(*spans["line_items.0.amount"])] == "5"
    assert content[slice(*spans["po_number"])] == "null"


def test_field_token_probs_uses_lowest_digit_token():
    tokens = ['{"', "total", '":', " 89", "4", ".", "24", ', "', "currency", '":', ' "', "MY", "R", '"}']
    lps = [0.0, 0.0, 0.0, -0.01, math.log(0.4), -0.001, -0.02, 0.0, 0.0, 0.0, 0.0, -0.05, -0.3, 0.0]
    content = "".join(tokens)
    probs = field_token_probs(content, [{"token": t, "logprob": lp} for t, lp in zip(tokens, lps)])
    assert probs["total"] == math.exp(math.log(0.4))
    # No digits in "MYR": falls back to all value tokens.
    assert probs["currency"] == math.exp(-0.3)


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_ollama_extractor_request_and_logprobs(monkeypatch):
    from ocrmod import extract as ex

    content = '{"invoice_number": "INV-9", "total": 108.0, "line_items": []}'
    tokens = ['{"invoice_number": "', "INV", "-", "9", '", "total": ', "10", "8", ".0", ', "line_items": []}']
    lps = [0.0, -0.01, 0.0, math.log(0.7), 0.0, -0.02, math.log(0.55), -0.01, 0.0]
    assert "".join(tokens) == content
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json, url=url)
        return _Resp({"message": {"content": content},
                      "logprobs": [{"token": t, "logprob": lp} for t, lp in zip(tokens, lps)]})

    monkeypatch.setattr(ex.httpx, "post", fake_post)
    result = ex.OllamaExtractor("http://x:11434", "qwen3.5-4b").extract("# Invoice INV-9")
    assert sent["url"].endswith("/api/chat")
    assert sent["format"] == ex.EXTRACTION_SCHEMA and sent["logprobs"] is True and sent["think"] is False
    assert sent["options"]["temperature"] == 0
    assert result.data["invoice_number"] == "INV-9" and result.data["vendor_name"] is None
    assert result.token_probs["total"] == math.exp(math.log(0.55))
    assert result.token_probs["invoice_number"] == math.exp(math.log(0.7))
