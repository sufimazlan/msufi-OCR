from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ocrmod.config import REPO_ROOT
from ocrmod.extract import ExtractionResult

SAMPLES = REPO_ROOT / "samples" / "synthetic"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_samples",
                                                  REPO_ROOT / "samples" / "generate_samples.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_samples"] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()


def expected_extraction(inv: dict) -> dict:
    subtotal, tax, total = gen.totals(inv)
    return {
        "vendor_name": inv["vendor"], "vendor_tax_id": inv["tax_id"],
        "invoice_number": inv["number"], "invoice_date": inv["date"], "due_date": inv["due"],
        "po_number": inv["po"], "currency": "MYR", "subtotal": float(subtotal),
        "tax_rate": float(inv["rate"]), "tax_amount": float(tax), "total": float(total),
        "line_items": [{"description": d, "quantity": float(q), "unit_price": float(p),
                        "amount": float(gen.Decimal(p) * q)} for d, q, p in inv["items"]],
    }


class FakeExtractor:
    """Stands in for Qwen: returns the known answer for whichever sample the text is from."""

    name = "fake"

    def __init__(self, token_probs: dict[str, float] | None = None):
        self.answers = [expected_extraction(inv) for inv in gen.INVOICES]
        self.token_probs = token_probs or {}

    def extract(self, markdown: str) -> ExtractionResult:
        def hits(ans):
            return sum(1 for v in (ans["invoice_number"], ans["po_number"], ans["vendor_name"])
                       if v and v in markdown) + (ans["invoice_number"] in markdown) * 10
        best = max(self.answers, key=hits)
        if hits(best) == 0:
            return ExtractionResult(data={"line_items": []}, warnings=["no text"])
        return ExtractionResult(data=best, token_probs=dict(self.token_probs))


@pytest.fixture(scope="session", autouse=True)
def samples() -> Path:
    if not (SAMPLES / "ground_truth.csv").exists():
        gen.main()
    return SAMPLES
