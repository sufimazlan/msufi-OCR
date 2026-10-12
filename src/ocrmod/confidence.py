"""Step 6: combine three signals into a per-field flag.

1. Validation: a failed check marks its fields red (or yellow for softer checks).
2. OCR agreement: is the value the LLM read from engine 1's Markdown also visible in
   engine 2's (PP-OCRv5) lines? If yes we also get that line's confidence and box,
   which is where the review crop comes from.
3. Token probability: the lowest digit-token probability for that field.

score = min(token prob, OCR line confidence, agreement). Below `red_below` is red, below
`yellow_below` is yellow. Those two numbers come from config/thresholds.json, which
scripts/calibrate.py fits on the real test set.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import cv2

from .config import Thresholds
from .schemas import Check, FieldResult, OCRLine
from .validate import norm_key

AMOUNT_FIELDS = {"subtotal", "tax_amount", "total", "tax_rate", "quantity", "unit_price", "amount"}
DATE_FIELDS = {"invoice_date", "due_date"}
ID_FIELDS = {"invoice_number", "po_number", "vendor_tax_id"}
SEVERITY = {"green": 0, "yellow": 1, "red": 2}

_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_DATE_FORMATS_DMY = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d %b %Y", "%d %B %Y",
                     "%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%b %d, %Y")


def flatten(data: dict[str, Any]) -> dict[str, Any]:
    """{"line_items": [{"amount": 5}]} -> {"line_items.0.amount": 5}"""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, list):
            for n, item in enumerate(v):
                for ik, iv in item.items():
                    out[f"{k}.{n}.{ik}"] = iv
        else:
            out[k] = v
    return out


def _numbers_in(text: str) -> list[Decimal]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(Decimal(m.replace(",", "")))
        except InvalidOperation:
            pass
    return out


def _dates_in(text: str) -> set[date]:
    found = set()
    candidates = re.findall(r"\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{1,2}[ \-][A-Za-z]{3,9}[ \-,]+\d{2,4}"
                            r"|[A-Za-z]{3,9} \d{1,2}, \d{4}", text)
    for c in candidates:
        for fmt in _DATE_FORMATS_DMY:
            try:
                found.add(datetime.strptime(" ".join(c.split()), fmt).date())
                break
            except ValueError:
                continue
    return found


def find_in_lines(leaf: str, value: Any, lines: list[OCRLine], min_ratio: float):
    """Best supporting OCR line for a value: (line, agreement 0..1) or (None, 0)."""
    from rapidfuzz import fuzz

    if value in (None, "") or not lines:
        return None, 0.0
    if leaf in AMOUNT_FIELDS:
        target = Decimal(str(value))
        hits = [ln for ln in lines if any(n == target for n in _numbers_in(ln.text))]
        return (max(hits, key=lambda ln: ln.conf), 1.0) if hits else (None, 0.0)
    if leaf in DATE_FIELDS:
        try:
            target = date.fromisoformat(str(value))
        except ValueError:
            return None, 0.0
        hits = [ln for ln in lines if target in _dates_in(ln.text)]
        return (max(hits, key=lambda ln: ln.conf), 1.0) if hits else (None, 0.0)
    if leaf in ID_FIELDS:
        key = norm_key(str(value))
        hits = [ln for ln in lines if key and key in norm_key(ln.text)]
        return (max(hits, key=lambda ln: ln.conf), 1.0) if hits else (None, 0.0)
    # Free text (vendor name, descriptions): fuzzy.
    target = str(value).upper()
    best, best_ratio = None, 0.0
    for ln in lines:
        r = fuzz.partial_ratio(target, ln.text.upper())
        if r > best_ratio:
            best, best_ratio = ln, r
    if best_ratio >= min_ratio:
        return best, best_ratio / 100
    return None, best_ratio / 100


def score_fields(data: dict[str, Any], checks: list[Check], lines: list[OCRLine] | None,
                 token_probs: dict[str, float], th: Thresholds) -> list[FieldResult]:
    results = []
    failed = [c for c in checks if not c.passed]
    for path, value in flatten(data).items():
        leaf = path.split(".")[-1]
        fr = FieldResult(path=path, value=value, token_prob=token_probs.get(path))
        signals: list[float] = []
        if fr.token_prob is not None:
            signals.append(fr.token_prob)
            if fr.token_prob < th.yellow_below:
                fr.reasons.append(f"low token prob {fr.token_prob:.2f}")
        if lines is not None and value not in (None, ""):
            line, agree = find_in_lines(leaf, value, lines, th.ocr_agree_min_ratio)
            fr.ocr_agree = line is not None
            if line is not None:
                fr.ocr_conf = line.conf
                fr.page, fr.bbox = line.page, list(line.bbox)
                signals += [line.conf, agree]
                if line.conf < th.yellow_below:
                    fr.reasons.append(f"low OCR conf {line.conf:.2f}")
            else:
                signals.append(agree)
                fr.reasons.append("OCR disagree")
        fr.score = round(min(signals), 4) if signals else None
        if fr.score is not None:
            fr.level = "red" if fr.score < th.red_below else "yellow" if fr.score < th.yellow_below else "green"
        for c in failed:
            if path in c.fields:
                fr.reasons.insert(0, c.name)
                if SEVERITY[c.level] > SEVERITY[fr.level]:
                    fr.level = c.level
        results.append(fr)
    # Missing required fields have no value but still need a red row in the review.
    for c in failed:
        for path in c.fields:
            if not any(r.path == path for r in results):
                results.append(FieldResult(path=path, level=c.level, reasons=[c.name]))
    return results


def save_crops(fields: list[FieldResult], page_images: list[Path], out_dir: Path,
               pad: int = 20) -> None:
    """Save a crop of each flagged field's region, for the reviewer."""
    crops_dir = out_dir / "crops"
    cache: dict[int, Any] = {}
    for f in fields:
        if f.level == "green" or f.bbox is None or f.page is None:
            continue
        img = cache.get(f.page)
        if img is None:
            img = cache[f.page] = cv2.imread(str(page_images[f.page]))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = (int(v) for v in f.bbox)
        crop = img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
        if crop.size == 0:
            continue
        crops_dir.mkdir(parents=True, exist_ok=True)
        name = f"{f.path.replace('.', '_')}.png"
        cv2.imwrite(str(crops_dir / name), crop)
        f.crop = f"crops/{name}"
