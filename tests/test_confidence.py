from ocrmod.config import Thresholds
from ocrmod.confidence import find_in_lines, flatten, score_fields
from ocrmod.schemas import Check, OCRLine

TH = Thresholds(calibrated=False, yellow_below=0.9, red_below=0.6, ocr_agree_min_ratio=85,
                vendor_match_min_ratio=88)


def line(text, conf=0.99):
    return OCRLine(text=text, conf=conf, bbox=(10, 20, 200, 40), page=0)


def test_find_amount_with_thousands_separator():
    ln, agree = find_in_lines("total", 1733.1, [line("TOTAL (MYR) 1,733.10")], 85)
    assert ln is not None and agree == 1.0
    assert find_in_lines("total", 1733.0, [line("TOTAL (MYR) 1,733.10")], 85)[0] is None


def test_find_dates_in_printed_formats():
    lines = [line("Issued 03-Sep-2026"), line("Date: 14/09/2026")]
    assert find_in_lines("invoice_date", "2026-09-03", lines, 85)[0].text.startswith("Issued")
    assert find_in_lines("invoice_date", "2026-09-14", lines, 85)[0].text.startswith("Date")
    assert find_in_lines("invoice_date", "2026-03-09", lines, 85)[0] is None


def test_find_id_ignores_punctuation():
    assert find_in_lines("invoice_number", "TI/09/2026/0088", [line("Inv # TI/09/2026/0088")], 85)[0]


def test_flatten():
    assert flatten({"a": 1, "line_items": [{"amount": 2}]}) == {"a": 1, "line_items.0.amount": 2}


def test_score_levels_and_reasons():
    data = {"total": 108.0, "subtotal": 100.0, "invoice_number": "X1", "line_items": []}
    lines = [line("Total 108.00", 0.98), line("Subtotal 100.00", 0.7)]
    checks = [Check(name="missing vendor_name", passed=False, fields=["vendor_name"])]
    fields = {f.path: f for f in score_fields(data, checks, lines, {"total": 0.97}, TH)}
    assert fields["total"].level == "green" and fields["total"].bbox
    assert fields["subtotal"].level == "yellow" and "low OCR conf 0.70" in fields["subtotal"].reasons
    assert fields["invoice_number"].level == "red" and "OCR disagree" in fields["invoice_number"].reasons
    assert fields["vendor_name"].level == "red"


def test_validation_failure_overrides_good_score():
    data = {"subtotal": 100.0, "line_items": []}
    checks = [Check(name="sum mismatch", passed=False, fields=["subtotal"])]
    f = score_fields(data, checks, [line("Subtotal 100.00")], {}, TH)[0]
    assert f.level == "red" and f.reasons[0] == "sum mismatch"
