"""Field-level accuracy on a labelled test set. Prints and saves AGGREGATE metrics only.

Real documents and their ground truth must stay outside the repo. This script refuses
inputs inside the repo (except samples/), works in a temporary folder that it deletes
afterwards, and writes no document values, file names or vendor names to its report.

    python scripts/evaluate.py --docs D:/ocr-testset/pdfs --truth D:/ocr-testset/truth.csv \
        --layout-engine paddle_vl --layout-engine glm_ocr

Ground truth CSV: one row per file, columns `file` plus any of the header fields
(vendor_name, invoice_number, invoice_date [YYYY-MM-DD], total, ...). Blank = not printed.
See samples/synthetic/ground_truth.csv for the format.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrmod.config import REPO_ROOT, Settings  # noqa: E402
from ocrmod.confidence import AMOUNT_FIELDS, ID_FIELDS  # noqa: E402
from ocrmod.pipeline import run_batch  # noqa: E402
from ocrmod.validate import norm_key, norm_vendor  # noqa: E402

FIELDS = ["vendor_name", "vendor_tax_id", "invoice_number", "invoice_date", "due_date",
          "po_number", "currency", "subtotal", "tax_rate", "tax_amount", "total"]


def guard_outside_repo(path: Path, what: str) -> None:
    p = path.resolve()
    if p.is_relative_to(REPO_ROOT) and not p.is_relative_to(REPO_ROOT / "samples"):
        sys.exit(f"Refusing: {what} {p} is inside the repo. Keep real documents outside it.")


def normalise(field: str, value) -> str | None:
    if value in (None, ""):
        return None
    if field in AMOUNT_FIELDS:
        try:
            return str(Decimal(str(value).replace(",", "")).quantize(Decimal("0.01")))
        except InvalidOperation:
            return str(value)
    if field in ID_FIELDS:
        return norm_key(str(value))
    if field == "vendor_name":
        return norm_vendor(str(value))
    return str(value).strip().upper()


def evaluate(engine: str, docs: list[Path], truth: dict[str, dict], work: Path, settings: Settings,
             lines_engine: str | None, extractor=None) -> tuple[dict, list[dict]]:
    results = run_batch(docs, work / engine, settings, layout_name=engine, lines_name=lines_engine,
                        extractor=extractor, registry_path=work / f"registry-{engine}.sqlite")
    per_field = defaultdict(lambda: {"n": 0, "correct": 0, "flagged_wrong": 0, "wrong": 0,
                                     "green_correct": 0, "green": 0})
    records = []
    failed = 0
    for res in results:
        if res.status == "failed":
            failed += 1
        gt = truth.get(res.source_name, {})
        for field in FIELDS:
            if field not in gt:
                continue
            expected = normalise(field, gt[field])
            got = normalise(field, res.extraction.get(field)) if res.status == "done" else None
            ok = expected == got
            fr = res.field(field)
            level = fr.level if fr else ("red" if res.status == "failed" else "green")
            s = per_field[field]
            s["n"] += 1
            s["correct"] += ok
            s["wrong"] += not ok
            s["flagged_wrong"] += (not ok) and level != "green"
            s["green"] += level == "green"
            s["green_correct"] += level == "green" and ok
            # Calibration record: no values, no names.
            records.append({"engine": engine, "field": field, "correct": ok, "level": level,
                            "score": fr.score if fr else None})
    total_n = sum(s["n"] for s in per_field.values())
    summary = {
        "engine": engine, "documents": len(results), "failed_documents": failed,
        "field_accuracy": round(sum(s["correct"] for s in per_field.values()) / max(1, total_n), 4),
        "wrong_fields_flagged": round(sum(s["flagged_wrong"] for s in per_field.values())
                                      / max(1, sum(s["wrong"] for s in per_field.values())), 4),
        "green_precision": round(sum(s["green_correct"] for s in per_field.values())
                                 / max(1, sum(s["green"] for s in per_field.values())), 4),
        "per_field": {f: {"n": s["n"], "accuracy": round(s["correct"] / max(1, s["n"]), 4),
                          "wrong_flagged": f"{s['flagged_wrong']}/{s['wrong']}"}
                      for f, s in per_field.items()},
    }
    return summary, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--layout-engine", action="append", dest="engines")
    ap.add_argument("--lines-engine")
    ap.add_argument("--report", type=Path, help="default: output/eval-<timestamp>.json")
    ap.add_argument("--keep-work", type=Path,
                    help="keep per-document outputs in this folder (must be outside the repo)")
    args = ap.parse_args()

    guard_outside_repo(args.docs, "--docs")
    guard_outside_repo(args.truth, "--truth")
    if args.keep_work:
        guard_outside_repo(args.keep_work, "--keep-work")
    settings = Settings.load()
    engines = args.engines or [settings["engines"]["layout"]]

    with open(args.truth, newline="", encoding="utf-8-sig") as f:
        truth = {row.pop("file"): row for row in csv.DictReader(f)}
    docs = sorted(p for p in args.docs.rglob("*") if p.suffix.lower() == ".pdf" and p.name in truth)
    if not docs:
        sys.exit("No PDFs matching the ground-truth file names")

    work = args.keep_work or Path(tempfile.mkdtemp(prefix="ocrmod-eval-"))
    summaries, records = [], []
    try:
        for engine in engines:
            s, r = evaluate(engine, docs, truth, work, settings, args.lines_engine)
            summaries.append(s)
            records += r
    finally:
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    print(f"\n{len(docs)} documents\n")
    for s in summaries:
        print(f"{s['engine']}: field accuracy {s['field_accuracy']:.1%}, wrong fields flagged "
              f"{s['wrong_fields_flagged']:.1%}, green precision {s['green_precision']:.1%}, "
              f"failed docs {s['failed_documents']}")
        for field, fs in s["per_field"].items():
            print(f"    {field:15} {fs['accuracy']:6.1%}  (n={fs['n']}, wrong flagged {fs['wrong_flagged']})")
    report = args.report or REPO_ROOT / "output" / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summaries": summaries, "records": records}, indent=1), encoding="utf-8")
    print(f"\nReport (aggregate + anonymous calibration records): {report}")
    print(f"Next: python scripts/calibrate.py {report}")


if __name__ == "__main__":
    main()
