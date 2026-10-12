"""Fit the yellow/red thresholds from an evaluation report (scripts/evaluate.py).

- yellow_below: the lowest score at which fields at or above it are still correct at
  least --green-precision of the time (default 99%). Anything scoring lower is flagged.
- red_below: walking up from the lowest scores in 0.05-wide bands, every band that is
  correct at most --red-accuracy of the time (default 50%) is red; the first band that
  is mostly right stops the walk.

    python scripts/calibrate.py output/eval-20261013-101500.json --engine paddle_vl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrmod.config import Settings  # noqa: E402


def fit(records: list[dict], green_precision: float, red_accuracy: float,
        band: float = 0.05) -> tuple[float, float, dict]:
    scored = sorted((r["score"], r["correct"]) for r in records if r["score"] is not None)
    if not scored:
        raise SystemExit("No scored fields in the report")
    candidates = sorted({s for s, _ in scored})

    yellow = None
    for t in candidates:  # lowest threshold that meets the precision target
        above = [c for s, c in scored if s >= t]
        if above and sum(above) / len(above) >= green_precision:
            yellow = t
            break
    if yellow is None:
        yellow = 1.01  # nothing is reliable enough: flag everything

    # Walk up from the lowest scores in bands; each band is red while it is mostly wrong.
    bands: dict[int, list[bool]] = {}
    for s, c in scored:
        bands.setdefault(int(s / band + 1e-9), []).append(c)
    red = candidates[0]
    for b in sorted(bands):
        if sum(bands[b]) / len(bands[b]) > red_accuracy:
            break
        red = (b + 1) * band
    above = [c for s, c in scored if s >= yellow]
    stats = {"fields": len(scored), "wrong": sum(1 for _, c in scored if not c),
             "flagged_share": round(sum(1 for s, _ in scored if s < yellow) / len(scored), 4),
             "green_precision": round(sum(above) / len(above), 4) if above else None}
    return round(min(yellow, 1.01), 4), round(min(red, yellow), 4), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--engine", help="use records from this layout engine only")
    ap.add_argument("--green-precision", type=float, default=0.99)
    ap.add_argument("--red-accuracy", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = json.loads(args.report.read_text(encoding="utf-8"))["records"]
    if args.engine:
        records = [r for r in records if r["engine"] == args.engine]
    yellow, red, stats = fit(records, args.green_precision, args.red_accuracy)
    print(f"yellow_below={yellow}  red_below={red}  {stats}")
    if stats["fields"] < 300:
        print("Warning: fewer than 300 scored fields; treat these thresholds as rough.")

    path = Settings.load().path("thresholds")
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(calibrated=True, yellow_below=yellow, red_below=red,
                note=f"Fitted by scripts/calibrate.py from {args.report.name}: {stats}")
    if args.dry_run:
        print(json.dumps(data, indent=2))
    else:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {path}. Commit it so every machine uses the same thresholds.")


if __name__ == "__main__":
    main()
