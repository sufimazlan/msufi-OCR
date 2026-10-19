"""Command line: `ocrmod process ...` and `ocrmod export ...`."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import Settings


def _collect_pdfs(inputs: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            pdfs += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".pdf")
        elif p.suffix.lower() == ".pdf" and p.exists():
            pdfs.append(p)
        else:
            sys.exit(f"Not a PDF or folder: {p}")
    if not pdfs:
        sys.exit("No PDFs found")
    return pdfs


def cmd_process(args) -> None:
    from .export_excel import export
    from .pipeline import run_batch

    settings = Settings.load(Path(args.config) if args.config else None)
    pdfs = _collect_pdfs(args.inputs)
    out = (Path(args.out) if args.out
           else settings.path("data_dir") / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Processing {len(pdfs)} PDF(s) -> {out}")
    try:
        results = run_batch(pdfs, out, settings, company=args.company, doc_type=args.doc_type,
                            layout_name=args.layout_engine, lines_name=args.lines_engine)
    except RuntimeError as exc:
        sys.exit(f"Error: {exc}")
    xlsx = export(results, out / "export.xlsx")
    for r in results:
        flags = [f for f in r.fields if f.level != "green"]
        state = "FAILED: " + (r.error or "") if r.status == "failed" else (
            f"{len(flags)} flag(s)" if flags else "clean")
        print(f"  {r.source_name}: {state}")
    if results and not results[0].thresholds_calibrated:
        print("Note: thresholds are placeholders until scripts/calibrate.py is run.")
    print(f"Excel: {xlsx}")


def cmd_export(args) -> None:
    from .export_excel import export
    from .pipeline import load_results

    out = Path(args.run_dir)
    print(f"Excel: {export(load_results(out), Path(args.xlsx) if args.xlsx else out / 'export.xlsx')}")


def cmd_serve(args) -> None:
    import uvicorn

    from .web.app import create_app

    settings = Settings.load(Path(args.config) if args.config else None)
    web = settings["web"]
    host, port = args.host or web["host"], args.port or web["port"]
    print(f"Open http://localhost:{port} in your browser. Data folder: {settings.path('data_dir')}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ocrmod", description="Local OCR document module")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Run the pipeline on PDFs or folders of PDFs")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--company", help="Subsidiary tag, e.g. ABC")
    p.add_argument("--doc-type", default="invoice", choices=["invoice"],
                   help="Only invoices for now; statements get their own schema later")
    p.add_argument("--out", help="Output folder (default: <data_dir>/runs/<timestamp>)")
    p.add_argument("--layout-engine", choices=["paddle_vl", "glm_ocr", "pdftext"])
    p.add_argument("--lines-engine", choices=["ppocr", "pdftext", "none"])
    p.add_argument("--config", help="Extra TOML file overriding config/default.toml")
    p.set_defaults(func=cmd_process)

    w = sub.add_parser("serve", help="Start the upload/progress web page (local only)")
    w.add_argument("--host")
    w.add_argument("--port", type=int)
    w.add_argument("--config", help="Extra TOML file overriding config/default.toml")
    w.set_defaults(func=cmd_serve)

    e = sub.add_parser("export", help="Rebuild the Excel file from a run folder")
    e.add_argument("run_dir")
    e.add_argument("--xlsx")
    e.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
