"""Runs the steps over a batch of PDFs.

The batch goes stage by stage (all OCR first, then all extraction) rather than document
by document. With 8GB of VRAM the Paddle models and Qwen don't fit together comfortably,
so this way each model is loaded once per batch instead of once per document.

Each document gets a folder in the output directory with its page images, the
intermediate outputs (layout.md, lines.json, extraction.json) and result.json.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .config import Settings, apply_gpu_limits
from .confidence import save_crops, score_fields
from .engines import make_layout_engine, make_line_engine
from .extract import Extractor, OllamaExtractor
from .ingest import ingest
from .schemas import DocumentResult, IngestedDoc, OCRLine
from .validate import Registry, VendorMaster, norm_key, norm_vendor, run_checks

log = logging.getLogger(__name__)


def _folder_name(pdf: Path, doc_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf.stem)[:60]
    return f"{stem}-{doc_id[:8]}"


def default_extractor(settings: Settings) -> OllamaExtractor:
    o = settings["ollama"]
    return OllamaExtractor(o["url"], o["extract_model"], o["num_ctx"], o["keep_alive"],
                           o["timeout_seconds"])


class _Work:
    """Per-document state while the batch moves through the stages."""

    def __init__(self, pdf: Path, out_dir: Path):
        self.pdf = pdf
        self.out_dir = out_dir
        self.doc: IngestedDoc | None = None
        self.markdown = ""
        self.lines: list[OCRLine] | None = None
        self.result: DocumentResult | None = None
        self.error: str | None = None


def run_batch(pdfs: list[Path], out_root: Path, settings: Settings, company: str | None = None,
              doc_type: str = "invoice", layout_name: str | None = None,
              lines_name: str | None = None, extractor: Extractor | None = None,
              registry_path: Path | None = None,
              progress: Callable[[int, str], None] | None = None) -> list[DocumentResult]:
    """`progress(index, stage)` is called as each document enters a stage: ingest, layout,
    lines, extract, then done or failed. Used by the web dashboard."""
    apply_gpu_limits(settings)
    report = progress or (lambda i, stage: None)
    layout_name = layout_name or settings["engines"]["layout"]
    lines_name = lines_name or settings["engines"]["lines"]
    ing = settings["ingest"]
    extractor = extractor or default_extractor(settings)
    check = getattr(extractor, "check", None)
    if check:
        check()
    work = [_Work(p, out_root) for p in pdfs]

    # Stage 1: ingest (CPU, plus the small orientation model when available)
    for i, w in enumerate(work):
        report(i, "ingest")
        try:
            tmp = out_root / "_ingest"
            doc = ingest(w.pdf, tmp, ing["dpi"], ing["max_skew_degrees"], ing["use_paddle_orientation"])
            w.out_dir = out_root / _folder_name(w.pdf, doc.doc_id)
            if w.out_dir.exists():
                _clear(w.out_dir)
            w.out_dir.mkdir(parents=True, exist_ok=True)
            (tmp / "pages").rename(w.out_dir / "pages")
            doc.page_images = [w.out_dir / "pages" / p.name for p in doc.page_images]
            w.doc = doc
        except Exception as exc:
            log.exception("Ingest failed for %s", w.pdf.name)
            w.error = f"ingest: {exc}"
    _clear(out_root / "_ingest")

    # Stage 2: layout engine -> Markdown
    _run_stage(work, make_layout_engine(layout_name, settings), "layout", _do_layout, report)

    # Stage 3: line engine -> lines with confidence and boxes
    line_engine = make_line_engine(lines_name, settings)
    if line_engine is not None:
        _run_stage(work, line_engine, "lines", _do_lines, report)

    # Stage 4-6: extraction, validation, confidence
    vendors = VendorMaster.load(settings.path("vendor_master"))
    registry = Registry(registry_path or settings.path("registry"))
    th = settings.thresholds()
    try:
        for i, w in enumerate(work):
            if w.error is None:
                report(i, "extract")
            w.result = _finish(w, extractor, vendors, registry, th, settings, company, doc_type,
                               {"layout": layout_name, "lines": lines_name,
                                "extractor": getattr(extractor, "name", "custom")})
            report(i, "failed" if w.result.status == "failed" else "done")
    finally:
        registry.close()
        unload = getattr(extractor, "unload", None)
        if unload:
            unload()
    return [w.result for w in work]


def _clear(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _run_stage(work, engine, label, fn, report):
    try:
        for i, w in enumerate(work):
            if w.error is None:
                report(i, label)
                try:
                    fn(w, engine)
                except Exception as exc:
                    log.exception("%s failed for %s", label, w.pdf.name)
                    w.error = f"{label}: {exc}"
    finally:
        engine.close()


def _do_layout(w: _Work, engine) -> None:
    pages = engine.parse(w.doc)
    w.markdown = "\n\n".join(f"<!-- page {i + 1} -->\n{md}" for i, md in enumerate(pages))
    (w.out_dir / "layout.md").write_text(w.markdown, encoding="utf-8")


def _do_lines(w: _Work, engine) -> None:
    w.lines = engine.lines(w.doc)
    (w.out_dir / "lines.json").write_text(json.dumps([asdict(ln) for ln in w.lines], indent=1),
                                          encoding="utf-8")


def _finish(w: _Work, extractor, vendors, registry, th, settings, company, doc_type,
            engines) -> DocumentResult:
    res = DocumentResult(
        doc_id=w.doc.doc_id if w.doc else "",
        source_name=w.pdf.name,
        folder=w.out_dir.name if w.doc else "",
        company=company,
        doc_type=doc_type,
        engines=engines,
        thresholds_calibrated=th.calibrated,
        page_images=[f"pages/{p.name}" for p in w.doc.page_images] if w.doc else [],
    )
    if w.error is None:
        try:
            ex = extractor.extract(w.markdown)
            res.extraction = ex.data
            res.warnings += ex.warnings
            (w.out_dir / "extraction.json").write_text(
                json.dumps({"data": ex.data, "token_probs": ex.token_probs, "raw": ex.raw,
                            "warnings": ex.warnings}, indent=1), encoding="utf-8")
            data = ex.data
            vendor = vendors.match(data.get("vendor_name"), data.get("vendor_tax_id"))
            matched = vendor.ratio >= th.vendor_match_min_ratio
            res.vendor_code = vendor.code if matched else None
            vendor_key = res.vendor_code or norm_vendor(data.get("vendor_name"))
            invoice_key = norm_key(data.get("invoice_number"))
            dups = registry.duplicates(vendor_key, invoice_key, res.doc_id) if invoice_key else []
            res.checks = run_checks(data, settings["validation"], vendor, th.vendor_match_min_ratio, dups)
            res.fields = score_fields(data, res.checks, w.lines, ex.token_probs, th)
            save_crops(res.fields, w.doc.page_images, w.out_dir)
            res.needs_review = any(f.level != "green" for f in res.fields)
            if invoice_key:
                registry.register(vendor_key, invoice_key, res.doc_id, res.source_name)
        except Exception as exc:
            log.exception("Extraction failed for %s", w.pdf.name)
            w.error = f"extract: {exc}"
    if w.error is not None:
        res.status, res.error, res.needs_review = "failed", w.error, True
    if w.doc:
        (w.out_dir / "result.json").write_text(res.model_dump_json(indent=1), encoding="utf-8")
    return res


def load_results(out_root: Path) -> list[DocumentResult]:
    return [DocumentResult.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(out_root.glob("*/result.json"))]
