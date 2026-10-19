"""Local web page: upload PDFs, watch progress, download the Excel file.

Runs on 127.0.0.1 only (CLAUDE.md: first version is for one user on the GPU PC). Uploaded
files and results live in the data folder outside the repo.
"""

from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..config import Settings, apply_gpu_limits
from .db import Batch, Document, make_session_factory
from .worker import Worker, batch_dir

HERE = Path(__file__).parent
STAGES = {
    "ingest": "Straightening pages",
    "layout": "Reading layout (PaddleOCR-VL)",
    "lines": "Reading lines (PP-OCR)",
    "extract": "Extracting fields (Qwen)",
}
STAGE_ORDER = ["ingest", "layout", "lines", "extract"]


def _safe_name(index: int, name: str) -> str:
    """Each upload gets its own numbered folder, so two files with the same name can't
    collide and the pipeline (and the Excel file) still sees the original file name."""
    stem = re.sub(r"[^A-Za-z0-9._ ()-]+", "_", Path(name).stem).strip()[:120] or "document"
    return f"{index:03d}/{stem}.pdf"


def batch_summary(batch: Batch) -> dict:
    """Overall progress. The pipeline runs stage by stage across the whole batch, so the
    useful line is "<furthest stage>: <documents that reached it> of <total>"."""
    docs = batch.documents
    finished = sum(1 for d in docs if d.status in ("done", "failed"))
    stage_text = None
    active = [d.stage for d in docs if d.stage]
    if batch.status == "processing" and active:
        stage = max(active, key=STAGE_ORDER.index)
        reached = sum(1 for d in docs if d.status in ("done", "failed")
                      or (d.stage and STAGE_ORDER.index(d.stage) >= STAGE_ORDER.index(stage)))
        stage_text = f"{STAGES[stage]}: {reached} of {len(docs)}"
    return {"total": len(docs), "finished": finished, "stage_text": stage_text,
            "percent": int(100 * finished / len(docs)) if docs else 0}


def create_app(settings: Settings, start_worker: bool = True, extractor=None,
               layout_name: str | None = None, lines_name: str | None = None) -> FastAPI:
    apply_gpu_limits(settings)  # before the worker first imports paddle
    data_dir = settings.path("data_dir")
    sessions = make_session_factory(data_dir / "ocrmod.sqlite")
    worker = Worker(settings, sessions, data_dir, extractor, layout_name, lines_name)
    web = settings["web"]
    max_bytes = int(web["max_upload_mb"]) * 1024 * 1024

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="OCR Document Module", lifespan=lifespan)
    app.state.worker = worker
    app.state.sessions = sessions
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(STAGES=STAGES, poll_seconds=web["poll_seconds"])

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with sessions() as s:
            batches = s.scalars(select(Batch).options(selectinload(Batch.documents))
                                .order_by(Batch.id.desc()).limit(20)).all()
            rows = [(b, batch_summary(b)) for b in batches]
        return templates.TemplateResponse(request, "index.html", {"rows": rows})

    @app.post("/batches")
    async def upload(files: list[UploadFile] = File(...), company: str = Form(""),
                     doc_type: str = Form("invoice")):
        pdfs = [f for f in files if f.filename]
        if not pdfs:
            raise HTTPException(400, "Choose at least one PDF")
        with sessions.begin() as s:
            batch = Batch(company=company.strip() or None, doc_type=doc_type)
            s.add(batch)
            s.flush()
            batch_id = batch.id
        input_dir = batch_dir(data_dir, batch_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        try:
            for i, f in enumerate(pdfs, start=1):
                head = await f.read(5)
                if not f.filename.lower().endswith(".pdf") or head != b"%PDF-":
                    raise HTTPException(400, f"{f.filename} is not a PDF")
                name = _safe_name(i, f.filename)
                size = len(head)
                (input_dir / name).parent.mkdir(parents=True, exist_ok=True)
                with open(input_dir / name, "wb") as out:
                    out.write(head)
                    while chunk := await f.read(1 << 20):
                        size += len(chunk)
                        if size > max_bytes:
                            raise HTTPException(400, f"{f.filename} is over {web['max_upload_mb']}MB")
                        out.write(chunk)
                saved.append((f.filename, name))
        except HTTPException:
            with sessions.begin() as s:
                s.delete(s.get(Batch, batch_id))
            shutil.rmtree(input_dir, ignore_errors=True)
            raise
        with sessions.begin() as s:
            for original, name in saved:
                s.add(Document(batch_id=batch_id, filename=original, stored_name=name))
        worker.notify()
        return RedirectResponse(f"/batches/{batch_id}", status_code=303)

    def _load(batch_id: int) -> Batch:
        with sessions() as s:
            batch = s.scalars(select(Batch).options(selectinload(Batch.documents))
                              .where(Batch.id == batch_id)).first()
        if batch is None:
            raise HTTPException(404, "No such batch")
        return batch

    @app.get("/batches/{batch_id}", response_class=HTMLResponse)
    def batch_page(request: Request, batch_id: int):
        batch = _load(batch_id)
        return templates.TemplateResponse(request, "batch.html",
                                          {"batch": batch, "summary": batch_summary(batch)})

    @app.get("/batches/{batch_id}/status", response_class=HTMLResponse)
    def batch_status(request: Request, batch_id: int):
        batch = _load(batch_id)
        with sessions() as s:
            ahead = len(s.scalars(select(Batch.id).where(Batch.status.in_(["queued", "processing"]),
                                                         Batch.id < batch_id)).all())
        return templates.TemplateResponse(request, "_status.html",
                                          {"batch": batch, "summary": batch_summary(batch),
                                           "ahead": ahead})

    @app.get("/batches/{batch_id}/export.xlsx")
    def download(batch_id: int):
        batch = _load(batch_id)
        path = batch_dir(data_dir, batch_id) / "export.xlsx"
        if batch.status != "done" or not path.exists():
            raise HTTPException(404, "Excel file not ready")
        stamp = batch.created_at.strftime("%Y%m%d-%H%M")
        label = re.sub(r"[^A-Za-z0-9_-]+", "_", batch.company or "batch")
        return FileResponse(path, filename=f"ocr-{label}-{batch_id}-{stamp}.xlsx")

    return app
