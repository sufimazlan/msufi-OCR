"""The GPU queue: one background thread that processes batches one at a time, oldest first.

The GPU is a single shared resource (CLAUDE.md), so there is exactly one worker. Each batch
goes through the normal pipeline (run_batch); its progress callback updates the database,
which the progress page polls.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..config import Settings
from ..export_excel import export
from ..pipeline import run_batch
from .db import Batch, Document

log = logging.getLogger(__name__)


def batch_dir(data_dir: Path, batch_id: int) -> Path:
    return data_dir / "batches" / f"{batch_id:05d}"


class Worker:
    def __init__(self, settings: Settings, sessions: sessionmaker, data_dir: Path,
                 extractor=None, layout_name: str | None = None, lines_name: str | None = None):
        self.settings = settings
        self.sessions = sessions
        self.data_dir = data_dir
        self.extractor = extractor
        self.layout_name = layout_name
        self.lines_name = lines_name
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle --

    def recover(self) -> None:
        """A batch left 'processing' means the server stopped mid-run: queue it again."""
        with self.sessions.begin() as s:
            for batch in s.scalars(select(Batch).where(Batch.status == "processing")):
                batch.status, batch.started_at = "queued", None
                for doc in batch.documents:
                    doc.status, doc.stage, doc.error = "queued", None, None

    def start(self) -> None:
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="gpu-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def notify(self) -> None:
        """Called after an upload so the worker doesn't wait for its next poll."""
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.process_next()
            except Exception:
                log.exception("Worker error")
                busy = False
            if not busy:
                self._wake.wait(timeout=5)
                self._wake.clear()

    # -- work --

    def process_next(self) -> bool:
        """Process the oldest queued batch. Returns False if there was nothing to do."""
        with self.sessions.begin() as s:
            batch = s.scalars(select(Batch).where(Batch.status == "queued")
                              .order_by(Batch.id).limit(1)).first()
            if batch is None:
                return False
            batch.status, batch.started_at = "processing", datetime.now()
            batch_id, company, doc_type = batch.id, batch.company, batch.doc_type
            doc_ids = [d.id for d in batch.documents]
            names = [d.stored_name for d in batch.documents]

        folder = batch_dir(self.data_dir, batch_id)
        pdfs = [folder / "input" / n for n in names]

        def progress(index: int, stage: str) -> None:
            with self.sessions.begin() as s:
                doc = s.get(Document, doc_ids[index])
                if stage in ("done", "failed"):
                    doc.status, doc.stage = stage, None
                else:
                    doc.status, doc.stage = "processing", stage

        try:
            results = run_batch(pdfs, folder / "output", self.settings, company=company,
                                doc_type=doc_type, layout_name=self.layout_name,
                                lines_name=self.lines_name, extractor=self.extractor,
                                progress=progress)
            export(results, folder / "export.xlsx")
        except Exception as exc:  # e.g. Ollama not running: the whole batch fails
            log.exception("Batch %s failed", batch_id)
            with self.sessions.begin() as s:
                batch = s.get(Batch, batch_id)
                batch.status, batch.error, batch.finished_at = "failed", str(exc)[:2000], datetime.now()
                for doc in batch.documents:
                    if doc.status != "done":
                        doc.status, doc.stage = "failed", None
            return True

        with self.sessions.begin() as s:
            batch = s.get(Batch, batch_id)
            for doc_id, res in zip(doc_ids, results):
                doc = s.get(Document, doc_id)
                doc.status = res.status
                doc.stage = None
                doc.error = res.error
                doc.flags = sum(1 for f in res.fields if f.level != "green")
                doc.red_flags = sum(1 for f in res.fields if f.level == "red")
            batch.status, batch.finished_at = "done", datetime.now()
        return True
