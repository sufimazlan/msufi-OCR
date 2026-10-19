"""Jobs database. SQLite for now (one file in the data folder); written with SQLAlchemy so
it can move to PostgreSQL or SQL Server by changing the URL."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import ForeignKey, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    company: Mapped[str | None] = mapped_column(String(100))
    doc_type: Mapped[str] = mapped_column(String(30), default="invoice")
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/processing/done/failed
    error: Mapped[str | None] = mapped_column(String(2000))
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]
    documents: Mapped[list[Document]] = relationship(
        back_populates="batch", order_by="Document.id", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"))
    filename: Mapped[str] = mapped_column(String(255))     # as uploaded
    stored_name: Mapped[str] = mapped_column(String(255))  # file name in the batch input folder
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/processing/done/failed
    stage: Mapped[str | None] = mapped_column(String(20))  # ingest/layout/lines/extract
    flags: Mapped[int] = mapped_column(default=0)
    red_flags: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(2000))
    batch: Mapped[Batch] = relationship(back_populates="documents")


def make_session_factory(db_path: Path) -> sessionmaker:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(conn, _):
        # WAL lets the progress page read while the worker writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
