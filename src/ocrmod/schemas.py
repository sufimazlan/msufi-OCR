"""Data shapes passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Level = Literal["green", "yellow", "red"]


# ---- What the LLM must return (also used as the Ollama `format` JSON schema) ----

class LineItem(BaseModel):
    description: str | None = Field(None, description="Item description as printed")
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = Field(None, description="Line total as printed")


class InvoiceExtraction(BaseModel):
    vendor_name: str | None = Field(None, description="Supplier name as printed")
    vendor_tax_id: str | None = Field(None, description="Supplier SST/TIN registration no.")
    invoice_number: str | None = None
    invoice_date: str | None = Field(None, description="YYYY-MM-DD")
    due_date: str | None = Field(None, description="YYYY-MM-DD")
    po_number: str | None = None
    currency: str | None = Field(None, description="ISO code, e.g. MYR")
    subtotal: float | None = Field(None, description="Amount before tax, as printed")
    tax_rate: float | None = Field(None, description="Tax percent as printed, e.g. 8")
    tax_amount: float | None = None
    total: float | None = Field(None, description="Amount payable, as printed")
    line_items: list[LineItem] = Field(default_factory=list)


# ---- OCR engine outputs ----

@dataclass
class OCRLine:
    text: str
    conf: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in page-image pixels
    page: int  # 0-based


@dataclass
class IngestedDoc:
    pdf_path: Path
    doc_id: str  # sha256 of the file
    page_images: list[Path]
    dpi: int
    rotations: list[float] = field(default_factory=list)  # degrees applied per page


# ---- Pipeline results ----

class Check(BaseModel):
    name: str
    passed: bool
    level: Level = "red"
    detail: str = ""
    fields: list[str] = Field(default_factory=list)  # field paths this check implicates


class FieldResult(BaseModel):
    path: str  # e.g. "total" or "line_items.0.amount"
    value: Any = None
    level: Level = "green"
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    token_prob: float | None = None
    ocr_conf: float | None = None
    ocr_agree: bool | None = None
    page: int | None = None
    bbox: list[float] | None = None
    crop: str | None = None  # path relative to the document's output folder


class DocumentResult(BaseModel):
    doc_id: str
    source_name: str
    folder: str = ""  # this document's folder inside the run's output directory
    company: str | None = None
    doc_type: str = "invoice"
    status: Literal["done", "failed"] = "done"
    needs_review: bool = False
    error: str | None = None
    engines: dict[str, str] = Field(default_factory=dict)
    thresholds_calibrated: bool = False
    page_images: list[str] = Field(default_factory=list)
    extraction: dict[str, Any] = Field(default_factory=dict)
    vendor_code: str | None = None
    fields: list[FieldResult] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def field(self, path: str) -> FieldResult | None:
        return next((f for f in self.fields if f.path == path), None)
