"""Step 5: rule checks in plain Python. Each failed check names the fields it implicates,
so the confidence step can flag exactly those."""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from .schemas import Check

REQUIRED = ("vendor_name", "invoice_number", "invoice_date", "total")


def dec(x: Any) -> Decimal | None:
    return None if x is None else Decimal(str(x)).quantize(Decimal("0.01"))


def norm_key(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


_COMPANY_SUFFIX = re.compile(r"\b(SDN\.?\s*BHD\.?|BHD\.?|BERHAD|PLT|ENTERPRISE|S/B)\b", re.I)


def norm_vendor(s: str | None) -> str:
    s = _COMPANY_SUFFIX.sub(" ", (s or "").upper())
    return " ".join(re.sub(r"[^A-Z0-9 ]", " ", s).split())


# ---- Vendor master ----

@dataclass
class VendorMatch:
    code: str | None
    name: str | None
    ratio: float


class VendorMaster:
    """CSV with columns vendor_code, vendor_name, tax_id (tax_id optional)."""

    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows
        self._names = [norm_vendor(r["vendor_name"]) for r in rows]

    @classmethod
    def load(cls, path: Path) -> VendorMaster:
        if not path.exists():
            return cls([])
        with open(path, newline="", encoding="utf-8-sig") as f:
            return cls(list(csv.DictReader(f)))

    def match(self, name: str | None, tax_id: str | None = None) -> VendorMatch:
        if tax_id:
            for r in self.rows:
                if r.get("tax_id") and norm_key(r["tax_id"]) == norm_key(tax_id):
                    return VendorMatch(r["vendor_code"], r["vendor_name"], 100.0)
        if not name or not self.rows:
            return VendorMatch(None, None, 0.0)
        hit = process.extractOne(norm_vendor(name), self._names, scorer=fuzz.token_sort_ratio)
        row = self.rows[hit[2]]
        return VendorMatch(row["vendor_code"], row["vendor_name"], float(hit[1]))


# ---- Duplicate registry (temporary until the database is chosen in M2) ----

class Registry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS invoices (vendor_key TEXT, invoice_key TEXT, doc_id TEXT,"
            " source_name TEXT, seen_at TEXT, UNIQUE(vendor_key, invoice_key, doc_id))"
        )

    def duplicates(self, vendor_key: str, invoice_key: str, doc_id: str) -> list[str]:
        """Other files (not a re-run of this same file) with the same vendor + invoice no."""
        rows = self.conn.execute(
            "SELECT source_name FROM invoices WHERE vendor_key=? AND invoice_key=? AND doc_id<>?",
            (vendor_key, invoice_key, doc_id),
        ).fetchall()
        return [r[0] for r in rows]

    def register(self, vendor_key: str, invoice_key: str, doc_id: str, source_name: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO invoices VALUES (?,?,?,?,?)",
            (vendor_key, invoice_key, doc_id, source_name, datetime.now().isoformat(timespec="seconds")),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---- Checks ----

def _parse_iso(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def run_checks(data: dict[str, Any], cfg: dict[str, Any], vendor: VendorMatch,
               vendor_min_ratio: float, duplicates: list[str], today: date | None = None) -> list[Check]:
    today = today or date.today()
    tol = Decimal(str(cfg["amount_tolerance"]))
    checks: list[Check] = []

    for f in REQUIRED:
        if data.get(f) in (None, ""):
            checks.append(Check(name=f"missing {f}", passed=False, fields=[f],
                                detail=f"{f} not found"))

    items = data.get("line_items") or []
    for n, it in enumerate(items):
        q, u, a = dec(it.get("quantity")), dec(it.get("unit_price")), dec(it.get("amount"))
        if None not in (q, u, a) and abs(q * u - a) > tol:
            checks.append(Check(name="line maths", passed=False, level="yellow",
                                fields=[f"line_items.{n}.amount"],
                                detail=f"line {n + 1}: {q} x {u} = {q * u}, printed {a}"))

    subtotal, tax, total = dec(data.get("subtotal")), dec(data.get("tax_amount")), dec(data.get("total"))
    amounts = [dec(it.get("amount")) for it in items]
    if items and None not in amounts:
        line_sum = sum(amounts, Decimal("0"))
        target, target_field = (subtotal, "subtotal") if subtotal is not None else (
            (total - tax, "total") if total is not None and tax is not None else (total, "total"))
        if target is not None:
            ok = abs(line_sum - target) <= tol * max(1, len(items))
            checks.append(Check(name="sum mismatch", passed=ok, fields=[target_field],
                                detail=f"lines sum to {line_sum}, {target_field} gives {target}"))

    if None not in (subtotal, tax, total):
        ok = abs(subtotal + tax - total) <= tol
        checks.append(Check(name="total mismatch", passed=ok,
                            fields=["subtotal", "tax_amount", "total"],
                            detail=f"{subtotal} + {tax} = {subtotal + tax}, total printed {total}"))

    if subtotal and tax is not None:
        rate = data.get("tax_rate")
        if rate is not None:
            expected = (subtotal * Decimal(str(rate)) / 100).quantize(Decimal("0.01"))
            ok = abs(expected - tax) <= tol
            checks.append(Check(name="tax maths", passed=ok, fields=["tax_amount", "tax_rate"],
                                detail=f"{rate}% of {subtotal} = {expected}, printed {tax}"))
        else:
            implied = tax / subtotal * 100
            ok = any(abs(implied - r) < Decimal("0.1") for r in cfg["allowed_tax_rates"])
            checks.append(Check(name="unusual tax rate", passed=ok, level="yellow",
                                fields=["tax_amount"], detail=f"implied rate {implied:.2f}%"))

    for f in ("invoice_date", "due_date"):
        raw = data.get(f)
        if raw is None:
            continue
        d = _parse_iso(raw)
        if d is None:
            checks.append(Check(name="invalid date", passed=False, fields=[f], detail=f"{f}={raw!r}"))
            continue
        if f == "invoice_date":
            if d > today + timedelta(days=cfg["max_future_days"]):
                checks.append(Check(name="date in future", passed=False, fields=[f], detail=str(d)))
            elif d < today - timedelta(days=cfg["max_invoice_age_days"]):
                checks.append(Check(name="date too old", passed=False, level="yellow", fields=[f],
                                    detail=str(d)))
    inv, due = _parse_iso(data.get("invoice_date")), _parse_iso(data.get("due_date"))
    if inv and due and due < inv:
        checks.append(Check(name="due before invoice date", passed=False, level="yellow",
                            fields=["due_date"], detail=f"{due} < {inv}"))

    if data.get("vendor_name"):
        ok = vendor.ratio >= vendor_min_ratio
        detail = (f"matched {vendor.code} {vendor.name} ({vendor.ratio:.0f})" if ok
                  else f"best match {vendor.name!r} only {vendor.ratio:.0f}")
        checks.append(Check(name="vendor not in master", passed=ok, level="yellow",
                            fields=["vendor_name"], detail=detail))

    if duplicates:
        checks.append(Check(name="duplicate invoice", passed=False, fields=["invoice_number"],
                            detail="also in: " + ", ".join(duplicates[:3])))
    return checks
