"""Generate fictional sample invoices for development and tests.

All names, numbers and addresses are invented. Run from the repo root:
    python samples/generate_samples.py

Writes samples/synthetic/*.pdf and samples/synthetic/ground_truth.csv. Some samples
contain deliberate problems so the flags have something to catch:
  inv-03  line items do not add up to the subtotal (sum mismatch)
  inv-04  same vendor and invoice number as inv-01 (duplicate)
  inv-05  vendor not in the vendor master
  scan-01 inv-01 rendered as a skewed, noisy image-only "scan"
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import cv2
import pymupdf
import numpy as np

OUT = Path(__file__).parent / "synthetic"

INVOICES = [
    dict(file="inv-01-layout-a.pdf", layout="a", vendor="Syarikat Contoh Bekalan Sdn Bhd",
         tax_id="W10-1808-32000001", number="INV-2026-0412", date="2026-09-14", due="2026-10-14",
         po="PO-77812", rate=8,
         items=[("A4 paper 80gsm (box)", 10, "42.50"), ("Toner cartridge TN-2480", 2, "189.00"),
                ("Delivery charge", 1, "25.00")]),
    dict(file="inv-02-layout-b.pdf", layout="b", vendor="Kedai Alat Tulis Maju Enterprise",
         tax_id="B16-2001-21000077", number="TI/09/2026/0088", date="2026-09-03", due="2026-10-03",
         po=None, rate=6,
         items=[("Monthly printer maintenance", 1, "1250.00"), ("Replacement drum unit", 1, "385.00")]),
    dict(file="inv-03-layout-c.pdf", layout="c", vendor="Teknik Servis Pintar Sdn Bhd",
         tax_id=None, number="TSP-5531", date="2026-08-28", due="2026-09-27", po="PO-77790", rate=8,
         items=[("Aircond servicing (unit)", 6, "120.00"), ("Gas top-up R32", 2, "95.00")],
         printed_subtotal="1100.00"),
    dict(file="inv-04-layout-a.pdf", layout="a", vendor="Syarikat Contoh Bekalan Sdn Bhd",
         tax_id="W10-1808-32000001", number="INV-2026-0412", date="2026-09-20", due="2026-10-20",
         po="PO-77820", rate=8, items=[("A4 paper 80gsm (box)", 4, "42.50")]),
    dict(file="inv-05-layout-b.pdf", layout="b", vendor="Perniagaan Baharu Cahaya",
         tax_id=None, number="PBC-0007", date="2026-09-25", due="2026-10-25", po=None, rate=0,
         items=[("Cleaning supplies (lot)", 1, "310.40")]),
]

VENDOR_MASTER = [
    ("V0001", "Syarikat Contoh Bekalan Sdn. Bhd.", "W10-1808-32000001"),
    ("V0002", "Kedai Alat Tulis Maju", "B16-2001-21000077"),
    ("V0003", "Teknik Servis Pintar Sdn Bhd", ""),
    ("V0004", "Logistik Kilat Utara Sdn Bhd", ""),
]

BUYER = ["Bill to:", "Contoh Holdings Sdn Bhd (Subsidiary A)", "Lot 12, Jalan Sampel 3/1",
         "40000 Shah Alam, Selangor"]


def money(x: Decimal) -> str:
    return f"{x:,.2f}"


def totals(inv: dict) -> tuple[Decimal, Decimal, Decimal]:
    subtotal = Decimal(inv.get("printed_subtotal") or sum(
        Decimal(p) * q for _, q, p in inv["items"]))
    tax = (subtotal * inv["rate"] / 100).quantize(Decimal("0.01"))
    return subtotal, tax, subtotal + tax


def dmy(iso: str, style: str) -> str:
    y, m, d = iso.split("-")
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{d}/{m}/{y}" if style == "slash" else f"{d}-{months[int(m) - 1]}-{y}"


def draw_invoice(inv: dict, path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    t = page.insert_text
    subtotal, tax, total = totals(inv)
    layout = inv["layout"]
    date_style = "slash" if layout != "b" else "mon"

    if layout == "a":
        t((50, 60), inv["vendor"], fontsize=16, fontname="hebo")
        t((50, 78), "No. 8, Jalan Contoh 2, 47100 Puchong, Selangor", fontsize=9)
        if inv["tax_id"]:
            t((50, 91), f"SST Reg. No: {inv['tax_id']}", fontsize=9)
        t((400, 60), "INVOICE", fontsize=20, fontname="hebo")
        meta = [("Invoice No:", inv["number"]), ("Date:", dmy(inv["date"], date_style)),
                ("Due Date:", dmy(inv["due"], date_style))]
        if inv["po"]:
            meta.append(("PO No:", inv["po"]))
        for i, (k, v) in enumerate(meta):
            t((400, 85 + i * 14), k, fontsize=9, fontname="hebo")
            t((470, 85 + i * 14), v, fontsize=9)
        y0 = 120
    elif layout == "b":
        t((50, 60), "TAX INVOICE", fontsize=18, fontname="hebo")
        t((330, 55), inv["vendor"], fontsize=11, fontname="hebo")
        t((330, 69), "22 Lorong Sampel, 10200 George Town, Penang", fontsize=8)
        if inv["tax_id"]:
            t((330, 81), f"Tax ID: {inv['tax_id']}", fontsize=8)
        t((50, 90), f"Inv #  {inv['number']}", fontsize=10)
        t((50, 104), f"Issued  {dmy(inv['date'], date_style)}", fontsize=10)
        t((50, 118), f"Payment due  {dmy(inv['due'], date_style)}", fontsize=10)
        y0 = 135
    else:
        t((50, 55), inv["vendor"].upper(), fontsize=13, fontname="hebo")
        t((50, 70), f"Invoice {inv['number']}   |   {dmy(inv['date'], date_style)}"
                    f"   |   Terms: 30 days (due {dmy(inv['due'], date_style)})", fontsize=9)
        if inv["po"]:
            t((50, 83), f"Your ref: {inv['po']}", fontsize=9)
        y0 = 100

    for i, line in enumerate(BUYER):
        t((50, y0 + i * 12), line, fontsize=9, fontname="hebo" if i == 0 else "helv")

    # Line item table
    y = y0 + 70
    cols = [50, 330, 390, 470]
    heads = ["Description", "Qty", "Unit Price (RM)", "Amount (RM)"]
    page.draw_rect(pymupdf.Rect(45, y - 12, 550, y + 4), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    for x, h in zip(cols, heads):
        t((x, y), h, fontsize=9, fontname="hebo")
    for desc, qty, price in inv["items"]:
        y += 18
        amount = Decimal(price) * qty
        t((cols[0], y), desc, fontsize=9)
        t((cols[1], y), str(qty), fontsize=9)
        t((cols[2], y), money(Decimal(price)), fontsize=9)
        t((cols[3], y), money(amount), fontsize=9)
        page.draw_line(pymupdf.Point(45, y + 5), pymupdf.Point(550, y + 5), color=(0.7, 0.7, 0.7))

    y += 30
    rows = [("Subtotal", money(subtotal)), (f"SST {inv['rate']}%", money(tax)),
            ("TOTAL (MYR)", money(total))]
    for k, v in rows:
        bold = k.startswith("TOTAL")
        t((370, y), k, fontsize=10, fontname="hebo" if bold else "helv")
        t((470, y), v, fontsize=10, fontname="hebo" if bold else "helv")
        y += 16
    t((50, 790), "This is a computer-generated sample. Fictional data for testing only.", fontsize=7)
    doc.save(path)


def make_scan(src: Path, dst: Path, angle: float = 2.5, seed: int = 7) -> None:
    """Rasterise a PDF page, rotate slightly, add noise, save as an image-only PDF."""
    with pymupdf.open(src) as doc:
        pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=255)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 12, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    out = pymupdf.open()
    page = out.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=jpg.tobytes())
    out.save(dst, deflate=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for inv in INVOICES:
        draw_invoice(inv, OUT / inv["file"])
        subtotal, tax, total = totals(inv)
        rows.append({
            "file": inv["file"], "vendor_name": inv["vendor"], "vendor_tax_id": inv["tax_id"] or "",
            "invoice_number": inv["number"], "invoice_date": inv["date"], "due_date": inv["due"],
            "po_number": inv["po"] or "", "currency": "MYR", "subtotal": f"{subtotal:.2f}",
            "tax_rate": inv["rate"], "tax_amount": f"{tax:.2f}", "total": f"{total:.2f}",
        })
    make_scan(OUT / INVOICES[0]["file"], OUT / "scan-01-skewed.pdf")
    rows.append({**rows[0], "file": "scan-01-skewed.pdf"})

    with open(OUT / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(Path(__file__).parent / "vendor_master.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vendor_code", "vendor_name", "tax_id"])
        writer.writerows(VENDOR_MASTER)
    print(f"Wrote {len(rows)} samples to {OUT}")


if __name__ == "__main__":
    main()
