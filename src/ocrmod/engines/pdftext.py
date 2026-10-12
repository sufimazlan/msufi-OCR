"""Dev-only engine: reads the text layer of born-digital PDFs with PyMuPDF.

Lets the whole pipeline run without a GPU (tests, the work laptop, the cloud). It is
useless on real scans, which have no text layer, and it assumes the page was not rotated
during ingest.
"""

from __future__ import annotations

import pymupdf

from ..schemas import IngestedDoc, OCRLine


class PdfTextEngine:
    name = "pdftext"

    def parse(self, doc: IngestedDoc) -> list[str]:
        with pymupdf.open(doc.pdf_path) as pdf:
            return [page.get_text("text", sort=True) for page in pdf]

    def lines(self, doc: IngestedDoc) -> list[OCRLine]:
        scale = doc.dpi / 72
        out: list[OCRLine] = []
        with pymupdf.open(doc.pdf_path) as pdf:
            for page_no, page in enumerate(pdf):
                for block in page.get_text("dict", sort=True)["blocks"]:
                    for line in block.get("lines", []):
                        text = "".join(span["text"] for span in line["spans"]).strip()
                        if text:
                            x0, y0, x1, y1 = (v * scale for v in line["bbox"])
                            out.append(OCRLine(text=text, conf=1.0, bbox=(x0, y0, x1, y1),
                                               page=page_no))
        return out

    def close(self) -> None:
        pass
