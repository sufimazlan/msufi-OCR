"""Check each model on the GPU PC, one at a time, and report time and peak VRAM.

Run inside the WSL2 venv (docs/setup-gpu.md):
    python scripts/gpu_check.py
    python scripts/gpu_check.py --skip paddle_vl     # skip a slow step
    python scripts/gpu_check.py --only paddle_vl     # run just one step

Each step prints a ">>> PASS/FAIL" line as it finishes. Ctrl+C stops the current step
and still prints the summary.

Uses only the synthetic sample invoice, never real documents.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocrmod.config import REPO_ROOT, Settings, apply_gpu_limits  # noqa: E402

SAMPLE = REPO_ROOT / "samples" / "synthetic" / "inv-01-layout-a.pdf"
EXPECT = "INV-2026-0412"


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class PeakVRAM:
    """Polls nvidia-smi in the background while a step runs."""

    def __enter__(self):
        self.base = vram_used_mb()
        self.peak = self.base or 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.5):
            v = vram_used_mb()
            if v:
                self.peak = max(self.peak, v)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def report(self) -> str:
        if self.base is None:
            return "VRAM n/a"
        return f"peak {self.peak} MB used on GPU (+{self.peak - self.base} MB during this step)"


RESULTS: list[tuple[str, str, str]] = []


def step(name: str, skip: set[str]):
    def deco(fn):
        def run(*a):
            if name in skip:
                RESULTS.append((name, "SKIP", ""))
                return None
            print(f"\n=== {name} ===")
            t0 = time.time()
            try:
                with PeakVRAM() as vram:
                    detail = fn(*a) or ""
                RESULTS.append((name, "PASS", f"{detail}  [{time.time() - t0:.1f}s, {vram.report()}]"))
            except KeyboardInterrupt:
                RESULTS.append((name, "STOP", f"stopped with Ctrl+C after {time.time() - t0:.0f}s"))
                print(f">>> STOP {name}")
                raise
            except Exception as exc:
                traceback.print_exc()
                RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"[:300]))
            print(f">>> {RESULTS[-1][1]} {name}: {RESULTS[-1][2]}")
        return run
    return deco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[],
                    help="nvidia paddle ollama orientation ppocr paddle_vl glm_ocr")
    ap.add_argument("--only", nargs="*", help="run only these steps")
    args = ap.parse_args()
    steps = ["nvidia", "paddle", "ollama", "orientation", "ppocr", "paddle_vl", "glm_ocr"]
    skip = set(args.skip) | ({s for s in steps if s not in args.only} if args.only else set())
    settings = Settings.load()
    apply_gpu_limits(settings)
    import os

    cap = os.environ.get("FLAGS_gpu_memory_limit_mb")
    print(f"Paddle GPU memory cap: {cap + ' MB' if cap else 'none'}")
    tmp = Path(tempfile.mkdtemp(prefix="ocrmod-gpucheck-"))

    @step("nvidia", skip)
    def nvidia():
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return out.stdout.strip()

    @step("paddle", skip)
    def paddle_gpu():
        import paddle

        assert paddle.device.is_compiled_with_cuda(), "paddlepaddle is the CPU build"
        paddle.device.set_device("gpu:0")
        x = paddle.randn([1024, 1024])
        float((x @ x).sum())
        return f"paddle {paddle.__version__}, GPUs: {paddle.device.cuda.device_count()}"

    @step("ollama", skip)
    def ollama():
        from ocrmod.engines.pdftext import PdfTextEngine
        from ocrmod.ingest import ingest
        from ocrmod.pipeline import default_extractor

        ext = default_extractor(settings)
        ext.check()
        doc = ingest(SAMPLE, tmp / "ollama", use_paddle_orientation=False)
        res = ext.extract("\n".join(PdfTextEngine().parse(doc)))
        ext.unload()
        got = res.data.get("invoice_number")
        assert got == EXPECT, f"expected {EXPECT}, got {got!r}"
        lp = "logprobs OK" if res.token_probs else "NO LOGPROBS (token signal will be missing)"
        return f"model {ext.model}: invoice_number={got}, total={res.data.get('total')}, {lp}"

    pages: list[Path] = []

    @step("orientation", skip)
    def orientation():
        import cv2

        from ocrmod.ingest import _PaddleOrientation, render_pages, rotate, straighten

        clf = _PaddleOrientation()
        page = render_pages(SAMPLE, 150)[0]
        report = []
        for angle in (0, 90, 180, 270):
            fixed, applied = straighten(rotate(page, angle), 5.0, clf)
            ok = fixed.shape[:2] == page.shape[:2] and clf.angle(fixed) == 0
            report.append(f"{angle}->{'ok' if ok else 'WRONG'}")
            if angle == 180:
                cv2.imwrite(str(tmp / "orientation-180-fixed.png"), fixed)
        assert all(r.endswith("ok") for r in report), (
            "classifier direction differs from ingest.rotate(); see " + str(tmp))
        return ", ".join(report)

    @step("ppocr", skip)
    def ppocr():
        from ocrmod.engines import make_line_engine
        from ocrmod.ingest import ingest

        doc = ingest(SAMPLE, tmp / "ppocr", use_paddle_orientation=False)
        pages[:] = doc.page_images
        eng = make_line_engine("ppocr", settings)
        lines = eng.lines(doc)
        eng.close()
        assert any(EXPECT in ln.text for ln in lines), "invoice number not found in PP-OCR lines"
        mean = sum(ln.conf for ln in lines) / len(lines)
        return f"{len(lines)} lines, mean confidence {mean:.3f}"

    @step("paddle_vl", skip)
    def paddle_vl():
        from ocrmod.engines import make_layout_engine
        from ocrmod.ingest import ingest

        doc = ingest(SAMPLE, tmp / "vl", use_paddle_orientation=False)
        eng = make_layout_engine("paddle_vl", settings)
        try:
            md = eng.parse(doc)[0]
        finally:
            print(f"Paddle memory so far: {eng.memory}")
        eng.close()
        (tmp / "paddle_vl.md").write_text(md, encoding="utf-8")
        mem = ", ".join(f"{k}={v:.0f}MB" for k, v in eng.memory.items() if v is not None)
        print(f"Paddle memory: {mem or 'n/a'}")
        assert EXPECT in md, f"invoice number not in Markdown; see {tmp / 'paddle_vl.md'}"
        return f"{len(md)} chars of Markdown, table={'<table' in md}, {mem}"

    @step("glm_ocr", skip)
    def glm_ocr():
        from ocrmod.engines import make_layout_engine
        from ocrmod.ingest import ingest

        doc = ingest(SAMPLE, tmp / "glm", use_paddle_orientation=False)
        eng = make_layout_engine("glm_ocr", settings)
        md = eng.parse(doc)[0]
        eng.close()
        (tmp / "glm_ocr.md").write_text(md, encoding="utf-8")
        assert EXPECT in md, f"invoice number not in output; see {tmp / 'glm_ocr.md'}"
        return f"{len(md)} chars"

    try:
        for fn in (nvidia, paddle_gpu, ollama, orientation, ppocr, paddle_vl, glm_ocr):
            fn()
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 70)
    for name, status, detail in RESULTS:
        print(f"{status:5} {name:12} {detail}")
    print(f"\nScratch files: {tmp}  (delete when done)")
    if all(s in ("PASS", "SKIP") for _, s, _ in RESULTS):
        shutil.rmtree(tmp, ignore_errors=True)
        print("All good; scratch files removed.")


if __name__ == "__main__":
    main()
