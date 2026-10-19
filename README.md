# msufi-OCR: local OCR document module for AP

Upload scanned supplier invoices, read them with local OCR models, extract the fields,
flag anything doubtful for review, and export to Excel. Everything runs on our own
machines (PDPA): no document content goes to any cloud service.

The full brief, constraints and decisions are in [CLAUDE.md](CLAUDE.md).

**Status:** M1 (command-line pipeline) done; M2/M3 started: a local upload page with a GPU
queue, live progress and Excel download. Review queue (M4) and the document view come next.

## How it works

```
PDF ──► 1 Ingest ──► 2 Layout engine ──► 4 Qwen extraction ──► 5 Validation ──► 6 Flags ──► 8 Excel
        PyMuPDF       PaddleOCR-VL          JSON schema +          sums, tax,       red /
        + deskew      (Markdown, tables)    token probabilities    dates, dupes,    yellow /
                    3 Line engine                                  vendor master    green
                      PP-OCRv5 (text, confidence, box per line) ─────────────────────►┘
```

- **Two OCR engines on purpose.** The layout engine (PaddleOCR-VL) gives clean Markdown
  for the LLM. The line engine (PP-OCRv5) gives a confidence and a position for every line.
  If a value Qwen extracted can't be found in PP-OCRv5's lines, that is the
  "OCR disagree" flag; if it can, its box gives the crop shown to the reviewer.
- **Each field gets a score** = the lowest of: Qwen's least-confident digit, PP-OCRv5's line
  confidence, and the engine agreement. A failed validation check always wins (for example,
  "sum mismatch" makes the subtotal red). The red/yellow cut-offs live in
  `config/thresholds.json` and are fitted from the test set by `scripts/calibrate.py`.
  **Until that has run, the thresholds are placeholders.**
- **Stage by stage, not document by document.** A batch runs all OCR first, then all
  extraction. With 8GB of VRAM, the Paddle models and Qwen are loaded one after the other.

## Folder structure

```
config/default.toml       settings (override per machine in config/local.toml, gitignored)
config/thresholds.json    flag thresholds (placeholder until calibrated)
src/ocrmod/
  ingest.py               1  render pages, fix rotation and skew
  engines/                2-3 paddle_vl, ppocr, glm_ocr (backup), pdftext (dev only)
  extract.py              4  Qwen via Ollama: JSON schema + logprobs
  validate.py             5  checks, vendor master match, duplicate registry
  confidence.py           6  per-field score, flags, crops
  export_excel.py         8  Excel with colour flags
  pipeline.py, cli.py     batch runner and the `ocrmod` command
  web/                    upload page, GPU worker queue, progress (FastAPI + HTMX)
scripts/gpu_check.py      test each model on the GPU PC (time, peak VRAM)
scripts/evaluate.py       accuracy on the real test set (aggregate output only)
scripts/calibrate.py      fit the flag thresholds from an evaluation report
samples/                  fictional invoices + vendor master (never real documents)
docs/setup-gpu.md         GPU PC setup (WSL2, PaddleOCR)
tests/                    run anywhere, no GPU needed
```

## Running it

**GPU PC, upload page:** in Ubuntu,

```bash
source ~/venvs/ocrmod/bin/activate
cd /mnt/c/Users/sufim/Dev/msufi-OCR
ocrmod serve
```

then open http://localhost:8000 in a Windows browser. Upload PDFs, watch the progress, and
download the Excel file when it says Done. Batches run one at a time (one GPU). Keep the
Ubuntu window open while it works; Ctrl+C stops the server, and an unfinished batch starts
again the next time you run `ocrmod serve`. Uploaded files and results are stored in
`~/ocrmod-data` (outside the repo).

**GPU PC, command line:** follow [docs/setup-gpu.md](docs/setup-gpu.md) once. Then, in Ubuntu:

```bash
source ~/venvs/ocrmod/bin/activate
cd /mnt/c/Users/sufim/Dev/msufi-OCR
ocrmod process path/to/pdfs --company SUB-A          # writes ~/ocrmod-data/runs/<timestamp>/export.xlsx
ocrmod process a.pdf --layout-engine glm_ocr         # try the backup engine
ocrmod export ~/ocrmod-data/runs/20261013-101500     # rebuild the Excel from a run
```

**Any machine without the models (work laptop, Windows PowerShell):** tests and the dev
engine only.

```powershell
cd C:\Users\sufim\Dev\msufi-OCR
python -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
python -m pytest -q
```

Each run folder has one sub-folder per document with the page images, `layout.md` (engine 1),
`lines.json` (engine 2), `extraction.json` (Qwen + token probabilities), `result.json`
(fields, flags, checks) and `crops/`. All of it lives in `~/ocrmod-data`, outside the repo,
so real documents and their values can't be committed by accident.

## Accuracy test and calibration (real documents stay outside the repo)

1. Put the ~50 test PDFs outside the repo, for example `C:\ocr-testset\pdfs\`, and fill in
   a ground-truth CSV in the same format as `samples/synthetic/ground_truth.csv`.
2. In Ubuntu:
   ```bash
   python scripts/evaluate.py --docs /mnt/c/ocr-testset/pdfs --truth /mnt/c/ocr-testset/truth.csv \
       --layout-engine paddle_vl --layout-engine glm_ocr
   python scripts/calibrate.py output/eval-<timestamp>.json --engine paddle_vl
   ```
3. Commit `config/thresholds.json`. It contains only numbers.

`evaluate.py` refuses inputs inside the repo, works in a temporary folder it deletes, and
its report has no values, file names or vendor names.

## Git: when to pull

| Situation | Do this (in `C:\Users\sufim\Dev\msufi-OCR`, Windows PowerShell) |
|---|---|
| Claude says "pushed" | `git pull` |
| Before you start working on any machine | `git pull` |
| You changed files and want Claude (or another machine) to see them | `git add -A`, `git commit -m "what changed"`, `git push` |
| Switching between the work laptop and the GPU PC | push on the one you leave, pull on the one you sit at |

Claude works in a fresh cloud copy of GitHub each time, so anything not pushed is invisible to it.

## Not in M1 yet

- Statements / long documents (Baidu Unlimited-OCR, possibly on the Mac via MLX)
- Job queue, API, dashboard, review queue, audit trail (M2-M4)
- ERP import format (to be confirmed)
