# OCR Document Module (working name)

## Goal
A web dashboard that works like a small ERP module. Core feature: upload scanned PDFs
(supplier invoices, statements, many different layouts), run local OCR with layout
understanding, extract structured fields, flag low-confidence values for human review,
and export the results.

## Owner
Sufi, AP Team Lead handling around 10 subsidiaries. The end users are the AP team.
Sufi works comfortably in Python, Excel VBA and Power Query. Explain non-obvious
architecture choices briefly.

## Hard constraints
- All document processing stays local. No document content goes to any cloud API
  (PDPA, to avoid cross-border transfer).
- Never put real company documents in this repo, and never read them during development.
  Use synthetic or redacted samples in /samples. Real test documents live outside the
  repo and are evaluated only by local scripts that output aggregate metrics.
- Hardware: PC with an RTX 4060 Ti (8GB VRAM) and 32GB RAM. Secondary machine:
  MacBook Air M4 (16GB), which can run MLX models. Confirmed setup is under
  "Environment" below.
- The GPU is a single shared resource, so jobs run through a queue, one at a time.

## Environment (confirmed 2026-10-12, GPU PC)
- Windows 11, NVIDIA driver 610.88 (CUDA 13.3; was 566.36 at setup). With GPU-heavy apps
  closed, the desktop uses about 1.3GB of VRAM.
- Sufi wants the pipeline to stay within about 5GB of VRAM: Paddle engines are capped by
  `[gpu] paddle_memory_limit_mb` (FLAGS_gpu_memory_limit_mb), Qwen uses about 3GB and
  loads only after Paddle has released its memory.
- Python 3.11.9 on Windows. WSL2 with Ubuntu 24.04 is installed for the PaddleOCR engines.
- Ollama 0.40.3 on Windows. Qwen3.5-4B Q4_K_M was imported from LM Studio's GGUF as the
  Ollama model `qwen3.5-4b` (about 3.0GB VRAM at 4096 context, 100% GPU).
- Docker is NOT installed. The PaddleOCR engines run in a Python venv inside WSL2
  (`~/venvs/ocrmod`, Python 3.12, paddlepaddle-gpu 3.3.1 cu126; see docs/setup-gpu.md).
  Ubuntu reaches Windows Ollama at localhost via WSL mirrored networking.
- paddleocr now defaults to PP-OCRv6 (medium det/rec) for the line engine, not PP-OCRv5.
  PaddleOCR-VL-1.6 with PP-DocLayoutV3 downloads and loads. With desktop apps open,
  PaddleOCR-VL pushed VRAM to 7.9/8.2GB and ran very slowly: close GPU apps before runs.
- Repo clone on the PC: `C:\Users\sufim\Dev\msufi-OCR` (in WSL:
  `/mnt/c/Users/sufim/Dev/msufi-OCR`). Git commands are run from Windows PowerShell.
- Work laptop (Intel iGPU, 8GB RAM): code review and CPU-only tests only. No models.

## Pipeline (decided)
1. **Ingest:** render each PDF page to an image at 200–300 DPI with PyMuPDF, then fix
   rotation and skew with OpenCV.
2. **Layout parsing:** run the full PaddleOCR-VL-1.6 pipeline (layout detection plus
   VL recognition) to produce Markdown with HTML tables.
   - Backup: GLM-OCR (0.9B) via llama.cpp or Ollama.
   - Long multi-page documents (bank statements, statements of account): Baidu
     Unlimited-OCR. Its weights are about 6.7GB in bf16, so it cannot stay loaded
     alongside Qwen on 8GB. Load the models one after the other, or run it on the
     Mac via MLX. Do not quantize its vision encoder.
3. **Second engine:** PP-OCRv5 provides a confidence score and bounding box per line.
4. **Field extraction:** Qwen3.5-4B Q4_K_M (already installed locally), working on
   the Markdown text only. Output is constrained to a JSON schema (Ollama `format`
   or a llama.cpp grammar), with logprobs requested.
5. **Validation (Python):** line items sum to the total, tax maths is correct, dates
   are valid, the invoice number isn't a duplicate, and the vendor matches the vendor
   master (fuzzy match with rapidfuzz).
6. **Confidence flagging:** combine three signals into a per-field score: validation
   failures, the two OCR engines disagreeing, and the lowest digit-token probability.
   Show red or yellow flags with a reason (for example "sum mismatch" or
   "OCR disagree"), and a crop of the flagged region taken from the bounding boxes.
   Thresholds are calibrated from the test set, not hard-coded.
7. **Review queue:** a person confirms or corrects flagged fields. Corrections are
   stored for later evaluation.
8. **Export:** Excel with colour flags. Later, the company ERP's import format
   (to be confirmed).

## Dashboard MVP
- Batch upload, tagged with company and document type
- Job status: queued, processing, done or failed
- Document view: page image beside the extracted fields, with flags highlighted
- Edit, approve and reject actions, with an audit trail
- Export

## Accuracy method
- Test set of about 50 real documents covering the messiest layouts (kept outside
  the repo, as above)
- Measure field-level accuracy, compare PaddleOCR-VL against GLM-OCR, and run
  Unlimited-OCR on long statements

## Open decisions (ask Sufi before building)
- Web stack: e.g. FastAPI with React or HTMX; .NET with SQL Server to match existing
  internal tools; or an Odoo module
- Database: PostgreSQL or SQL Server
- Authentication, users and multi-company setup
- Deployment: a single PC or a shared server; OS details

## Working style
Plan first. Propose the architecture and folder structure, then build in milestones:
- M1: command-line pipeline working on the sample documents
- M2: API and job queue
- M3: dashboard
- M4: review queue and export
