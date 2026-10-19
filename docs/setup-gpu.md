# GPU PC setup (PaddleOCR in WSL2)

Already done (2026-10-12): NVIDIA driver, Git, Python 3.11 on Windows, WSL2 + Ubuntu
24.04, Ollama with `qwen3.5-4b`, repo cloned to `C:\Users\sufim\Dev\msufi-OCR`.

**Why WSL2:** the PaddleOCR-VL pipeline is best supported on Linux. WSL2 runs Ubuntu
inside Windows and uses your Windows NVIDIA driver, so the GPU works there with no extra
driver. The code stays in one place (`C:\Users\sufim\Dev\msufi-OCR`), which Ubuntu sees
as `/mnt/c/Users/sufim/Dev/msufi-OCR`. Keep running git from Windows PowerShell.

**Only model weights are downloaded** (from PaddlePaddle/Hugging Face on first run).
No document content ever leaves the PC.

## 1. Let Ubuntu reach Windows Ollama (mirrored networking)

By default Ubuntu in WSL2 has its own network, so `localhost:11434` inside Ubuntu is not
the Windows Ollama. Mirrored mode makes them share `localhost`.

In **PowerShell**:

```powershell
notepad $env:USERPROFILE\.wslconfig
```

Paste this, save, close Notepad:

```ini
[wsl2]
networkingMode=mirrored
```

Then restart WSL:

```powershell
wsl --shutdown
```

Check from Ubuntu (make sure Ollama is running on Windows: its icon is in the tray):

```powershell
wsl -d Ubuntu-24.04 -- curl -s http://localhost:11434/api/tags
```

You should see JSON that mentions `qwen3.5-4b`. If not, see Troubleshooting below.

## 2. Python environment inside Ubuntu

Open Ubuntu (`wsl -d Ubuntu-24.04` in PowerShell, or "Ubuntu" from the Start menu) and run:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
python3 -m venv ~/venvs/ocrmod
source ~/venvs/ocrmod/bin/activate
cd /mnt/c/Users/sufim/Dev/msufi-OCR
pip install -U pip
pip install -e ".[dev]"
python -m pytest -q
```

The venv lives in the Linux home folder (`~/venvs/ocrmod`) because Python packages on
`/mnt/c` are very slow. Every new Ubuntu window needs:

```bash
source ~/venvs/ocrmod/bin/activate
cd /mnt/c/Users/sufim/Dev/msufi-OCR
```

## 3. PaddlePaddle (GPU) and PaddleOCR

Versions move quickly. Check the current command on the official install page first:
https://www.paddlepaddle.org.cn/en/install/quick (choose: Linux, pip, GPU, CUDA 12.6).
Your driver (CUDA 12.7) can run CUDA 12.6 builds. Installed on the GPU PC (2026-10-13):

```bash
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -c "import paddle; paddle.utils.run_check()"   # expect: works well on 1 GPU
```

This pulls in NVIDIA's CUDA libraries as pip packages, so the venv grows to several GB.
Afterwards `pip cache purge` frees the download cache.

If `import cv2` breaks afterwards (two OpenCV builds installed), fix it with:

```bash
pip uninstall -y opencv-python-headless opencv-contrib-python opencv-python
pip install opencv-contrib-python
```

## 4. Check every model

**Close GPU-heavy apps first, from the system tray too:** Riot Client/Vanguard,
Overwolf, Steam, Discord, LM Studio, extra browser windows. PaddleOCR-VL alone fills most
of the 8GB; when VRAM is full, Windows spills to normal RAM and the GPU slows to a crawl
(seen on the first run: 7.9GB used, GPU-Util 13%).

Paddle is capped at `paddle_memory_limit_mb` (default 4500MB, about 5GB with CUDA's own
overhead) in `config/default.toml`; change it in `config/local.toml`. Over the cap it fails
with an out-of-memory error rather than crawling.

```bash
python scripts/gpu_check.py --skip glm_ocr
python scripts/gpu_check.py --only paddle_vl      # re-run a single step
```

Each step prints a `>>> PASS` / `>>> FAIL` line when it finishes. Ctrl+C stops the
current step and still prints the summary.

It runs each model on the synthetic sample one at a time and prints PASS/FAIL, time and
peak VRAM. **Send Claude the summary table at the end.** The first run downloads the
Paddle models (a few GB), so it is slow once.

GLM-OCR is the backup engine and optional for now. To include it, pull it in Windows
PowerShell first (check the exact name at https://ollama.com/library):
`ollama pull glm-ocr`. Otherwise run with `--skip glm_ocr`.

## 5. Run the pipeline on the samples

```bash
ocrmod process samples/synthetic --company TEST
```

Open the Excel file it prints (it is under `output/`, which Windows Explorer shows at
`C:\Users\sufim\Dev\msufi-OCR\output\`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `curl` to `localhost:11434` fails from Ubuntu | Check `.wslconfig` is saved as `.wslconfig`, not `.wslconfig.txt` (in PowerShell: `dir $env:USERPROFILE\.wslconfig`). Run `wsl --shutdown` again. If mirrored mode still fails, set the Windows environment variable `OLLAMA_HOST=0.0.0.0`, restart Ollama, and put `url = "http://<windows-ip>:11434"` under `[ollama]` in `config/local.toml`. |
| `nvidia-smi` not found in Ubuntu | Update WSL from PowerShell: `wsl --update`, then `wsl --shutdown`. |
| Out of GPU memory | Close GPU-heavy apps. In `config/local.toml` set `keep_alive = "0"` under `[ollama]` so Qwen unloads right after each document. |
| `paddle_vl` hangs after loading: GPU idle, Ctrl+C does nothing | CUDA lazy-loading hang under WSL2. `[gpu] cuda_module_loading = "EAGER"` (default) fixes it; check the first line of `gpu_check.py` says `CUDA_MODULE_LOADING=EAGER`. To see where any hang is: `pip install py-spy`, then from a second Ubuntu window `sudo env "PATH=$PATH" py-spy dump --native --pid $(pgrep -f gpu_check.py \| head -1)`. |
| Out-of-memory error while loading PaddleOCR-VL | Its loader briefly needs about 5.9GB. Keep `paddle_memory_limit_mb` at 6500 or more. |
| `paddle_vl` step fails with an API error | The paddleocr API changed. Send Claude the error from `gpu_check.py`; the wrapper is in `src/ocrmod/engines/paddle_vl.py`. |
