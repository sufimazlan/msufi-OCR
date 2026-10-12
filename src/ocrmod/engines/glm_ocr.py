"""GLM-OCR (0.9B) via the local Ollama server: the backup layout engine."""

from __future__ import annotations

import base64

import httpx

from ..schemas import IngestedDoc


class GLMOCREngine:
    name = "glm_ocr"

    def __init__(self, url: str, model: str, prompt: str, timeout: float = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.timeout = timeout

    def parse(self, doc: IngestedDoc) -> list[str]:
        pages = []
        for img in doc.page_images:
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": self.prompt,
                    "images": [base64.b64encode(img.read_bytes()).decode()],
                }],
                "stream": False,
                "options": {"temperature": 0},
            }
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            pages.append(r.json()["message"]["content"])
        return pages

    def close(self) -> None:
        # Ask Ollama to unload the model so the GPU is free for the next stage.
        try:
            httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0},
                       timeout=30)
        except httpx.HTTPError:
            pass
