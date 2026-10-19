"""Step 4: field extraction with Qwen via Ollama, on the Markdown text only.

The output is forced into InvoiceExtraction's JSON schema (Ollama `format`), and we ask
for per-token log-probabilities. Afterwards each token is mapped back to the JSON field it
belongs to, so every field gets "the probability of its least-certain digit".
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .schemas import InvoiceExtraction

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract fields from supplier invoices for an accounts payable team.
Rules:
- Copy values exactly as printed. Never calculate, guess or fill in a missing value: use null.
- Amounts: plain numbers without currency symbols or thousands separators (1234.50).
- Dates: convert to YYYY-MM-DD. Printed dates are day-first (05/10/2026 is 5 October 2026).
- line_items: one entry per printed line item, in order. Skip subtotal/tax/total rows.
"""


def inline_schema(schema: dict) -> dict:
    """Resolve $ref/$defs so the grammar converter gets a self-contained schema."""
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def require_all(schema: dict) -> dict:
    """Make every property required (values stay nullable) and drop defaults.

    Pydantic leaves fields with defaults out of "required". The grammar Ollama builds from
    the schema then lets the model close the JSON object early: on the first real sample
    Qwen wrote vendor_name, vendor_tax_id and invoice_number and stopped, so every later
    field came back null. With all keys required, the model must write each one in order
    and can only choose null as the value.
    """
    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
        return node

    return visit(copy.deepcopy(schema))


EXTRACTION_SCHEMA = require_all(inline_schema(InvoiceExtraction.model_json_schema()))


_HTML_NOISE = re.compile(r"""\s+(?:style|border|class|align|width|height)=(?:'[^']*'|"[^"]*"|\S+)""")


def compact_html(markdown: str) -> str:
    """Drop presentation attributes (PaddleOCR-VL adds a long style= to every table cell);
    they are over half the characters of a typical invoice table and carry no content."""
    return _HTML_NOISE.sub("", markdown)


def build_prompt(markdown: str, max_chars: int) -> tuple[str, bool]:
    markdown = compact_html(markdown)
    truncated = len(markdown) > max_chars
    body = markdown[:max_chars]
    return f"Document (Markdown, tables as HTML):\n\n{body}\n\nReturn the JSON.", truncated


# ---- Map tokens back to JSON leaf values ----

def leaf_spans(text: str) -> dict[str, tuple[int, int]]:
    """Character span of every scalar value in a JSON string, keyed by dotted path.

    A tiny recursive-descent scan; json.loads cannot report offsets.
    """
    spans: dict[str, tuple[int, int]] = {}
    i = 0

    def ws():
        nonlocal i
        while i < len(text) and text[i] in " \t\r\n":
            i += 1

    def string_end(start: int) -> int:
        j = start + 1
        while text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        return j + 1

    def value(path: str):
        nonlocal i
        ws()
        c = text[i]
        if c == "{":
            i += 1
            ws()
            if text[i] == "}":
                i += 1
                return
            while True:
                ws()
                end = string_end(i)
                key = json.loads(text[i:end])
                i = end
                ws()
                i += 1  # colon
                value(f"{path}.{key}" if path else key)
                ws()
                if text[i] == ",":
                    i += 1
                    continue
                i += 1  # closing brace
                return
        if c == "[":
            i += 1
            ws()
            if text[i] == "]":
                i += 1
                return
            n = 0
            while True:
                value(f"{path}.{n}")
                n += 1
                ws()
                if text[i] == ",":
                    i += 1
                    continue
                i += 1
                return
        start = i
        if c == '"':
            i = string_end(i)
        else:
            while i < len(text) and text[i] not in ",}] \t\r\n":
                i += 1
        spans[path] = (start, i)

    value("")
    return spans


def field_token_probs(content: str, logprobs: list[dict]) -> dict[str, float]:
    """Lowest token probability per field. Digit tokens only when the value has digits."""
    tokens = [(t.get("token", ""), t.get("logprob")) for t in logprobs]
    rebuilt = "".join(tok for tok, _ in tokens)
    if rebuilt != content:
        # Leading whitespace or thinking text can differ; align on the JSON start.
        offset = rebuilt.find(content)
        if offset < 0:
            log.warning("Logprob tokens do not match the content; skipping token signal")
            return {}
    else:
        offset = 0
    positions, pos = [], 0
    for tok, lp in tokens:
        positions.append((pos - offset, pos - offset + len(tok), tok, lp))
        pos += len(tok)
    out: dict[str, float] = {}
    for path, (start, end) in leaf_spans(content).items():
        inside = [(tok, lp) for s, e, tok, lp in positions if s < end and e > start and lp is not None]
        if not inside:
            continue
        digits = [lp for tok, lp in inside if any(ch.isdigit() for ch in tok)]
        chosen = digits or [lp for _, lp in inside]
        out[path] = math.exp(min(chosen))
    return out


# ---- Extractors ----

@dataclass
class ExtractionResult:
    data: dict[str, Any]
    token_probs: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    raw: str = ""  # the model's JSON exactly as returned


class Extractor(Protocol):
    name: str

    def extract(self, markdown: str) -> ExtractionResult: ...


class OllamaExtractor:
    name = "ollama"

    def __init__(self, url: str, model: str, num_ctx: int = 8192, keep_alive: str = "5m",
                 timeout: float = 300):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout

    def check(self) -> None:
        """Fail fast, before minutes of OCR, if Ollama or the model is missing."""
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=10)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url} ({exc}). Is Ollama running? "
                "From WSL2 this needs mirrored networking: see docs/setup-gpu.md") from exc
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(f"Ollama has no model {self.model!r}. Installed: {sorted(names)}")

    def extract(self, markdown: str) -> ExtractionResult:
        # Roughly 3 characters per token, leaving room for the prompt and the answer.
        prompt, truncated = build_prompt(markdown, max_chars=int(self.num_ctx * 3 * 0.6))
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "format": EXTRACTION_SCHEMA,
            "stream": False,
            "think": False,
            "logprobs": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        content = body["message"]["content"]
        data = InvoiceExtraction.model_validate_json(content).model_dump()
        warnings = ["Document text was truncated to fit the model context"] if truncated else []
        missing = [k for k in EXTRACTION_SCHEMA["properties"] if k not in json.loads(content)]
        if missing:
            warnings.append(f"Model left out fields (treated as empty): {', '.join(missing)}")
        logprobs = body.get("logprobs")
        if logprobs:
            probs = field_token_probs(content, logprobs)
        else:
            probs = {}
            warnings.append("Ollama returned no logprobs; token-probability signal missing")
        return ExtractionResult(data=data, token_probs=probs, warnings=warnings, raw=content)

    def unload(self) -> None:
        try:
            httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0},
                       timeout=30)
        except httpx.HTTPError:
            pass
