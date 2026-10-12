"""Groq LLM client (OpenAI-compatible Chat Completions API).

Used ONLY off the safety path: the operator assistant, the insights handover note and
the vision verifier's second opinion. Nothing here can raise, lower or clear an alert.

Env:
  GROQ_API_KEY        required to enable LLM features
  GROQ_MODEL          chat + tool use       (default openai/gpt-oss-120b)
  GROQ_REASONING_EFFORT  low | medium | high for gpt-oss models (default medium)
  GROQ_VISION_MODEL   image input           (default qwen/qwen3.8-27b)
  GROQ_VISION_RPM     max vision calls per minute (default 3 = Groq free tier; raise on Dev tier)
  GROQ_BASE_URL       override for a proxy / self-hosted OpenAI-compatible server (vLLM)

Model IDs on Groq change (preview models come and go, and an org can restrict models).
If Groq answers 404 / model_not_found, the client lists the models your key can use
(GET /models), switches to the first available one from a preference list, logs it and
retries once.

Diagnose your key and models:   python -m gajarakshak.llm

The default chat model is open-weights (gpt-oss-120b, Apache-2.0), so production can move the same
model to a self-hosted vLLM endpoint in an Indian region by changing GROQ_BASE_URL only.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import threading
import time

import httpx

log = logging.getLogger("gajarakshak.llm")

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"

# Tried in order when the configured model is not available to this key.
CHAT_FALLBACKS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                  "qwen/qwen3.8-27b", "llama-3.1-8b-instant"]
VISION_FALLBACKS = ["qwen/qwen3.8-27b", "meta-llama/llama-4-scout-17b-16e-instruct",
                    "meta-llama/llama-4-maverick-17b-128e-instruct"]

_THINK = re.compile(r"<think>.*?</think>", re.S)
_TRY_AGAIN = re.compile(r"try again in ([\d.]+)\s*(ms|s)", re.I)


def _retry_after(r: httpx.Response, default: float = 1.0) -> float:
    """Seconds Groq asks us to wait: retry-after header, else the 'try again in Xs' text."""
    try:
        return float(r.headers["retry-after"])
    except (KeyError, ValueError):
        pass
    m = _TRY_AGAIN.search(r.text or "")
    if m:
        v = float(m.group(1))
        return v / 1000 if m.group(2).lower() == "ms" else v
    return default
_MODEL_ERRORS = ("model_not_found", "model_decommissioned", "model_terms_required", "model_permission_blocked")


class LLMError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str = "", code: str = ""):
        super().__init__(msg)
        self.status, self.body, self.code = status, body, code

    @property
    def model_problem(self) -> bool:
        return self.status == 404 or self.code in _MODEL_ERRORS


def strip_think(text: str | None) -> str:
    """Reasoning models may emit <think>…</think>; never show that to an operator."""
    return _THINK.sub("", text or "").strip()


class GroqClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None,
                 vision_model: str | None = None, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None,
                 sync_transport: httpx.BaseTransport | None = None):
        self.api_key = (api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")).strip()
        self.base_url = (base_url or os.environ.get("GROQ_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
        self.model = (model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL).strip()
        self.vision_model = (vision_model or os.environ.get("GROQ_VISION_MODEL") or DEFAULT_VISION_MODEL).strip()
        self.reasoning_effort = (os.environ.get("GROQ_REASONING_EFFORT") or "medium").strip().lower()
        self.timeout = timeout
        self._transport = transport            # tests inject httpx.MockTransport here
        self._sync_transport = sync_transport
        self._available: list[str] | None = None
        self.last_error: str | None = None
        # Pace vision calls: Groq counts every image as ~2k input tokens; the free tier's 7k ITPM is ~3 images/min.
        rpm = float(os.environ.get("GROQ_VISION_RPM") or 3)
        self.vision_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._vision_lock = threading.Lock()
        self._vision_last = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> dict:
        return {"provider": "groq", "enabled": self.enabled, "model": self.model,
                "reasoning_effort": self.reasoning_effort if self.model.startswith("openai/gpt-oss") else None,
                "vision_model": self.vision_model, "base_url": self.base_url, "last_error": self.last_error}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, model, messages, tools, max_tokens, temperature) -> dict:
        body = {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                "temperature": temperature}
        if model.startswith("openai/gpt-oss"):
            # reasoning model: hidden reasoning tokens count against max_completion_tokens
            body["reasoning_effort"] = self.reasoning_effort
            body["max_completion_tokens"] = max(max_tokens, 2048)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def _check(self, r: httpx.Response, model: str) -> dict:
        if r.status_code >= 400:
            msg, code = r.text[:300], ""
            try:
                err = r.json().get("error", {})
                msg, code = err.get("message", msg), err.get("code") or err.get("type") or ""
            except ValueError:
                pass
            self.last_error = f"HTTP {r.status_code} {code}: {msg}"
            log.warning("Groq %s for model %r: %s", r.status_code, model, msg)
            raise LLMError(f"Groq HTTP {r.status_code} ({code or 'error'}): {msg}", r.status_code, r.text[:500], code)
        self.last_error = None
        data = r.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected Groq response: {e}", r.status_code, r.text[:500]) from e

    # ── model discovery ─────────────────────────────────────────────────────
    def _parse_models(self, r: httpx.Response) -> list[str]:
        if r.status_code >= 400:
            self._check(r, "(list models)")
        return [m["id"] for m in r.json().get("data", []) if m.get("active", True)]

    async def list_models(self) -> list[str]:
        if self._available is None:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as c:
                self._available = self._parse_models(await c.get(f"{self.base_url}/models", headers=self._headers()))
        return self._available

    def list_models_sync(self) -> list[str]:
        if self._available is None:
            with httpx.Client(timeout=self.timeout, transport=self._sync_transport) as c:
                self._available = self._parse_models(c.get(f"{self.base_url}/models", headers=self._headers()))
        return self._available

    def _switch(self, available: list[str], vision: bool) -> str | None:
        current = self.vision_model if vision else self.model
        for m in (VISION_FALLBACKS if vision else CHAT_FALLBACKS):
            if m in available and m != current:
                log.warning("Groq model %r is not available to this key; switching to %r", current, m)
                if vision:
                    self.vision_model = m
                else:
                    self.model = m
                return m
        log.error("none of the fallback models are available; models for this key: %s", available)
        return None

    # ── chat ────────────────────────────────────────────────────────────────
    async def chat(self, messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
                   max_tokens: int = 800, temperature: float = 0.2) -> dict:
        """One chat completion. Returns the assistant message dict (may contain tool_calls).
        Retries once on 429 / 5xx; on an unavailable model, switches model once."""
        if not self.enabled:
            raise LLMError("GROQ_API_KEY not set")
        use = model or self.model
        switched = False
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as c:
            for attempt in range(4):
                try:
                    r = await c.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                     json=self._body(use, messages, tools, max_tokens, temperature))
                except httpx.HTTPError as e:
                    if attempt == 0:
                        await asyncio.sleep(0.8)
                        continue
                    raise LLMError(f"network error talking to Groq: {type(e).__name__}") from e
                if r.status_code == 429 and attempt < 3:
                    wait = _retry_after(r)
                    if wait <= 10:                       # interactive: wait briefly, else fail fast
                        log.info("Groq rate limit; retrying in %.1fs", wait)
                        await asyncio.sleep(wait + 0.2)
                        continue
                if r.status_code in (500, 502, 503, 504) and attempt == 0:
                    await asyncio.sleep(0.8)
                    continue
                try:
                    return self._check(r, use)
                except LLMError as e:
                    if e.model_problem and not switched and model is None:
                        switched = True
                        nxt = self._switch(await self.list_models(), vision=False)
                        if nxt:
                            use = nxt
                            continue
                    raise
        raise LLMError("Groq request failed after retries")

    def chat_sync(self, messages: list[dict], model: str | None = None, max_tokens: int = 300,
                  temperature: float = 0.0, vision: bool = False) -> dict:
        """Blocking variant for worker threads and CLI tools (vision verifier, perception demo)."""
        if not self.enabled:
            raise LLMError("GROQ_API_KEY not set")
        use = model or (self.vision_model if vision else self.model)
        switched, last = False, None
        with httpx.Client(timeout=self.timeout, transport=self._sync_transport) as c:
            for attempt in range(5):
                try:
                    r = c.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                               json=self._body(use, messages, None, max_tokens, temperature))
                except httpx.HTTPError as e:
                    last = e
                    time.sleep(0.8)
                    continue
                if r.status_code == 429 and attempt < 4:
                    wait = min(_retry_after(r), 60.0)
                    log.info("Groq rate limit on %s; waiting %.1fs", use, wait)
                    time.sleep(wait + 0.3)
                    continue
                if r.status_code in (500, 502, 503, 504) and attempt == 0:
                    time.sleep(1.0)
                    continue
                try:
                    return self._check(r, use)
                except LLMError as e:
                    if e.model_problem and not switched and model is None:
                        switched = True
                        nxt = self._switch(self.list_models_sync(), vision=vision)
                        if nxt:
                            use = nxt
                            continue
                    raise
        raise LLMError(f"network error talking to Groq: {type(last).__name__}")

    # ── vision ────────────────────────────────────────────────────────────
    def _pace_vision(self) -> None:
        with self._vision_lock:
            wait = self._vision_last + self.vision_interval - time.monotonic()
            if wait > 0:
                log.info("pacing vision call: %.0fs (GROQ_VISION_RPM)", wait)
                time.sleep(wait)
            self._vision_last = time.monotonic()

    def classify_image(self, jpeg: bytes, question: str) -> str:
        self._pace_vision()
        msg = self.chat_sync([{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}], max_tokens=200, vision=True)
        return strip_think(msg.get("content"))


def _diagnose() -> None:
    """python -m gajarakshak.llm — check key, list models, test chat + tool call."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = GroqClient()
    print(f"base_url     {c.base_url}\nkey          {'set (' + c.api_key[:6] + '…)' if c.enabled else 'NOT SET'}")
    print(f"chat model   {c.model}\nvision model {c.vision_model}")
    if not c.enabled:
        return
    try:
        models = c.list_models_sync()
    except LLMError as e:
        print(f"\n✗ cannot list models: {e}\n  → check the key (Groq console › API Keys) and GROQ_BASE_URL")
        return
    print(f"\nmodels available to this key ({len(models)}):\n  " + "\n  ".join(sorted(models)))
    for label, m in (("chat", c.model), ("vision", c.vision_model)):
        print(f"\n{label} model {m!r}: {'✓ available' if m in models else '✗ NOT available'}")
    tool = [{"type": "function", "function": {"name": "get_situation", "description": "status",
                                              "parameters": {"type": "object", "properties": {}}}}]
    try:
        msg = asyncio.run(c.chat([{"role": "user", "content": "Call get_situation."}], tools=tool, max_tokens=50))
        print(f"\n✓ chat + tool call OK on {c.model}: tool_calls={bool(msg.get('tool_calls'))}")
    except LLMError as e:
        print(f"\n✗ chat failed: {e}")


if __name__ == "__main__":
    _diagnose()
