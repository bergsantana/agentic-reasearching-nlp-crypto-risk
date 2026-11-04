"""A small client for the vLLM servers (OpenAI-compatible API): SemIf's option logits and the
embeddings.

SemIf's readout over HTTP: the prompt goes to ``/v1/completions`` as a string (rendered here,
``classify.prompt``), with ``max_tokens = 1`` and ``logprob_token_ids`` set to the option
letters. The server runs with ``--logprobs-mode raw_logits``, so what comes back for those
tokens is the model's raw last-position logits, the numbers SemIf softmaxes. Nothing depends on
the token the server samples.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx

from cryptonlp.classify.errors import ClassifyError

T = TypeVar("T")
R = TypeVar("R")

TOKEN_KEY = "token_id:{}"


class VllmError(ClassifyError):
    pass


@dataclass(frozen=True)
class ServedModel:
    id: str
    root: str  # the model path or repo the server loaded
    max_model_len: int | None


class VllmClient:
    """Thread-safe: one ``httpx.Client`` shared by a pool of worker threads."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120,
        concurrency: int = 32,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> VllmClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                r = self._http.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise VllmError(f"{path}: {exc}") from exc
                time.sleep(1 + attempt)
                continue
            if r.status_code >= 500 and attempt < 2:
                time.sleep(1 + attempt)
                continue
            if r.status_code != 200:
                raise VllmError(f"{path}: HTTP {r.status_code}: {r.text[:500]}")
            data: dict[str, Any] = r.json()
            return data
        raise AssertionError("unreachable")

    # --- server ------------------------------------------------------------------------------

    def served(self) -> ServedModel:
        try:
            r = self._http.get("/v1/models")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise VllmError(f"vLLM server not reachable: {exc}") from exc
        for m in r.json().get("data", []):
            if m.get("id") == self.model:
                return ServedModel(m["id"], str(m.get("root") or ""), m.get("max_model_len"))
        raise VllmError(f"the server does not serve {self.model!r}")

    def tokenize(self, prompt: str) -> list[int]:
        data = self._post(
            "/tokenize", {"model": self.model, "prompt": prompt, "add_special_tokens": False}
        )
        return [int(t) for t in data["tokens"]]

    def tokenize_chat(self, messages: Sequence[dict[str, str]]) -> list[int]:
        """The server's own rendering of the chat template (thinking off), as token IDs."""
        data = self._post(
            "/tokenize",
            {
                "model": self.model,
                "messages": list(messages),
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return [int(t) for t in data["tokens"]]

    # --- SemIf ---------------------------------------------------------------------------------

    def option_logits(self, prompt: str, token_ids: Sequence[int]) -> list[float]:
        """Raw last-position logits of ``token_ids`` after ``prompt``."""
        data = self._post(
            "/v1/completions",
            {
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": False,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 1,
                "logprob_token_ids": list(token_ids),
                "return_tokens_as_token_ids": True,
                # No generation_config defaults (repetition penalty and the like) from the model.
                "repetition_penalty": 1.0,
            },
        )
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmError(f"no logprobs in the response: {str(data)[:300]}") from exc
        missing = [t for t in token_ids if TOKEN_KEY.format(t) not in top]
        if missing:
            raise VllmError(f"logits missing for tokens {missing}")
        return [float(top[TOKEN_KEY.format(t)]) for t in token_ids]

    # --- embeddings ----------------------------------------------------------------------------

    def embed(self, texts: Sequence[str], *, max_tokens: int | None = None) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if max_tokens is not None:
            body["truncate_prompt_tokens"] = max_tokens
        data = self._post("/v1/embeddings", body)
        rows = sorted(data["data"], key=lambda d: d["index"])
        if len(rows) != len(texts):
            raise VllmError(f"{len(rows)} embeddings for {len(texts)} texts")
        return [[float(x) for x in d["embedding"]] for d in rows]

    # --- many at once --------------------------------------------------------------------------

    def map(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """``fn`` over ``items`` with ``concurrency`` requests in flight, results in order."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(items))) as pool:
            return list(pool.map(fn, items))
