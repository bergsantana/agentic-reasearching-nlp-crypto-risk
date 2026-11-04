"""A fake vLLM server for tests: no GPU, deterministic answers.

Tokenization is one token per character (its code point), so a letter is one token and appending
it adds exactly that token, as SemIf's boundary check needs. ``/tokenize`` with messages renders
the real vendored chat template, so the template-parity check runs for real. Option logits are a
hash of the prompt and the token, or whatever ``logits_for`` returns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from cryptonlp.classify.prompt import ChatTemplate

LogitsFor = Callable[[str, Sequence[int]], list[float]]


def hashed_logits(prompt: str, token_ids: Sequence[int]) -> list[float]:
    out = []
    for t in token_ids:
        h = hashlib.sha256(f"{prompt}|{t}".encode()).digest()
        out.append(int.from_bytes(h[:4], "big") / 2**32 * 8 - 4)
    return out


class FakeVllm:
    def __init__(
        self,
        served: str,
        template: ChatTemplate,
        *,
        logits_for: LogitsFor = hashed_logits,
        dimensions: int = 1024,
    ) -> None:
        self.served = served
        self.template = template
        self.logits_for = logits_for
        self.dimensions = dimensions
        self.completions = 0
        self.prompts: list[str] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": self.served, "root": "x", "max_model_len": 8192}]}
            )
        body: dict[str, Any] = json.loads(request.content)
        if body.get("model") != self.served:
            return httpx.Response(404, json={"error": f"model {body.get('model')} not served"})
        if path == "/tokenize":
            if "messages" in body:
                text = self.template.render(body["messages"], enable_thinking=False)
            else:
                text = body["prompt"]
            return httpx.Response(200, json={"tokens": [ord(c) for c in text]})
        if path == "/v1/completions":
            self.completions += 1
            self.prompts.append(body["prompt"])
            assert body["max_tokens"] == 1 and body["return_tokens_as_token_ids"]
            ids = body["logprob_token_ids"]
            logits = self.logits_for(body["prompt"], ids)
            top = {f"token_id:{t}": v for t, v in zip(ids, logits, strict=True)}
            return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}]})
        if path == "/v1/embeddings":
            data = []
            for i, text in enumerate(body["input"]):
                h = hashlib.sha256(text.encode()).digest()
                vec = [((h[j % 32] + j) % 17 - 8) / 8 for j in range(self.dimensions)]
                data.append({"index": i, "embedding": vec})
            return httpx.Response(200, json={"data": data})
        return httpx.Response(404)
