"""SemIf's direct prompt, built exactly as SemIf builds it (Lee, 2026; ``semif_phase1.core``).

A decision is a state (here, the post text), a criterion and 2 to 16 described options. The
options are shown as letters A, B, C...; the model reads the whole prompt once, and the
probability of each option is the softmax of the logits of its letter at the last position.
Nothing is generated.

SemIf renders the prompt with the model's chat template (``apply_chat_template`` in
transformers, thinking off). We render the same template, vendored from the pinned model
revision (``config/semif/``), with the same Jinja settings, so the prompt string, and so its
SHA-256, is byte for byte SemIf's. ``tests/test_semif.py`` checks this against SemIf's own
prompt hashes for its 144 authored decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, NoReturn

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

LETTERS = "ABCDEFGHIJKLMNOP"
# SemIf's system prompt and prompt version, verbatim. Changing either changes every prompt hash.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
PROMPT_VERSION = "direct-options-v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Option:
    id: str
    description: str


@dataclass(frozen=True)
class Decision:
    """One SemIf call: a criterion over a state, with typed options."""

    id: str
    state: str
    question: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.question or not self.state:
            raise ValueError("a decision needs an id, a state and a question")
        if not 2 <= len(self.options) <= len(LETTERS):
            raise ValueError(f"decision {self.id}: 2 to {len(LETTERS)} options")
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"decision {self.id}: option ids must be unique")

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self.options)


def direct_messages(decision: Decision) -> list[dict[str, str]]:
    """SemIf's ``direct_messages``: the system prompt, then the decision as one JSON object
    with the evidence first (so every decision on the same post shares a prefix)."""
    payload = {
        "evidence": decision.state,
        "criterion": decision.question,
        "options": [
            {"letter": LETTERS[i], "description": o.description}
            for i, o in enumerate(decision.options)
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _raise(message: str) -> NoReturn:
    raise jinja2.TemplateError(message)


def _tojson(
    x: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: Any = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


class ChatTemplate:
    """A model's chat template rendered the way transformers renders it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.source = path.read_text(encoding="utf-8")
        self.sha256 = sha256_text(self.source)

    @cached_property
    def _template(self) -> jinja2.Template:
        # transformers' settings (chat_template_utils._compile_jinja_template).
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        return env.from_string(self.source)

    def render(
        self, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool = False
    ) -> str:
        return self._template.render(
            messages=list(messages), add_generation_prompt=True, enable_thinking=enable_thinking
        )


def render_prompt(template: ChatTemplate, decision: Decision) -> str:
    """The exact prompt string SemIf sends for this decision (thinking off)."""
    return template.render(direct_messages(decision), enable_thinking=False)
