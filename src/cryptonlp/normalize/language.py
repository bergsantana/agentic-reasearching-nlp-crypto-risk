"""Language identification. The study keeps English only (paper, Section 4.4).

Client tags are unreliable (Mastodon clients often default to the account's language, and many
Bluesky posts carry none), so the language is detected from the text with Lingua, which is
built for short texts. Only when too little prose is left after removing URLs, handles, tags,
addresses and numbers is the declared tag used instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from lingua import LanguageDetector, LanguageDetectorBuilder

from cryptonlp.normalize.config import LanguageConfig
from cryptonlp.normalize.text import for_language, letter_count

UNDETERMINED = "und"


@dataclass(frozen=True)
class LanguageResult:
    lang: str  # ISO 639-1 code, or "und"
    method: str  # "detected", "declared" or "none"
    confidence: float | None


@cache
def _detector() -> LanguageDetector:
    # All languages, so a non-English post is never forced into the closest of a short list.
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


def primary_subtag(tag: str) -> str:
    """'en-US' -> 'en'; 'zh-CN' -> 'zh'."""
    return tag.split("-")[0].split("_")[0].strip().lower()


def identify(
    texts: Sequence[str], declared: Sequence[Sequence[str]], config: LanguageConfig
) -> list[LanguageResult]:
    prose = [for_language(t) for t in texts]
    enough = [i for i, p in enumerate(prose) if letter_count(p) >= config.min_letters]
    results: list[LanguageResult | None] = [None] * len(texts)
    if enough:
        values = _detector().compute_language_confidence_values_in_parallel(
            [prose[i] for i in enough]
        )
        for i, ranked in zip(enough, values, strict=True):
            if ranked and ranked[0].value >= config.min_confidence and ranked[0].value > 0:
                top = ranked[0]
                results[i] = LanguageResult(
                    top.language.iso_code_639_1.name.lower(), "detected", float(top.value)
                )
    out = []
    for i, r in enumerate(results):
        if r is None:
            tags = [primary_subtag(t) for t in declared[i] if t.strip()]
            r = (
                LanguageResult(tags[0], "declared", None)
                if tags
                else LanguageResult(UNDETERMINED, "none", None)
            )
        out.append(r)
    return out
