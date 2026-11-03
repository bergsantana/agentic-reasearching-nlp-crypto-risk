"""Evident spam: a few conservative pattern rules from ``config/normalize.toml``."""

from __future__ import annotations

import re

from cryptonlp.normalize.config import SpamConfig


class SpamFilter:
    def __init__(self, config: SpamConfig) -> None:
        self.config = config
        flags = re.IGNORECASE | re.DOTALL
        self._rules = [
            (
                r.name,
                re.compile(r.pattern, flags),
                r.requires_url,
                re.compile(r.unless, flags) if r.unless else None,
            )
            for r in config.rules
        ]

    def rules_matched(
        self, text: str, *, has_url: bool, n_cashtags: int, n_hashtags: int
    ) -> tuple[str, ...]:
        matched = [
            name
            for name, pattern, requires_url, unless in self._rules
            if (has_url or not requires_url)
            and pattern.search(text)
            and not (unless and unless.search(text))
        ]
        if n_cashtags > self.config.max_cashtags:
            matched.append("too_many_cashtags")
        if n_hashtags > self.config.max_hashtags:
            matched.append("too_many_hashtags")
        return tuple(matched)
