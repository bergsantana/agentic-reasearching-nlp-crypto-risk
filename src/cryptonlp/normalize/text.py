"""Text helpers shared by the language filter, the entity rules and dedupe."""

from __future__ import annotations

import hashlib
import re
import unicodedata

URL = re.compile(r"\bhttps?://\S+|\bwww\.[^\s/]+\.\S+", re.IGNORECASE)
# @alice, @alice.bsky.social, @alice@mastodon.social
MENTION = re.compile(r"(?<![\w@])@[\w.-]+(?:@[\w-]+(?:\.[\w-]+)+)?")
CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z][A-Za-z0-9]{1,9})(?![\w])")
HASHTAG = re.compile(r"(?<![\w#&])#(\w*[^\W\d_]\w*)")
_HEXISH = re.compile(r"\b0x[0-9a-fA-F]+\b")
_HAS_DIGIT = re.compile(r"\S*\d\S*")
_SPACE = re.compile(r"\s+")


def strip_nul(text: str) -> str:
    """Drop NUL characters: PostgreSQL text cannot hold them, and they carry no meaning.

    Some clients mangle escapes (``\\u00e9`` arrives as NUL followed by ``e9``); the raw
    store keeps the post as received, the clean copy loses only the NULs.
    """
    return text.replace("\x00", "")


def fold(text: str) -> str:
    """NFKC, case-folded, whitespace collapsed: what "the same text" means for dedupe."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def text_sha256(text: str) -> str:
    return hashlib.sha256(fold(text).encode("utf-8")).hexdigest()


def without_urls(text: str) -> str:
    return URL.sub(" ", text)


def for_language(text: str) -> str:
    """The prose of a post: no URLs, handles, cashtags, hashtags, addresses or numbers."""
    for pattern in (URL, MENTION, CASHTAG, HASHTAG, _HEXISH, _HAS_DIGIT):
        text = pattern.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def for_near_dupes(text: str) -> str:
    """Folded text without URLs and handles, which differ between otherwise identical copies."""
    return fold(MENTION.sub(" ", without_urls(text)))


def shingles(text: str, k: int) -> set[bytes]:
    if len(text) <= k:
        return {text.encode("utf-8")}
    return {text[i : i + k].encode("utf-8") for i in range(len(text) - k + 1)}
