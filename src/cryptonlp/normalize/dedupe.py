"""Near-duplicate search: MinHash signatures with banded locality-sensitive hashing.

Signatures are stored in ``clean.minhash`` for the dedupe window, so each run only loads the
recent ones instead of re-reading old text. A post can only be a duplicate of a post received
no later than itself, so a later post never changes an earlier verdict.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from datasketch import MinHash

from cryptonlp.normalize.config import DedupeConfig
from cryptonlp.normalize.text import for_near_dupes, shingles

SEED = 1  # fixed, so signatures are the same on every run and every machine


def signature(text: str, config: DedupeConfig) -> np.ndarray | None:
    """MinHash of the post's character shingles, or None if it is too short to compare."""
    folded = for_near_dupes(text)
    if len(folded) < config.min_chars:
        return None
    m = MinHash(num_perm=config.num_perm, seed=SEED)
    m.update_batch(sorted(shingles(folded, config.shingle_chars)))
    return np.asarray(m.hashvalues, dtype=np.uint32)


def to_bytes(sig: np.ndarray) -> bytes:
    return sig.astype("<u4").tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<u4").astype(np.uint32)


@dataclass(frozen=True)
class Match:
    root: str  # URI of the original the match leads back to
    score: float  # estimated Jaccard similarity with the matched post


@dataclass(frozen=True)
class _Indexed:
    uri: str
    received_at: datetime
    root: str
    sig: np.ndarray


class NearDuplicateIndex:
    def __init__(self, config: DedupeConfig) -> None:
        self.config = config
        self._items: list[_Indexed] = []
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._items)

    def _bands(self, sig: np.ndarray) -> list[tuple[int, bytes]]:
        r = self.config.rows
        return [(b, sig[b * r : (b + 1) * r].tobytes()) for b in range(self.config.bands)]

    def add(self, uri: str, received_at: datetime, sig: np.ndarray, root: str | None) -> None:
        idx = len(self._items)
        self._items.append(_Indexed(uri, received_at, root or uri, sig))
        for key in self._bands(sig):
            self._buckets[key].append(idx)

    def query(self, sig: np.ndarray, received_at: datetime) -> Match | None:
        """The most similar earlier post at or above the threshold (earliest on a tie)."""
        candidates = {i for key in self._bands(sig) for i in self._buckets.get(key, ())}
        best: tuple[float, datetime, _Indexed] | None = None
        for i in candidates:
            item = self._items[i]
            if item.received_at > received_at:
                continue
            score = float(np.mean(item.sig == sig))
            if score < self.config.threshold:
                continue
            if best is None or (score, -item.received_at.timestamp()) > (
                best[0],
                -best[1].timestamp(),
            ):
                best = (score, item.received_at, item)
        if best is None:
            return None
        return Match(best[2].root, best[0])
