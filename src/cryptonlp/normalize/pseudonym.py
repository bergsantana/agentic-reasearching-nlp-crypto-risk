"""Keyed hashes of author IDs (HMAC-SHA256). The key lives outside the repository.

The same key must be used for the whole study: a new key gives every author a new hash, so
accounts could no longer be followed across the change. Back the key up; ``key_id`` (stored
with every row) shows if it ever changed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

KEY_BYTES = 32


class PseudonymKeyError(RuntimeError):
    """The pseudonymization key is missing or malformed."""


class PseudonymKey:
    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise PseudonymKeyError(f"the pseudonymization key must be {KEY_BYTES} bytes")
        self._key = key

    @property
    def key_id(self) -> str:
        """A fingerprint that names the key without revealing it."""
        return hashlib.sha256(b"cryptonlp pseudonym key id\0" + self._key).hexdigest()[:16]

    def author_hash(self, source: str, author_id: str) -> str:
        message = f"{source}\0{author_id}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()


def load_key(path: Path) -> PseudonymKey:
    if not path.exists():
        raise PseudonymKeyError(
            f"no pseudonymization key at {path}. Create one with `cryptonlp keygen` and back it "
            "up outside the repository (set CRYPTONLP_PSEUDONYM_KEY_FILE to use another path)"
        )
    text = path.read_text().strip()
    try:
        key = bytes.fromhex(text)
    except ValueError as exc:
        raise PseudonymKeyError(f"{path} must hold {KEY_BYTES * 2} hex characters") from exc
    return PseudonymKey(key)


def generate_key(path: Path) -> PseudonymKey:
    """Write a new random key, readable only by its owner. Never overwrites an existing key."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = secrets.token_bytes(KEY_BYTES)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PseudonymKeyError(f"{path} already exists; a key is never overwritten") from exc
    with os.fdopen(fd, "w") as f:
        f.write(key.hex() + "\n")
    return PseudonymKey(key)
