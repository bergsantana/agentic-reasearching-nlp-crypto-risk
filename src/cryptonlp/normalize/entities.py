"""Rule-based entities: assets from a lexicon, cashtags, hashtags and on-chain addresses.

Entity linking is deterministic and auditable, so it is done with rules, not by the classifier
(claims.md, D4). Addresses are validated (checksums for Bitcoin, decoded length for Solana) so
that random strings are not counted.
"""

from __future__ import annotations

import hashlib
import re

from cryptonlp.normalize.config import EntitiesConfig
from cryptonlp.normalize.text import CASHTAG, HASHTAG, without_urls


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def cashtags(text: str) -> tuple[str, ...]:
    return _unique(["$" + m.group(1).upper() for m in CASHTAG.finditer(without_urls(text))])


def hashtags(text: str) -> tuple[str, ...]:
    return _unique([m.group(1).lower() for m in HASHTAG.finditer(without_urls(text))])


class AssetMatcher:
    """Whole-word, case-insensitive aliases, with an optional leading $ or #."""

    def __init__(self, config: EntitiesConfig) -> None:
        self._patterns = []
        for asset, aliases in sorted(config.assets.items()):
            words = "|".join(
                re.escape(a.lower()).replace(r"\ ", r"[\s-]+")
                for a in sorted(set(aliases), key=len, reverse=True)
            )
            self._patterns.append(
                (asset, re.compile(rf"(?<![\w$#])[$#]?(?:{words})(?![\w])", re.IGNORECASE))
            )

    def assets(self, text: str) -> tuple[str, ...]:
        clean = without_urls(text)
        return tuple(asset for asset, pattern in self._patterns if pattern.search(clean))


# --- addresses -----------------------------------------------------------------------------

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58)}
_BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_BECH32_GEN = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
_BECH32_CONST = 1
_BECH32M_CONST = 0x2BC830A3

EVM = re.compile(r"(?<![0-9A-Za-z])0x[0-9a-fA-F]{40}(?![0-9A-Za-z])")
BTC_BECH32 = re.compile(r"(?<![0-9A-Za-z])(?:bc1|BC1)[0-9A-Za-z]{11,71}(?![0-9A-Za-z])")
BASE58 = re.compile(r"(?<![0-9A-Za-z])[1-9A-HJ-NP-Za-km-z]{25,44}(?![0-9A-Za-z])")


def b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + _B58_INDEX[c]
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\0" * (len(s) - len(s.lstrip("1"))) + body


def is_btc_base58(s: str) -> bool:
    """P2PKH or P2SH mainnet address with a valid Base58Check checksum."""
    if s[0] not in "13":
        return False
    raw = b58decode(s)
    if len(raw) != 25 or raw[0] not in (0x00, 0x05):
        return False
    return hashlib.sha256(hashlib.sha256(raw[:-4]).digest()).digest()[:4] == raw[-4:]


def _polymod(values: list[int]) -> int:
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i, g in enumerate(_BECH32_GEN):
            if (top >> i) & 1:
                chk ^= g
    return chk


def is_btc_bech32(s: str) -> bool:
    """SegWit (bech32) or Taproot (bech32m) mainnet address with a valid checksum."""
    if s.lower() != s and s.upper() != s:
        return False
    s = s.lower()
    hrp, _, data = s.rpartition("1")
    if hrp != "bc" or len(data) < 7 or any(c not in _BECH32 for c in data):
        return False
    expanded = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    return _polymod(expanded + [_BECH32.index(c) for c in data]) in (
        _BECH32_CONST,
        _BECH32M_CONST,
    )


def is_solana(s: str) -> bool:
    """A Base58 string that decodes to a 32-byte key, with mixed case and a digit.

    Solana addresses carry no checksum; requiring the mix keeps ordinary words and lower-case
    identifiers out.
    """
    if not 32 <= len(s) <= 44:
        return False
    if not (any(c.isdigit() for c in s) and any(c.islower() for c in s)):
        return False
    if not any(c.isupper() for c in s):
        return False
    return len(b58decode(s)) == 32


def addresses(text: str) -> tuple[str, ...]:
    """'evm:0x...' (lower case), 'btc:...', 'sol:...', in order found."""
    clean = without_urls(text)
    found: list[tuple[int, str]] = []
    for m in EVM.finditer(clean):
        found.append((m.start(), "evm:" + m.group(0).lower()))
    for m in BTC_BECH32.finditer(clean):
        if is_btc_bech32(m.group(0)):
            found.append((m.start(), "btc:" + m.group(0).lower()))
    for m in BASE58.finditer(clean):
        s = m.group(0)
        if is_btc_base58(s):
            found.append((m.start(), "btc:" + s))
        elif is_solana(s):
            found.append((m.start(), "sol:" + s))
    return _unique([a for _, a in sorted(found)])
