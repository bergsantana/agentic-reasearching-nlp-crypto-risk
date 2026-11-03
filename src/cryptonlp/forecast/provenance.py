"""Data manifests: the list of raw-store files a fit or a walk-forward run read, with hashes.

A weekly arm 1 fit reads every candle file since ``train_start`` (hundreds of files), too many
to list in each ledger entry. The list is written once as a manifest file named after its own
SHA-256, and the ledger entry cites the manifest. The validator agent checks the manifest's
hash, then each file's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from cryptonlp.schemas.ledger import DataFile, sha256_text

MANIFEST_DIR = "manifests"


def manifest_text(files: Iterable[DataFile]) -> str:
    unique = sorted({(f.path, f.sha256) for f in files})
    body = [{"path": path, "sha256": sha} for path, sha in unique]
    return json.dumps({"files": body}, sort_keys=True, separators=(",", ":"))


def write_manifest(files: Iterable[DataFile], root: Path) -> DataFile:
    """Write the manifest under ``root/manifests/`` (if new) and return it as a ``DataFile``
    whose path is relative to ``root`` (the ledger directory)."""
    text = manifest_text(files)
    digest = sha256_text(text)
    relative = f"{MANIFEST_DIR}/{digest}.json"
    path = root / relative
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return DataFile(path=relative, sha256=digest)
