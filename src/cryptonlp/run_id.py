"""Run IDs: a UTC timestamp plus the git commit of the code that ran.

Every job prints its run ID and stores it with every result, so each number in the paper
traces back to the exact code that produced it.

Format: ``20261103T142233Z-1a2b3c4d5e6f`` (``-dirty`` appended when tracked code differs
from the commit; ``nogit`` in place of the commit outside a repo or before the first commit).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

RUN_ID_PATTERN = r"^\d{8}T\d{6}Z-(?:[0-9a-f]{12}(?:-dirty)?|nogit)$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
_TIME_FORMAT = "%Y%m%dT%H%M%SZ"

# Only changes here make a run "dirty". Edits to paper/ or tech/ notes don't change results.
CODE_PATHS = ("src", "config", "db", "pyproject.toml", "uv.lock", "docker-compose.yml")

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunIdParts:
    started_at: datetime
    commit: str | None
    dirty: bool


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def git_revision(repo: Path = REPO_ROOT) -> tuple[str | None, bool]:
    """Return (12-character commit hash or None, whether tracked code has uncommitted changes)."""
    commit = _git(repo, "rev-parse", "--short=12", "HEAD")
    if not commit:
        return None, False
    status = _git(repo, "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS)
    return commit, bool(status)


def new_run_id(now: datetime | None = None, repo: Path = REPO_ROOT) -> str:
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("run ID time must be timezone-aware")
    stamp = now.astimezone(UTC).strftime(_TIME_FORMAT)
    commit, dirty = git_revision(repo)
    if commit is None:
        return f"{stamp}-nogit"
    return f"{stamp}-{commit}{'-dirty' if dirty else ''}"


def parse_run_id(run_id: str) -> RunIdParts:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"not a valid run ID: {run_id!r}")
    stamp, rest = run_id.split("-", 1)
    started_at = datetime.strptime(stamp, _TIME_FORMAT).replace(tzinfo=UTC)
    if rest == "nogit":
        return RunIdParts(started_at, None, False)
    commit, _, flag = rest.partition("-")
    return RunIdParts(started_at, commit, flag == "dirty")


def main() -> None:
    print(new_run_id())
