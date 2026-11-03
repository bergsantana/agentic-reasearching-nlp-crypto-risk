from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from cryptonlp.run_id import new_run_id, parse_run_id

NOW = datetime(2026, 11, 3, 14, 22, 33, tzinfo=UTC)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "notes.md").write_text("notes\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repo(repo: Path) -> None:
    run_id = new_run_id(NOW, repo)
    parts = parse_run_id(run_id)
    assert run_id.startswith("20261103T142233Z-")
    assert parts.started_at == NOW
    assert parts.commit is not None and len(parts.commit) == 12
    assert not parts.dirty


def test_code_change_marks_dirty(repo: Path) -> None:
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert new_run_id(NOW, repo).endswith("-dirty")
    assert parse_run_id(new_run_id(NOW, repo)).dirty


def test_non_code_change_stays_clean(repo: Path) -> None:
    (repo / "notes.md").write_text("edited\n")
    assert not new_run_id(NOW, repo).endswith("-dirty")


def test_outside_a_repo(tmp_path: Path) -> None:
    run_id = new_run_id(NOW, tmp_path)
    assert run_id == "20261103T142233Z-nogit"
    assert parse_run_id(run_id).commit is None


def test_time_is_converted_to_utc(tmp_path: Path) -> None:
    local = NOW.astimezone(timezone(timedelta(hours=-4)))
    assert new_run_id(local, tmp_path) == "20261103T142233Z-nogit"


def test_naive_time_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        new_run_id(datetime(2026, 11, 3), tmp_path)  # noqa: DTZ001


@pytest.mark.parametrize("bad", ["", "2026-11-03-abc", "20261103T142233Z-XYZ", "20261103T142233Z"])
def test_parse_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_run_id(bad)
