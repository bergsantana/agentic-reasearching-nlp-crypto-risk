"""Settings read from the environment (prefix ``CRYPTONLP_``) or a git-ignored ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from cryptonlp.schemas.common import Phase

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTONLP_", env_file=".env", extra="ignore")

    # Real data: collectors, ledger, labels. Tests and CI must never reach it.
    database_url: str = "postgresql://cryptonlp:cryptonlp@127.0.0.1:5432/cryptonlp"
    # The only database tests, CI and the simulated clock may use. Its name must end in _test.
    test_database_url: str = "postgresql://cryptonlp:cryptonlp@127.0.0.1:5433/cryptonlp_test"
    # Password for the read-only role Grafana uses. Set by ``cryptonlp db init``.
    grafana_db_password: str = "grafana"

    # Raw store (Parquet, never edited) and live recordings for the replay day. Both git-ignored.
    raw_dir: Path = Path("data/raw")
    replay_dir: Path = Path("data/replay")
    # Record every collector's messages into replay_dir (for the frozen replay day).
    record_replay: bool = False
    # Collector settings (watch terms, instances, symbols). Agents get read-only access.
    collectors_config: Path = Path("config/collectors.toml")

    # Study phase stamped on ledger rows the system writes on its own (chain-break violations,
    # interventions logged without --phase). Set to pilot, then confirmatory, in step 9.
    phase: Phase = Phase.DEVELOPMENT
    # Ledger exports and the weekly anchors sent to the advisor. Git-ignored.
    ledger_dir: Path = Path("data/ledger")

    @property
    def raw_path(self) -> Path:
        return _resolve(self.raw_dir)

    @property
    def replay_path(self) -> Path:
        return _resolve(self.replay_dir)

    @property
    def ledger_path(self) -> Path:
        return _resolve(self.ledger_dir)

    @property
    def collectors_config_path(self) -> Path:
        return _resolve(self.collectors_config)
