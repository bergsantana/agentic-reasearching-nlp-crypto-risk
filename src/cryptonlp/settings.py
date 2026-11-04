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
    # Normalization rules (step 3): language, dedupe, entity lexicon, spam rules.
    normalize_config: Path = Path("config/normalize.toml")
    # The hourly forecast loop (step 3b): assets, settle time, baseline window.
    loop_config: Path = Path("config/loop.toml")
    # The labeled set (step 4): sampling plan, re-labeling, split, Label Studio address.
    labeling_config: Path = Path("config/labeling.toml")
    # Label Studio: a personal access token (Account & Settings, in its UI), and the address
    # when it differs from config/labeling.toml (the scheduler reaches it inside Compose).
    label_studio_api_key: str = ""
    label_studio_url: str = ""
    # Password of Label Studio's own database role (set by ``cryptonlp db init``).
    label_studio_db_password: str = "labelstudio"
    # Exports of the labeled set and agreement reports. They hold post URIs: git-ignored.
    labeling_dir: Path = Path("data/labeling")
    # HMAC key for author hashes. Outside the repository, never committed; back it up.
    # Compose mounts it as a secret at /run/secrets/pseudonym_key.
    pseudonym_key_file: Path = Path.home() / ".config" / "cryptonlp" / "pseudonym.key"

    # Study phase stamped on ledger rows the system writes on its own (chain-break violations,
    # interventions logged without --phase). Set to pilot, then confirmatory, in step 9.
    phase: Phase = Phase.DEVELOPMENT
    # Schedule the hourly forecast loop in ``cryptonlp serve``. Off in development except during
    # smoke runs (tech/build-plan.md, "Run mode by phase"): its ledger rows are permanent.
    forecast_loop: bool = False
    # Ledger exports, the weekly anchors sent to the advisor, and the data manifests arm 1's
    # experiment packages cite (manifests/). Git-ignored.
    ledger_dir: Path = Path("data/ledger")
    # MLflow (step 5): walk-forward runs, later the classifier registry. Compose's ``mlflow``.
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    # Walk-forward tables (metrics, comparisons, forecasts), one directory per run. Git-ignored.
    walkforward_dir: Path = Path("data/walkforward")
    # The classifier (step 6): SemIf on vLLM, embeddings, the bake-off. The servers' addresses
    # when they differ from config/classify.toml (the scheduler reaches them inside Compose).
    classify_config: Path = Path("config/classify.toml")
    semif_url: str = ""
    embed_url: str = ""
    # Embed new in-study posts every hour in ``cryptonlp serve`` (needs the `embed` server).
    embeddings: bool = False
    # Bake-off inputs, raw predictions and reports, one directory per run. They hold post URIs
    # and text: git-ignored.
    bakeoff_dir: Path = Path("data/bakeoff")

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
    def walkforward_path(self) -> Path:
        return _resolve(self.walkforward_dir)

    @property
    def normalize_config_path(self) -> Path:
        return _resolve(self.normalize_config)

    @property
    def labeling_config_path(self) -> Path:
        return _resolve(self.labeling_config)

    @property
    def labeling_path(self) -> Path:
        return _resolve(self.labeling_dir)

    @property
    def classify_config_path(self) -> Path:
        return _resolve(self.classify_config)

    @property
    def bakeoff_path(self) -> Path:
        return _resolve(self.bakeoff_dir)

    @property
    def loop_config_path(self) -> Path:
        return _resolve(self.loop_config)

    @property
    def collectors_config_path(self) -> Path:
        return _resolve(self.collectors_config)
