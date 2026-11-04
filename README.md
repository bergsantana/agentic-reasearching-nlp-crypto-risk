# Crypto NLP research agents (TCC)

Autonomous AI agents that run a pre-registered study: do Bluesky and Mastodon posts add forecasting
value for Bitcoin and Ethereum risk beyond price-only statistics?

- System design: [Crypto NLP Research Agents System Design.md](Crypto%20NLP%20Research%20Agents%20System%20Design.md)
- Stack and build guide: [tech/stack-and-build-guide.md](tech/stack-and-build-guide.md)
- Build plan and checklist: [tech/build-plan.md](tech/build-plan.md)
- Paper: [paper/](paper/) (rules in `paper/CLAUDE.md`)

## Setup

```bash
uv sync                          # Python 3.12 and pinned dependencies (uv.lock)
cp .env.example .env             # then change the passwords and set the Telegram bot
docker compose up -d --wait      # postgres, postgres-test (tests and CI only) and Grafana
```

## Collecting (step 1)

```bash
docker compose --profile collect up -d --build   # Bluesky stream, scheduler, Prefect
docker compose --profile collect run --rm scheduler collect binance-backfill  # first time
docker compose --profile collect stop             # stop collecting (buffers are flushed)
```

| UI | Address | What for |
| --- | --- | --- |
| Grafana | http://127.0.0.1:3000 | System health: collector lag, posts per hour, job runs, raw store, deletions. Alerts to Telegram. |
| Prefect | http://127.0.0.1:4200 | Scheduled runs, retries, logs |

What runs:

| Job | When (UTC) | Writes |
| --- | --- | --- |
| `collector-bluesky` | continuously | Jetstream posts matching `config/collectors.toml`, every 10 minutes; deletions in the stream |
| `collect-mastodon` | every 5 minutes | hashtag timelines on the configured instances |
| `collect-trends` | hourly at :01 | Bluesky and Mastodon trending topics, as received |
| `collect-binance-tail` | hourly at :02 | closed 1m and 1h candles from the API, re-pulling the last 3 hours |
| `collect-binance-backfill` | daily at 03:30 | Vision archives (about 2 years back), with their checksums |
| `deletion-sync` | daily at 04:00 | posts deleted at the source, then purged from the raw files |
| `rawstore-verify` | daily at 05:30 | recomputes every raw file's SHA-256 against the manifest |

Every job can also be run by hand, e.g. `uv run cryptonlp collect mastodon`,
`uv run cryptonlp deletions`, `uv run cryptonlp rawstore verify` (`uv run cryptonlp --help`).

**Raw store.** `data/raw/<dataset>/[symbol=…/interval=…/]date=YYYY-MM-DD/*.parquet`, each with a
`.sha256` sidecar that `sha256sum -c` accepts, listed in `raw.files`. Files are read-only and
never edited. Honoring a deletion writes a successor file without the deleted posts and records
which file it supersedes; the old hash stays in `raw.files`. Posts keep the event exactly as
received (`payload`) and the watch terms that matched (`matched`); every file records the hash
of the watch list.

**Replay day.** See [tests/fixtures/replay/README.md](tests/fixtures/replay/README.md).

## Normalization (step 3)

Every raw post gets one row in `clean.posts`, every 10 minutes: detected language, assets (BTC/ETH),
cashtags, hashtags, on-chain addresses, spam rules matched, exact or near duplicate of which post,
and the author as an HMAC hash. `in_study` (English, not spam, not a duplicate, live) is what the
study reads, through the view `clean.study_posts`. Rules and thresholds: `config/normalize.toml`.

```bash
uv run cryptonlp keygen                  # once: the pseudonymization key, outside the repo. Back it up
uv run cryptonlp normalize run           # normalize what is pending (also every 10 minutes)
uv run cryptonlp normalize sample        # CSV for the hand review, in data/review/ (holds text)
uv run cryptonlp normalize rebuild --yes # after changing config/normalize.toml (same key!)
```

A deletion at the source removes the row at once. The same key must be used for the whole study;
every row stores its fingerprint (`key_id`).

## Hourly loop (step 3b)

One cycle, at minute 25 of every hour: label new in-study posts (`clean.labels`), write the NLP
features of every bar that has closed and settled (`features.hourly`), forecast the next bar into
the ledger before it opens, and score every forecast whose bar has closed. Every step reads the
database clock, the one the ledger checks deadlines against. Forecasters: the "yesterday's
volatility" baseline and arm 1 (step 5, below). The classifier is SemIf (step 6, below), or the
step 3b stand-in `StubClassifier` (VADER and keyword rules, `model_version = "stub"`), as
`config/classify.toml` `[loop] classifier` says. Settings: `config/loop.toml`.

```bash
uv run cryptonlp loop run --test         # one cycle against the test database
uv run cryptonlp loop simulate 2026-11-05T00:00Z 2026-11-06T00:00Z   # replayed day, simulated clock
uv run cryptonlp loop run --yes          # one cycle against the main ledger (rows are permanent)
```

The scheduler runs it only when `CRYPTONLP_FORECAST_LOOP=true` in `.env`: switch it on for a smoke
run (`docker compose --profile collect up -d`), and off again afterwards. Grafana's "Forecast loop"
row shows forecasts on time, late, missed and still unscored.

## Statistics arm (step 5)

Arm 1: GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), HAR (Equation 5 and a log variant) and historical
VaR, forecasting next-hour RV, 1% and 5% VaR and the probability of a large move
(`src/cryptonlp/forecast/arm1/`, settings in `config/loop.toml` `[arm1]`). Models are re-estimated
every Monday on an expanding window, and the forecast for bar t uses bars up to t-2 only. In the
loop, each weekly fit goes into the ledger as an experiment package, which the week's forecasts
cite.

```bash
docker compose up -d mlflow              # MLflow at http://127.0.0.1:5000
uv run cryptonlp arm1 walkforward        # walk-forward on development data (read-only), logged to MLflow
uv run cryptonlp arm1 walkforward --asset BTC --no-mlflow   # tables only, in data/walkforward/<run ID>/
```

The walk-forward fits each weekly block on the bars before it and scores every model against the
baseline (QLIKE, MSE, Diebold-Mariano), the VaR (Kupiec, Christoffersen, tick loss) and the
large-move probability (Brier and its decomposition). It never writes to the ledger.

## Labeling (step 4)

2,000 in-study posts labeled by hand for the five tasks in Label Studio (http://127.0.0.1:8080,
profile `collect`), then 200 re-labeled blind at least 14 days later. The manual,
[src/cryptonlp/labeling/manual.md](src/cryptonlp/labeling/manual.md), defines the labels and the
sampling plan. Register it in the ledger **before the first post is labeled**. Settings:
`config/labeling.toml`. Tables: schema `labeling` (`db/init/006_labeling.sql`).

```bash
# once: log in to Label Studio, create a personal access token (Account & Settings), and put it in
# .env as CRYPTONLP_LABEL_STUDIO_API_KEY; then `docker compose --profile collect up -d`
uv run cryptonlp ledger register-document src/cryptonlp/labeling/manual.md --key labeling-manual \
    --kind labeling_manual --title "Labeling manual" --version 1.0
uv run cryptonlp labeling draw           # each day: 500 posts from yesterday (UTC) -> new project
uv run cryptonlp labeling pull           # after each session (the scheduler also runs it hourly)
uv run cryptonlp labeling status         # progress, posts per hour, invalid annotations
uv run cryptonlp labeling relabel        # from 14 days after the first labels: 200 posts, blind
uv run cryptonlp labeling agreement      # Cohen's kappa per task -> data/labeling/
uv run cryptonlp labeling freeze         # once: the time split (500 calibration, rest test)
uv run cryptonlp labeling freeze --relabel
```

Each day is drawn in strata by platform and by candidate status (scam or promotion candidates get
10% of the batch), with each stratum's N and n recorded for re-weighting. Label Studio sees only
`item_id` and the text. Annotations stay editable until `freeze`, which writes them to
`clean.labels` (`human:pass1`, `human:relabel`). Until then, `labeling.annotations` keeps every
version. A deletion at the source removes the item, its annotations and its Label Studio task.
`draw`, `relabel`, `push`, `status`, `agreement` and `freeze` take `--test`: the test database,
with "TEST" in the Label Studio project titles.

## Classifier (step 6)

SemIf (Lee, 2026) on the base model it was published with, Qwen3.5-4B at revision `851bf6e8`,
served by vLLM (Compose profile `gpu`, port 8001). SemIf's prompt is built here exactly as SemIf
builds it, and its answer is read the same way: the raw logits of the option letters, softmaxed.
Each post gets nine decisions from the rubric (`config/semif/rubric.toml`): sentiment, scam, event,
and one yes/no per topic. Labels go to `clean.labels` with a model version that changes whenever
the revision, the rubric or the calibration temperatures do. Settings: `config/classify.toml`.
Embeddings (Qwen3-Embedding-0.6B, port 8002) go to `clean.embeddings` (pgvector) for the Analyst.

```bash
docker compose --profile gpu up -d --wait   # SemIf and the embeddings (GPU; first start ~5 min)
uv run cryptonlp classify check             # pinned revisions served, prompts render as the server's
uv run cryptonlp classify conformance       # SemIf's 144 published decisions through our readout
uv run cryptonlp classify run [--limit N]   # label in-study posts not labeled yet (the whole store)
uv run cryptonlp classify embed             # embed posts without a vector
```

The bake-off (H4) scores SemIf, Wortega's OpenJev, CryptoBERT, FinBERT and VADER on the frozen
labeled set. Tune the rubric on the calibration slice only, register H4, then score the test split:

```bash
uv run cryptonlp classify bakeoff export --split calibration   # -> data/bakeoff/<run ID>/
uv run cryptonlp classify bakeoff semif <run ID>
uv run cryptonlp classify bakeoff vader <run ID>
docker compose --profile gpu stop           # the scorer needs the GPU and RAM to itself
docker compose run --rm hf-scorer cryptobert data/bakeoff/<run ID>   # also finbert, openjev
uv run cryptonlp classify bakeoff evaluate <run ID>   # temperatures, metrics, report, MLflow
# set [semif.temperatures] and decide [h4] (margin, fallback threshold, decided = true), commit
uv run cryptonlp classify register-h4 --labeled-set data/labeling/labeled-set-<run ID>.parquet --yes
uv run cryptonlp classify bakeoff export --split all   # refused until H4 is registered
# ... score every model as above, then:
uv run cryptonlp classify bakeoff evaluate <run ID> --register   # H4 test, fallback, MLflow registry
```

Docker on this machine shares one 6 GB RAM cap across containers: never run `hf-scorer` while the
`gpu` profile is up (the scorer refuses if the GPU or RAM is too full).

## Pre-registration ledger (step 2)

Schema `ledger` in the main database (`db/init/003_ledger.sql`): `hypotheses`, `documents`,
`experiment_packages`, `predictions`, `scores`, `verdicts`, `interventions` and `violations`, all
insert-only and all in **one hash chain**. The database assigns each row its `seq`, its time (its
own clock), its canonical JSON text and `hash = SHA-256(prev_hash || canonical)`. A prediction for
bar *t* is rejected if it arrives, or commits, at or after bar *t* opens, and the rejection is kept
as a violation. **Ledger rows are permanent: try commands with `--test` first.**

```bash
uv run cryptonlp ledger intervene --stage 2 --type approval --checkpoint --reason "..."  # log it
uv run cryptonlp ledger register-document path/to/plan.md --key analysis-plan \
    --kind analysis_plan --title "Analysis plan" --version 1.0                         # before use
uv run cryptonlp ledger check            # recompute the chain (also daily at 05:45 UTC)
uv run cryptonlp ledger anchor           # head hash for the advisor (also Mondays 06:00 UTC)
uv run cryptonlp ledger export           # JSON lines, for the replication package
uv run cryptonlp ledger verify-export FILE --anchor ANCHOR.json   # SHA-256 only, no database
```

| Job | When (UTC) | Writes |
| --- | --- | --- |
| `ledger-check` | daily at 05:45 | `ops.ledger_checks`; a newly found break also becomes a `chain_break` violation |
| `ledger-anchor` | Mondays at 06:00 | `data/ledger/anchors/ledger-anchor-<date>-seq<N>.json`: send it to the advisor |

The check catches an edited column, an edited hash, a deleted or reordered row, and (with the
anchors) a tail rewritten with every hash recomputed, or cut back. It can't stop the operator, who
owns the database: the guarantee is detection (paper, Section 4.9). Python API:
`cryptonlp.ledger.append(conn, record)` and `record_prediction(conn, forecast)`, with the record
types in `cryptonlp.schemas.ledger`.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                    # DB tests skip locally if postgres-test is down; CI requires it
uv run cryptonlp-run-id          # the run ID every job prints and stores with its results
scripts/gpu_check.sh             # GPU inside Docker, then vLLM loading a small model
```

## Rules the code enforces

- **Two databases.** Tests, CI and the simulated clock only use `postgres-test` (port 5433, data on
  tmpfs, database name ending in `_test`). `connect_main()` refuses to run under pytest or CI.
- **Run IDs.** `20261103T142233Z-<12-char commit>`, with `-dirty` when code under `src/`, `db/`,
  `config/` or the project files differs from the commit. Paper edits don't make a run dirty.
- **Shared schemas** in `src/cryptonlp/schemas/`: posts, candles, labels, hourly features, forecasts
  and findings. All times are UTC (naive datetimes are rejected), records are immutable, and unknown
  fields are an error. The schemas also check the leakage rule (no post received at or after the bar
  close) and that a forecast is issued before its bar opens. The ledger enforces the deadline again
  with the database clock.
- **Simulated clock.** `ledger.set_simulated_clock(conn, t)` replays time for tests and the step 3b
  replay. The database refuses it in any database whose name doesn't end in `_test`.
- **Leakage.** A bar's features use only posts received during it, and are written only once the
  bar is final (`features.hourly` repeats the check). A forecast sees only bars that had closed
  when it was issued: `ForecastRequest` raises `LeakageError` otherwise.
