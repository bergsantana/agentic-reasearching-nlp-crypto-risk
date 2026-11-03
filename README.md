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

## Pre-registration ledger (step 2)

Schema `ledger` in the main database (`db/init/003_ledger.sql`): `hypotheses`, `documents`,
`experiment_packages`, `predictions`, `scores`, `verdicts`, `interventions` and `violations`, all
insert-only and all in **one hash chain**. The database assigns each row its `seq`, its time (its
own clock), its canonical JSON text and `hash = SHA-256(prev_hash || canonical)`. A prediction for
bar *t* is rejected if it arrives, or commits, at or after bar *t* opens, and the rejection is kept
as a violation. **Ledger rows are permanent: try commands with `--test` first.**

```bash
uv run cryptonlp ledger intervene --stage 2 --type approval --checkpoint --reason "..."  # log it
uv run cryptonlp ledger register-document path/to/manual.md --key labeling-manual \
    --kind labeling_manual --title "Manual de rotulagem" --version 1.0                 # before use
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
