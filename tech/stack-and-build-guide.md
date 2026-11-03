# Tech stack and build guide

Nov 2, 2026 · @Berg

This file picks a tool for each component in [the system design](../Crypto%20NLP%20Research%20Agents%20System%20Design.md), explains each choice, and gives the order to build the components in. The tools were chosen for two reasons: they fit the method in `paper/CLAUDE.md`, and they are widely used in MLOps and agent-developer jobs, so the TCC also works as a portfolio.

The method rules still apply. Where this guide and the paper disagree, the paper wins, and any departure from the design doc is recorded in `paper/claims.md`.

## At a glance

| Layer | Tools |
| --- | --- |
| Language and packaging | Python 3, uv, Docker Compose |
| Data | Parquet raw store, PostgreSQL + pgvector, Prefect |
| Ledger | PostgreSQL (append-only, hash-chained, deadline enforced by the database clock) |
| Labeling | Argilla or Label Studio |
| Models | vLLM serving SemIf; Hugging Face Transformers baselines; MLflow tracking and registry |
| Statistics | `arch`, statsmodels, scikit-learn, LightGBM |
| Agents | LangGraph, MCP tool servers, Pydantic, Langfuse |
| Quality and operations | pytest, GitHub Actions, frozen replay data, Grafana (from step 1), Evidently (later) |
| Researcher UI | Streamlit control panel, plus the built-in UIs of Prefect, MLflow, Langfuse and the labeling tool |

## The stack, explained

### Foundation

| Tool | What it does here | Why this one | Job signal |
| --- | --- | --- | --- |
| **uv** | Python versions, dependencies, lock file | Fast, with one lock file (`uv.lock`) that pins every version. The lock file fills the article's "[A FAZER: versões das bibliotecas]". | Modern Python tooling |
| **Docker Compose** | Runs Postgres, MLflow, Langfuse, Prefect and vLLM together | A single `docker compose up` brings back the whole system, which the reproducibility section needs | Containers are in almost every MLOps ad |
| **GitHub Actions** | Lint, tests, ledger tamper test, eval run on the holdout | Continuous integration (CI) for code and for models | CI/CD |
| **pytest** | Unit tests and leakage tests | A leakage test (a feature uses a post from after the bar closed) is worth more than any unit test | Testing |

### Data

| Tool | What it does here | Why this one | Job signal |
| --- | --- | --- | --- |
| **Parquet files** (raw store) | Untouched posts and candles, partitioned by source and day, each file with a SHA-256 hash | Append-only and cheap, and every derived table can be rebuilt from it. The hashes go into the ledger's experiment packages. | Data lake basics |
| **PostgreSQL** | Labeled store, ledger, agent tasks and run logs | The paper already commits to it for the ledger. One database is simpler to run. | SQL is everywhere |
| **pgvector** | Embeddings for the Analyst's semantic search | Keeps vectors in the same database as the labels, so there is no extra vector-database service | Retrieval and RAG |
| **Prefect** | Schedules collectors, normalization, classification, deletion sync, daily chain check, hourly forecast job | Python-native, light to self-host, with retries and run history. Dagster is a fine alternative. Airflow is heavier but more common in older ads. | Pipeline orchestration |
| **DVC** (optional) | Versions the labeled sets and Parquet snapshots | Only if the ledger's hashes turn out not to be enough | Data versioning |

### Labeling and models

| Tool | What it does here | Why this one | Job signal |
| --- | --- | --- | --- |
| **Argilla** or **Label Studio** | Labeling the 2,000 posts, the re-labeling of 200, and the Curator's review queue | Both self-host, export to Hugging Face datasets and support multi-label tasks. Argilla is closer to the Hugging Face ecosystem. | Data-centric ML |
| **vLLM** | Serves SemIf's base model (Qwen3.5-4B) and, optionally, a local agent model | Serving models is a core MLOps skill, and vLLM is the common choice. llama.cpp is the fallback if vLLM doesn't support the GPU (see "Hardware notes"). | Model serving |
| **Hugging Face Transformers** | CryptoBERT, FinBERT and the fallback fine-tuning; VADER via its own package | The baselines in H4 are published on Hugging Face | Standard |
| **MLflow** | Logs every classifier and forecasting run; registry with model versions and rollback | The design stores a model version beside every label and retrains only when the holdout beats the live model, which is exactly what a registry does | The default MLOps tool |

### Statistics and forecasting

| Tool | What it does here |
| --- | --- |
| **`arch`** | GARCH, EGARCH, GJR-GARCH and GARCH-X (arms 1 and 3) |
| **statsmodels** | HAR regression, Newey-West errors, Diebold-Mariano test |
| **scikit-learn**, **LightGBM** | Regularized regression and gradient boosting (arms 2 and 3), walk-forward splits |
| Own code | QLIKE, Kupiec and Christoffersen backtests, Benjamini-Hochberg correction, power simulation by block bootstrap. These are short, and writing them yourself keeps them easy to audit. |

### Agents

| Tool | What it does here | Why this one | Job signal |
| --- | --- | --- | --- |
| **LangGraph** | The Orchestrator and each agent as a graph with explicit states, budgets and halt conditions | Explicit graphs make loops, budgets and human checkpoints visible and testable, unlike a free-running chat loop | One of the most-asked agent frameworks |
| **MCP** (Model Context Protocol) | One tool server per agent role: read-only SQL, ledger writer, labeling queue, trend APIs | The tools a server offers *are* the agent's permissions, which implements the least-privilege table in the design doc | Becoming the standard way to give agents tools |
| **Pydantic** | Typed schemas for findings, hypotheses, verdicts and briefs; agents' structured outputs are validated before anything is written | A malformed output fails loudly instead of being stored | Structured outputs |
| **Langfuse** (self-hosted) | Traces every LLM and tool call, with token cost and latency; datasets and evals for agent behavior | Its traces produce most of the QP1 process measures: interventions, cost and time per stage | Agent tracing and evaluation |
| **Agent LLM** | One hosted API model for the agents, plus the local model as a comparison arm | Agent jobs expect experience with hosted APIs and tool calling. The local arm gives a cost and quality comparison. Pin the exact model ID and record it in the ledger. | |

### Monitoring and control

| Tool | What it does here | When | Job signal |
| --- | --- | --- | --- |
| **Grafana** | One dashboard for system health, reading Postgres directly through a read-only role, with alerts by email or Telegram | From step 1, with panels added in each step | Observability |
| **Streamlit** | Researcher control panel: stage approvals, intervention logging, audit of hypotheses, briefs | Step 7 | Internal tools |
| **Evidently** | Drift reports: share of the "other" topic, label distribution by week | During the confirmatory period | ML monitoring |
| **Prometheus** (optional) | Process-level metrics, if Postgres queries stop being enough | Only if needed | Observability |

Details are in [UIs](#uis) below.

## How the design maps to the stack

| Design component | Built with |
| --- | --- |
| Collectors (Bluesky, Mastodon, Binance) | Python async clients (`websockets`, `httpx`) run by Prefect, writing Parquet |
| Normalize, dedupe, pseudonymize | Prefect flow; language detection; MinHash for near duplicates |
| Classify | Prefect batch job calling SemIf on vLLM; labels, confidence and model version into Postgres |
| Index | pgvector |
| Ledger | Postgres tables and triggers; Prefect daily chain check |
| Orchestrator, Trend scout, Analyst, Verifier, Curator, Reporter, validator | LangGraph agents, each with its own MCP server; traced in Langfuse |
| Forecasting arms 1–3 | `arch`, statsmodels, LightGBM; runs logged in MLflow |
| Budgets and halts | LangGraph state, with limits read from config the agents can't write to |
| Researcher checkpoints and interventions (Quadro 5) | Streamlit control panel writing to the ledger |
| System health | Grafana dashboard on Postgres |

## Suggested repository layout

```
tech/                  this guide and other technical notes
src/
  collectors/          bluesky.py, mastodon.py, binance.py, trends.py
  pipeline/            normalize.py, dedupe.py, pseudonymize.py, classify.py, index.py
  ledger/              schema.sql, chain.py, check.py
  labeling/            manual.md, export.py
  features/            bars.py, nlp_features.py   (leakage rules live here)
  forecast/            arm1_stats.py, arm2_nlp.py, arm3_combined.py, walkforward.py, tests.py
  agents/              orchestrator.py, analyst.py, verifier.py, curator.py, reporter.py, validator.py
  mcp_servers/         one server per agent role
  flows/               Prefect flows and schedules
  stubs/               StubClassifier, baseline forecaster, fake LLM
  schemas/             shared Pydantic models (posts, labels, features, forecasts, findings)
ui/
  control_panel/       Streamlit app
dashboards/            Grafana provisioning (data source, dashboard JSON, alert rules)
tests/
  fixtures/replay/     one frozen day of Jetstream, Mastodon and Binance data
config/                budgets, thresholds, pilot_days (agents get read-only access)
docker-compose.yml
pyproject.toml
uv.lock
```

## Build guide

The schedule, effort estimates, run modes and checklist are in [build-plan.md](build-plan.md). This section explains what each step involves.

**Why this order:** the 3-month confirmatory period is the critical path. Everything it depends on is built first, as fast as possible. Anything that can be finished during it is pushed into it.

This order differs from the design doc's roadmap in one way. In the article's research cycle (Quadro 5), the agents *run* the study during the confirmatory period, so the agents for stages 4–6 are built before it starts, not after the signal study. Record this in `paper/claims.md` as a departure from the design doc.

### Step 0. Repo skeleton (a few days)

- Set up uv, `pyproject.toml`, linting and pytest.
- Write `docker-compose.yml` with Postgres + pgvector first. Add the other services as each step needs them.
- Create a **separate test database**. CI and the simulated clock only ever touch this one, never the real ledger.
- Define the run ID: a timestamp plus the git commit, printed by every job and stored with every result.
- Write the shared Pydantic schemas in `src/schemas/` (posts, labels, features, forecasts, findings). Every component and every stand-in exchanges these, so a stand-in can be swapped for the real component without other changes.
- Add a GitHub Actions workflow that runs lint and tests.

**Done when** CI passes on an empty pipeline.

### Step 1. Collectors and raw store (start as early as possible)

- **Bluesky:** a Jetstream WebSocket client filtered to `app.bsky.feed.post`. Save the cursor so it can reconnect; Bluesky-hosted instances only allow 36 hours of lookback. Keep the time Jetstream saw the event as the reception time.
- **Mastodon:** poll the public and hashtag timelines on a few chosen instances, using `since_id`.
- **Trends:** store a snapshot of Bluesky trending topics and Mastodon `/api/v1/trends/tags` every hour. Neither platform keeps a trend history you can query later.
- **Binance:** backfill about 2 years of BTCUSDT and ETHUSDT candles at 1 minute and 1 hour from the Vision archive. Keep the checksums. Fill the recent tail from the market-data API, with backoff on rate limits.
- Write Parquet partitioned by source and day, hash each file, and never edit a file.
- Run deletion sync as a scheduled job.
- **Record a replay day:** save one full day of Jetstream, Mastodon and Binance data to `tests/fixtures/replay/`, and add a `replay` mode to the collectors that feeds it back in. CI uses it from step 3b onward.
- **Start the Grafana dashboard:** collector lag, posts per hour by source and job failures, plus an alert when Bluesky lag passes 1 hour.

**Done when** the collectors run continuously with deletions synced (stage 2 gate in Quadro 5), and the dashboard shows it.

**Why first:** post data only builds up with time. A day without collectors is mostly lost.

### Step 2. Pre-registration ledger

- Create the tables `hypotheses`, `experiment_packages`, `predictions`, `scores` and `verdicts` (the five tables in the paper), plus `interventions` (date, stage, type, reason).
- Add triggers that reject UPDATE and DELETE.
- Each row stores `prev_hash`, and `hash = SHA-256(prev_hash || canonical JSON of the row)`.
- For the deadline, a prediction for bar *t* is rejected if the database clock (`now()`) is past bar *t*'s open. Never trust a timestamp sent by the client.
- Add a `phase` column (`pilot`, `development`, `confirmatory`) so pilot and development rows can never mix into confirmatory results.
- Write a Prefect job that recomputes the whole chain daily, and a weekly export of the last hash for the advisor.

**Done when** a CI test that edits an old row breaks the chain and the daily check catches it.

**Why here:** the labeling manual and H4 must be registered before labeling and the bake-off.

### Step 3. Normalize, dedupe, pseudonymize

- Detect the language and keep English only.
- Drop exact duplicates by hash and near duplicates by MinHash.
- Extract cashtags, tickers and contract addresses.
- Replace author IDs with hashes, using a keyed hash whose secret stays outside the repo. Keep the post URI for deletion sync.
- Apply the spam pattern rules.

**Done when** the dedupe rates and spam filter have been checked by hand on a sample.

### Step 3b. Thin end-to-end loop

Connect the thinnest possible version of the whole hourly cycle now, with stand-ins for the parts that don't exist yet. Later steps replace each stand-in with the real component.

- `StubClassifier`: pattern rules or VADER, behind the same interface SemIf will use, writing `model_version = "stub"`.
- Hourly features from the stub labels, with the leakage rule applied.
- Baseline forecaster ("yesterday's volatility"), behind the same interface as arms 1–3.
- One hourly Prefect flow: classify new posts, compute features, write forecasts to the ledger before the bar opens, and score them when it closes.
- A simulated clock, so CI can run the frozen replay day through the loop in minutes. It runs only against the test database.
- Grafana panels: forecasts on time vs. late, and scored vs. pending.

**Done when** CI replays the frozen day with the expected forecasts and scores, and smoke run 1 (24 hours live, `phase = development`) misses no bars.

**Why here:** the first full run of the hourly loop now happens months before the pilot, and every later step is tested inside a working system.

### Step 4. Labeling (long and manual; run steps 3b, 5 and 7 in parallel)

- Write the labeling manual and register it in the ledger *before* labeling starts.
- Set up Argilla or Label Studio with the five tasks.
- Sample 2,000 posts, stratified by platform and week and enriched with rule-flagged scam candidates.
- Label them. After at least 2 weeks, re-label 200 without looking at the earlier labels, and compute Cohen's kappa.
- Split by time: the oldest 500 for calibration, the newest 1,500 for the test.

**Done when** the labeled set and holdout are accepted (stage 3).

### Step 5. Market features and statistics arm (in parallel with step 4)

- Hourly bars in UTC, and realized volatility from the 1-minute returns.
- Leakage rule in code: a feature for bar *t* may only use posts received before bar *t* closes. Write a test that fails if this is broken.
- Walk-forward harness with an expanding window and no shuffling.
- Arm 1 (GARCH family, HAR, historical VaR) and the "yesterday's volatility" baseline.
- Metrics and tests: MSE, QLIKE, Kupiec, Christoffersen, Diebold-Mariano with Newey-West errors.
- Log every run in MLflow.
- Swap arm 1 into the thin loop in place of the baseline forecaster.

**Done when** arm 1 runs walk-forward on development data and beats or matches the baseline, and the CI replay passes with arm 1.

### Step 6. Classifier serving and bake-off

- Serve the SemIf base model on vLLM, with the revision pinned.
- Register H4 in the ledger, along with the model revision, rubric and prompt hash.
- Run SemIf, Wortega's OpenJev, CryptoBERT, FinBERT and VADER on the holdout, per class and per platform. Calibrate on the 500-post calibration slice.
- Pick fallbacks only where SemIf clearly loses, and register the chosen versions in MLflow.
- Classify the whole store, writing labels with confidence and model version. Then build the pgvector index.
- Swap SemIf into the thin loop in place of `StubClassifier`, and add a Grafana panel for classification latency per hour of posts.

**Done when** H4 is scored, the fallbacks are chosen and logged, and smoke run 2 (24 hours live with real labels) passes. Smoke run 2 also gives a first measure of whether one hour of posts is classified well before the next bar opens.

### Step 7. Agent foundation and control panel

- One MCP server per role, exposing only the tools in the design doc's Agents table. For example, the Analyst's server has read-only SQL and vector search, and no write tools.
- LangGraph Orchestrator: turns tasks into runs, enforces per-run and per-day budgets on tokens, requests and spend, and halts on repeated errors. The budgets live in `config/`, which the agents can't write to.
- A fake LLM with scripted responses, so CI can run agent flows with no API cost.
- Langfuse on every LLM and tool call.
- Agent test cases as Langfuse datasets: a post containing injected instructions, a runaway loop, a finding that rests on one account.
- Treat post text as data: it goes into prompts only inside delimited data fields, never as instructions.
- The researcher control panel (see [UIs](#uis)). Every human action outside a checkpoint is logged through it to the `interventions` table.
- A Grafana panel for agent spend against budget.

**Done when** a test with a runaway loop gets halted by budget, every tool call shows up in Langfuse, the agent test cases pass on the fake LLM, and an intervention logged in the control panel appears in the ledger.

### Step 8. Agents for the confirmatory period

- **Orchestrator forecast loop (stage 5):** every hour, arms 1–3 produce forecasts, which are written to the ledger before the bar opens and scored when it closes.
- **Verifier (stage 6):** bot and shill filters, plus single-source and burst checks. It also produces the filtered and unfiltered feature sets for the bot-filter ablation.
- **Curator:** a queue of low-confidence items, feeding the labeling tool.
- **Trend scout and Analyst (stage 4):** watch queries, clustering and anomaly flags. Exploratory hypotheses (at most 10 a month) are written to the ledger before they are tested, and shown in the control panel for audit.
- Run the agent test cases again, this time on the real agent LLM.

**Done when** smoke run 3 passes: forecasts flow through the ledger with no manual help for 24 hours.

### Step 9. Pilot, power simulation, registration

- **Pilot:** run the whole system for `pilot_days` (set in config; 5 to 7 days, and 7 covers one full weekly cycle) with `phase = pilot`. Use it to find bugs and check the hourly post volume. Pilot data never counts toward results.
- **Power simulation:** run it on development data with a block bootstrap. If 3 months gives low power, either lengthen the period or register the minimum detectable effect.
- **Register** H1–H3, the analysis plan, the power estimate and the results-table templates.
- **Start the 3-month confirmatory period.**

### During the confirmatory period

- **Validator agent (stage 7):** replays each registered experiment package and writes "reproduced" or "divergent". It must be ready before the first block closes.
- **Reporter (stage 8):** writes briefs in which every claim cites post IDs, with an automatic citation check.
- **Monitoring:** Evidently drift panels and a queue-size panel in Grafana, plus a scheduled CI eval on the holdout.
- **Optional, for the CV:** deploy the classifier as a FastAPI service on a cloud VM or a local k3s cluster.
- **Last:** Reddit, for analysis only and outside the confirmatory study, after checking its current terms.

## Testing strategy

The goal is to test the system as a whole while it's still incomplete, so components that don't fit together show up early instead of in the pilot.

| Technique | How it works | Introduced in |
| --- | --- | --- |
| **Thin end-to-end loop** | The full hourly cycle runs from step 3b, with stand-ins for unbuilt parts. Each later step swaps in a real component. | Step 3b |
| **Stand-ins with the real interfaces** | `StubClassifier`, the baseline forecaster and a fake LLM implement the same interfaces and Pydantic schemas as the real components, so swapping one in changes nothing else | Steps 3b and 7 |
| **Frozen replay data** | One recorded day of Jetstream, Mastodon and Binance data. The collectors' `replay` mode feeds it through the pipeline, so CI checks known input against known output. | Step 1 |
| **Simulated clock** | Runs the hourly loop over the replay day in minutes. It only runs against the test database. The real ledger always uses the database clock. | Step 3b |
| **Unit and rule tests** | The ledger tamper test, the leakage test, the deadline test (a late forecast is rejected), dedupe and pseudonymization | Steps 2, 3, 5 |
| **Agent test cases** | Langfuse datasets with scripted scenarios: injected instructions, a runaway loop, a finding resting on one account. Run on the fake LLM in CI, and on the real LLM before smoke run 3. | Steps 7–8 |
| **Smoke runs** | 24 hours live with `phase = development`: after step 3b, after step 6 and after step 8 | Steps 3b, 6, 8 |
| **Pilot** | The whole system for `pilot_days` with `phase = pilot` | Step 9 |

Two rules keep tests from contaminating the study:

- CI and the simulated clock use only the test database.
- Smoke runs and the pilot are marked with their `phase`, so their rows can never enter confirmatory results.

## UIs

The system has two UIs of its own, plus the tools' built-in ones.

### Grafana dashboard (from step 1)

Read-only. It answers "is the system healthy right now?" Grafana reads Postgres directly through a read-only role, so Prometheus isn't needed at first. Panels are added as steps finish:

| Panel | From step | Alert |
| --- | --- | --- |
| Collector lag per source | 1 | **Bluesky lag over 1 hour** (the replay limit is 36 hours) |
| Posts per hour, by source | 1 | An hour with no posts from a source that is normally active |
| Job failures (Prefect runs) | 1 | Any failed run of a scheduled job |
| Last chain-check result | 2 | Chain broken |
| Forecasts on time vs. late; scored vs. pending | 3b | Any missed bar during the pilot or the confirmatory period |
| Classification latency per hour of posts | 6 | Latency approaching the next bar's deadline |
| Agent spend against budget | 7 | 80% of the daily budget |
| Review-queue size; share of the "other" topic | 8 | Queue growing faster than you can review it |

Keep the dashboard JSON and alert rules in `dashboards/`, so the dashboard is rebuilt with the rest of the system. Alerts only fire while the machine running Grafana is on. That's one reason to put it on the cloud VM if you use one.

### Researcher control panel (step 7)

A small Streamlit app, local only (bound to `localhost`), for your own part of the research cycle. It writes through its own database role, which can only insert approvals and interventions.

| Page | What it does | Why it matters |
| --- | --- | --- |
| Stage approvals | Approve or reject the gates in Quadro 5 (stages 2, 3, 6, 8), with a comment | Each approval is a ledger record with a date |
| Log intervention | A form for date, stage, type (correction, interruption or approval) and reason | QP1 counts human interventions. If logging one takes a SQL insert, some will go unlogged and the count will come out too low. |
| Hypotheses awaiting audit | Agent-proposed exploratory hypotheses, with their registered test and ledger hash | Your audit role in stage 4. It also settles the open question in Section 4.2 of the article, whichever way you decide it. |
| Briefs | The Reporter's briefs, with links to the cited posts | Your review in stage 8 |
| Status and links | Current phase, the last chain hash, and links to Grafana, Prefect, MLflow, Langfuse and the labeling tool | One place to start |

### Built-in UIs

| UI | Use it for |
| --- | --- |
| Prefect | Pipeline runs, retries, schedules |
| MLflow | Classifier and forecasting runs, model registry |
| Langfuse | Agent traces, cost, latency, test-case results |
| Argilla or Label Studio | Labeling and the Curator's review queue |

## Data windows

| Data | How far back | How long forward |
| --- | --- | --- |
| Binance candles | About 2 years, from the Vision archive | Continuously until the confirmatory period ends |
| Bluesky and Mastodon posts | No backfill for the study. Backfilled posts are allowed only as development data, with `origin = backfill`. | Continuously from now until the confirmatory period ends. Deletion sync runs as long as raw posts are kept. |
| Trending topics | Unavailable (Mastodon shows about the past week) | Hourly snapshots from day one |

Backfilled posts are kept out of the study for three reasons:

- Deleted posts are missing, and scams and bots are the most likely to have been removed.
- Bluesky's `createdAt` is set by the client and can be backdated.
- The confirmatory period must be prospective.

## Hardware notes

- **GPU:** an RTX 5070 Ti with 16 GB. SemIf's 4B base model in BF16 takes about 8 GB of weights. Running a second local LLM at the same time is tight. Run them one after the other, or quantize the agent model.
- **Blackwell support:** the 5070 Ti is a Blackwell card. Check that the pinned PyTorch, CUDA and vLLM versions support it. If vLLM doesn't, use llama.cpp.
- **WSL2 and Docker:** GPU access inside Docker on WSL2 needs the NVIDIA Container Toolkit. Test `nvidia-smi` inside a container before building on it.
- **Langfuse:** self-hosting the current version needs more than Postgres (check its own Compose file). Budget for the extra services or start with its cloud free tier.

## Still to decide

- Argilla or Label Studio.
- Which hosted model drives the agents, and the monthly spend cap.
- Which Mastodon instances to poll.
- Whether exploratory hypotheses enter the ledger without the researcher's approval (open in the article, Section 4.2).
- The repository URL (for the article's "Disponibilidade de dados e código" section).
