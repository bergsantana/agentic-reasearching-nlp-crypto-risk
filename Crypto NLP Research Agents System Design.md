# Crypto NLP Research Agents: System Design

Oct 21, 2026 · @Berg

A small crew of autonomous agents ingests public crypto discussion from Bluesky, Mastodon and Reddit, classifies it into sentiment, topic, scam and event signals, and hands a research agent evidence-backed briefs.

## At a glance

&#91;embedded content: system architecture · 7 agent roles\]

Posts flow from sources to the labeled store, then through the Analyst, Verifier and Reporter. Low-confidence labels go to the Curator, who retrains the classifiers.

## Data sources

Bluesky and Mastodon give a free, open stream; Reddit is free for non-commercial research only, but its terms bar training models on it without permission.

| Source | Access | Cost | Role in the system | Limit to respect |
| --- | --- | --- | --- | --- |
| [Bluesky Jetstream](https://anderegg.ca/2024/11/25/playing-with-the-bluesky-firehose) | Unauthenticated JSON WebSocket of public posts | Free | Primary live stream | Bluesky is building user consent signals for AI training; honor them |
| [Bluesky trending API](https://apify.com/jrpbuilds/bluesky-trending-topics.md) | Public endpoint, no auth, 25 topics per request | Free | Trend seeds for agents | Global, English-focused topics |
| [Mastodon / Fediverse](https://apify.com/puskin/mastodon-scraper.md) | Public timelines and hashtags, no auth on most instances | Free | Second stream, long-form discussion | Per-instance rules; GDPR care |
| [Reddit Data API](https://apiserpent.com/blog/best-reddit-api-providers-2026) | OAuth, 100 queries per minute | Free, non-commercial | Deep threads, crypto subreddits | Terms bar ML training without permission |
| [Bluesky 2M posts](https://huggingface.co/Edens-Gate/Bluesky-2M-backup) | Hugging Face dataset | Free | Bootstrap training and tests | Use subject to Bluesky terms |

Start with Bluesky and Mastodon for training data; use Reddit for analysis only.

## Market data (Binance)

Binance publishes free, keyless price data for Bitcoin and Ethereum, so the market side of the study costs nothing.

| Source | What it gives | Access | Use it for |
| --- | --- | --- | --- |
| [Binance Vision archive](https://dev.to/pi19404/a-technical-guide-to-downloading-and-managing-binance-historical-crypto-market-data-d82) | Monthly and daily zipped files of klines (OHLCV candles), trades and aggregated trades per symbol | Static files, no key | Historical backfill |
| [Public market-data API](https://pkg.go.dev/github.com/algo-one/binance-data-downloader/internal/vision) | Read-only market data from the data-api.binance.vision mirror, no key or signature | HTTPS with the trading API's rate limits | Recent candles and gap-filling |

- **Symbols and bars.** Pull BTCUSDT and ETHUSDT klines at 1 minute for realized volatility and at 1 hour to join with post features.
- **Rate limits.** Requests consume IP-based weight, and Binance's own docs point to the Vision archive for history. Backfill from the archive and use the API only for the recent tail, with backoff on rate-limit errors.
- **Terms.** The Vision datasets are provided for computational, educational and research purposes, and commercial use needs a separate written licence, which fits a research project. Binance may restrict API access at its discretion, and access from some regions, such as the US, has been blocked, so check yours.
- **Data quality.** Binance occasionally restates historical candles, so re-pull recent periods, keep the raw files with their checksums, and never edit them.
- **Joining with posts.** Store every timestamp in UTC and bucket posts into the same hourly bars. A feature for a bar may only use posts published before that bar closes.

A Market collector, one more Collector instance, writes candles to a separate market table in the raw store.

## Agents

Seven roles share one store and talk through tasks and records, never directly; each has a narrow tool set so a failure stays contained.

| Role | Job | Reads | Writes | Tools |
| --- | --- | --- | --- | --- |
| Orchestrator | Turns research questions into tasks, sets budgets and schedules, stops runaway loops | Task queue, run logs | Tasks, budgets | Task API only |
| Collectors (one per source) | Stream or poll posts, dedupe, keep source metadata | Source APIs | Raw posts | HTTP client, rate limiter |
| Trend scout | Finds rising crypto topics and proposes watch queries | Trend endpoints, labeled store | Watch queries | Search, trend APIs |
| Analyst | Clusters labeled posts, tracks sentiment and topic shifts, flags anomalies | Labeled store | Findings with evidence IDs | Read-only SQL, vector search |
| Verifier | Checks each finding for bots, shill bursts, single-source bias and duplicates | Findings, raw posts | Verdicts, confidence | Read-only SQL, account-age and dedupe checks |
| Curator | Samples uncertain items for human labels, builds training sets, requests retraining | Low-confidence items | Labeled sets, retrain requests | Labeling queue, dataset writer |
| Reporter | Writes briefs where every claim cites post IDs | Verified findings | Briefs | Read-only store, document writer |

Only the Orchestrator can start or stop other agents. Only the Curator can write training data.

## Pipeline and storage

Raw posts land untouched in an append-only store, and everything downstream is derived, so any label or index can be rebuilt.

1. **Ingest.** Collectors write each post with text, source, post URI, timestamp and a hashed author ID to the raw store, partitioned by source and day.
2. **Normalize.** Detect language, drop exact and near duplicates, extract cashtags, tickers and contract addresses, and filter obvious spam.
3. **Pseudonymize.** Keep hashed author IDs only; keep the post URI so deletions at the source can be honored.
4. **Classify.** Models write labels with a confidence score and the model version, never overwriting earlier labels.
5. **Index.** Labeled posts go to a relational store for queries and a vector index for semantic search.
6. **Sync deletions.** A scheduled job removes posts, labels and embeddings for anything deleted at the source.

Storing the model version beside every label makes relabeling reproducible when a classifier is retrained.

## NLP tasks

Five tasks cover the signals the Analyst needs. SemIf scores all of them from one open model; a task moves to a fine-tuned model only if it falls short on the holdout.

| Task | Labels | Starter data | Notes |
| --- | --- | --- | --- |
| Sentiment | negative, neutral, positive | SemIf zero-shot; crypto posts labeled by the Curator only if fine-tuning is needed | Crypto slang flips polarity ("rekt", "moon"), so fine-tune on in-domain posts |
| Topic | price talk, regulation, security incident, protocol or tech, launch or airdrop, macro, other | Curator-labeled sample from all three free sources | Multi-label; keep "other" to catch drift |
| Scam and shill | organic, promotion, likely scam (phishing, giveaway, impersonation) | SemIf zero-shot plus pattern rules, then human review | Highest-cost errors; tune for recall and send borderline cases to review |
| Entity linking | token, project, exchange | Cashtag and contract-address rules plus an alias lexicon, scoped to Bitcoin and Ethereum | Lexicon updated by the Trend scout |
| Event type | hack, outage, listing, regulatory action, none | SemIf zero-shot, human-checked | Feeds the Analyst's anomaly flags |

Every prediction stores a confidence score; items below a set threshold go to the Curator's review queue instead of the Analyst.

## Classifier strategy

Start with SemIf, the open alternative to Jev: it reads typed option probabilities straight from a small open model in one forward pass, with no per-task training, so all five tasks can run at once with labels defined at run time. Keep the other routes as fallbacks measured on the same holdout.

| Option | Examples | Good for | Watch out |
| --- | --- | --- | --- |
| SemIf zero-shot (chosen) | [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) on a small open base such as Qwen3.5-4B | All five tasks at once, labels defined at run time, no training, probabilities in one pass | Benchmarked on general decisions, not crypto posts; not hardened against prompt injection; validate and calibrate on your holdout |
| Open pretrained, as is | [CryptoBERT](https://huggingface.co/ElKulako/cryptobert), [FinBERT](https://huggingface.co/ProsusAI/finbert) | A sentiment baseline for comparison | Trained on other platforms' text; sentiment only |
| Open model fine-tuned on your labels | CryptoBERT or [FinTwitBERT](https://huggingface.co/StephanAkkerman/FinTwitBERT) (MIT) as the base | A task where SemIf underperforms on your holdout | Needs Curator labels; check the base model's licence |
| Self-trained | A small encoder from a general base | A task no open route handles well | Highest labeling cost and the full evaluation burden |

- **SemIf first, measured.** Run SemIf on the human-labeled holdout per task and per platform, and record per-class F1, calibration and latency as the baseline to beat.
- **Fall back only where it loses.** Keep SemIf for any task where it is good enough; bring in a fine-tuned or self-trained model only for a task where it clearly underperforms on your holdout.
- **Check platform shift.** Report scores separately for Bluesky, Mastodon and Reddit, since open models were trained elsewhere.
- **Not a safety guard.** SemIf is not hardened against prompt injection, and crypto posts are full of it, so the Verifier's bot and shill checks and deterministic rules stay in front of it, never replaced by it. Log the base model's licence and revision in the registry.
- **How SemIf is used here.** One post is the state, and each task is a set of typed options with a rubric; its shared-state mode scores the post for sentiment, topic, scam and event in one prefill. The same primitive also serves small agent decisions, such as whether evidence supports a claim. Calibrate its probabilities on a labeled slice before trusting the confidence scores, as its authors advise.
- **Hardware.** SemIf's 4B reference model should fit the 16 GB RTX 5070 Ti (my estimate: about 8 GB of weights in BF16). Its 27B quantized variant likely won't (about 17 GB of weights by my estimate), and there are CPU and Apple Silicon backends.

## Guardrails

Agents only read: they never post, trade, follow accounts or act on instructions found inside posts.

- **Posts are untrusted data.** Crypto social media is full of shill bursts and injected instructions. Agents never pass post text into a prompt as instructions, and the Verifier checks any finding that rests on one account or one burst.
- **Source terms.** Reddit's Data API Terms exclude ML training without permission. Treat Reddit as blocked for training until you have checked the current terms or a licence, and use it for analysis only. Bluesky data is open, but honor its consent signals as they roll out.
- **Privacy.** Store hashed author IDs, sync deletions, and never build profiles or location lookups on private individuals.
- **Least privilege.** Each agent gets only the tools in the Agents table. Collectors cannot write labels, and only the Curator writes training data.
- **Budgets.** The Orchestrator enforces per-run and per-day caps on API spend, tokens and requests, and halts on repeated errors instead of retrying around limits.
- **No advice.** Briefs report signals and uncertainty. They never recommend trades or price targets.

## Evaluation

Classifiers are judged on human-labeled holdouts split by time and by source; agents are judged by replaying past weeks and checking every cited post.

| What | Test | Pass signal |
| --- | --- | --- |
| Each classifier | Human-labeled holdout; per-class precision, recall and F1; calibration of confidence | No class collapses; scam recall meets the target you set |
| Drift | Train on older weeks, test on newest; train on Bluesky, test on Reddit | Small drop across time and source |
| Findings | Replay a past week and compare against a hand-made list of real events | Known events surface; few false alarms |
| Briefs | Check that each claim cites a post ID and that the post supports it | Zero uncited claims; unsupported claims caught by the Verifier |
| Operations | Cost per 1,000 posts, latency, review-queue size, share of "other" topic | Stays inside budget; drift alerts fire on shifts |

The Curator retrains only when a holdout beats the live model, and every release keeps its model version for rollback.

## Signal study: NLP, statistics, or both

This study tests whether NLP signals add forecasting value for Bitcoin and Ethereum risk beyond traditional statistics, using three arms scored on the same out-of-sample data.

Prior evidence is mixed, so test it rather than assume it. One [study of over 16 million Bitcoin tweets](https://www.emerald.com/sef/article/doi/10.1108/SEF-12-2025-0942/1363900/Human-signals-automated-noise-identifying-the-true) found sentiment polarity had no predictive power for returns, while public attention improved a GARCH-X volatility model and reacted to price shocks instead of forecasting them. Another [study](https://ijeti.uniben.edu/?p=510) reports sentiment leading volatility. A [minute-level study](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.644102/pdf) compared GARCH with GARCH-X using tweets, trading volume and order book signals.

| Arm | Inputs | Models | Question it answers |
| --- | --- | --- | --- |
| 1. Statistics only | Binance returns, realized volatility, volume | GARCH family (GARCH, EGARCH, GJR-GARCH), a HAR-style regression on realized volatility, historical VaR | How far do prices alone get you? |
| 2. NLP only | Hourly post volume, sentiment shares, topic shares, scam and shill share, event flags | Regularized regression, gradient boosting | Do posts alone predict risk? |
| 3. Combined | Everything from arms 1 and 2 | GARCH-X (GARCH with outside signals), gradient boosting on both feature sets | Do posts add value beyond prices? |

**Targets:** next-hour and next-day realized volatility, the probability of a large move, and VaR exceedances. Leave price direction for later, since the large tweet study found no directional content in polarity.

- **Walk-forward testing.** Train on an expanding past window, test on the next block, and repeat. Never shuffle time series.
- **No leakage.** Features may only use posts published before each bar closes, and posts must be scored by a classifier trained on data older than the test block.
- **Baselines and tests.** Every arm must beat "yesterday's volatility". Use MSE and QLIKE for volatility, Kupiec and Christoffersen backtests for VaR, and a Diebold-Mariano test to check whether arm 3 beats arm 1.
- **Bot filtering.** Automation-like accounts dominated the large tweet dataset and diluted polarity, so run the Verifier's bot and shill filters first and compare results with and without them.
- **Report nulls.** If NLP adds nothing, say so with the numbers; that is a valid finding. Briefs describe signals, never trades.

## Roadmap

Build in seven phases, each ending on a gate that must pass before the next starts.

1. **Foundation.** Bluesky, Mastodon and Binance collectors, normalization, deduplication and the raw store. Gate: a continuous run with source deletions synced.
2. **Labeling.** The Curator builds first labeled sets and the human-labeled holdout. Gate: labels and holdout reviewed and accepted by a human.
3. **Model bake-off.** Run SemIf on the holdout per task and per platform, and fall back to a fine-tuned model only where it underperforms. Gate: SemIf scored and calibrated, with any fallbacks chosen and logged.
4. **Signal study.** Run the statistics-only, NLP-only and combined arms with walk-forward tests on Bitcoin and Ethereum. Gate: results reported with out-of-sample tests, including a null result if that is what you find.
5. **Analysis.** The Analyst and Verifier run on replayed weeks. Gate: replay surfaces the known events with few false alarms.
6. **Reporting and autonomy.** The Reporter and Orchestrator run on a schedule with budgets and halts. Gate: every brief passes the citation check.
7. **Expansion.** Add Reddit for analysis use only. Gate: current Reddit terms reviewed.

## Decisions

- [x] Research use only, not a commercial product
- [x] No X data: sources are Bluesky, Mastodon and Reddit
- [x] Scope: Bitcoin and Ethereum ecosystems
- [x] English only
- [x] Models run locally on an RTX 5070 Ti
- [x] Briefs are for you alone: a daily digest plus on-demand briefs; scam and event alerts can be added later

## Sources

Source facts come from search results, not opened pages; verify terms and prices before building. As of the date above.

- [Playing with the Bluesky firehose](https://anderegg.ca/2024/11/25/playing-with-the-bluesky-firehose)
- [Bluesky dataset and consent (Gigazine)](https://wbgsv0a.gigazine.net/gsc_news/en/20241128-bluesky-machine-learning)
- [Bluesky Trending Topics (Apify)](https://apify.com/jrpbuilds/bluesky-trending-topics.md)
- [Mastodon scraper (Apify)](https://apify.com/puskin/mastodon-scraper.md)
- [Reddit API providers 2026](https://apiserpent.com/blog/best-reddit-api-providers-2026)
- [Bluesky 2M posts dataset](https://huggingface.co/Edens-Gate/Bluesky-2M-backup)
- [Binance Vision dataset terms](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf)
- [Guide to Binance historical data](https://dev.to/pi19404/a-technical-guide-to-downloading-and-managing-binance-historical-crypto-market-data-d82)
- [Binance read-only market-data mirror](https://pkg.go.dev/github.com/algo-one/binance-data-downloader/internal/vision)
- [Binance Spot market data docs](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/ws-api/market)
- [Community cache notes on restated Binance candles](https://github.com/finom/static-klines)
- [CryptoBERT paper](https://www.sentic.net/cryptocurrency-sentiment-classification.pdf)
- [kk08 CryptoBERT model card](https://huggingface.co/kk08/CryptoBERT)
- [FinTwitBERT model card](https://huggingface.co/StephanAkkerman/FinTwitBERT)
- [Bitcoin tweets, sentiment and attention study (Emerald)](https://www.emerald.com/sef/article/doi/10.1108/SEF-12-2025-0942/1363900/Human-signals-automated-noise-identifying-the-true)
- [Investor sentiment and crypto volatility study](https://ijeti.uniben.edu/?p=510)
- [Time-varying Bitcoin volatility with GARCH-X (Frontiers)](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2021.644102/pdf)
- [SemIf (formerly OpenJev) repository](https://github.com/TheoLeeCJ/SemIf-OpenJev)
