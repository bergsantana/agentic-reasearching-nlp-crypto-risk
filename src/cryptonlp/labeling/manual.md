# Labeling manual

Version 1.0 · draft, not yet registered · Nov 3, 2026

This manual says how the human-labeled set is drawn and labeled. It is registered in the pre-registration ledger before the first post is labeled, and it becomes Appendix A of the article. Any later change is registered as a new version, with the reason, before it is used.

## 1. Purpose

The labeled set is the reference the classifiers are measured against. The first 500 posts (by time) calibrate the classifiers' probabilities, and the other 1,500 are the test set for the comparison of SemIf, OpenJev (Wortega), CryptoBERT, FinBERT and VADER, including hypothesis H4. The labels describe what a post's text says. They are not a judgment on whether the author is right.

## 2. What you see and what you use

You see the text of one post and nothing else: no platform, author, time, images, links, thread or model output. This is all a classifier sees, so the labels rest on the same information.

- Label from the text alone. Do not open links, and do not look up the post, its author, its replies or the price at the time.
- You may look up what a name, ticker or slang term refers to, for example whether a name is a crypto project or an exchange.
- Hashtags and cashtags are part of the text. A tag counts as a reference (task 5), but a list of tags alone does not make a topic (task 2).

## 3. Procedure

1. **One batch a day.** Each day a batch of 500 posts is drawn from the previous UTC day (`cryptonlp labeling draw`) and appears as a new project in Label Studio. Label it that day, in the order shown.
2. **Sessions.** Take a break at least every 45 minutes. Stop for the day if your attention drops; the rest of the batch can wait for the next session.
3. **Answer every task.** For each post, answer the five tasks in order (section 5), then submit. The interface does not let you skip a task. If the post should not be labeled at all, use a problem flag instead (section 4).
4. **No pre-filled answers.** The interface starts empty for every post, and you never see a model's labels while labeling.
5. **Corrections.** If you notice a mistake in a post you already submitted, open it and correct it. The last version counts, and every version is kept. Do not go back to change earlier posts because of a later post that seemed similar: correct only what the manual says is wrong.
6. **Doubt.** When unsure, choose the best answer under this manual and tick "Unsure". Do not leave a task unanswered because of doubt.
7. **Pull.** After each session, run `cryptonlp labeling pull`. Problems in the annotations (a missing answer, conflicting choices) are listed by `cryptonlp labeling status`.
8. **Re-labeling, blind.** At least 14 days after the first pass, 200 posts are drawn again and appear in a separate project. Label them as if new. Do not open the first-pass projects, exports or this manual's examples to recall what you chose. Read the manual again before you start.

## 4. Posts not to label

Flag the post, and answer nothing else, only if:

- **Not English:** most of the prose is in another language. Short foreign words or names in an English post do not count.
- **Unreadable or empty:** no prose to judge, for example only links, tags or emoji, or text too garbled to read.

Everything else is labeled, including posts that are not really about crypto (they get topic "Other" and asset "Neither"), spam, jokes and very short posts. Flagged posts leave the set and are counted. They are a measure of the language filter and of what the study's posts contain.

## 5. The tasks

The five tasks are independent. Answer each on its own: a scam can be positive, and a hack report can be about Bitcoin and price talk at the same time.

### 5.1 Sentiment (one answer)

How the post reads for someone holding the crypto assets, projects or market it discusses.

| Label | When |
| --- | --- |
| Positive | Optimistic, bullish, approving or excited about crypto; or it reports good news for prices or projects (gains, adoption, approvals, inflows) |
| Negative | Pessimistic, bearish, fearful, angry or disapproving; or it reports bad news (falls, losses, hacks, bans, outages, outflows). Warnings about scams are negative |
| Neutral | Factual with no good or bad side, questions, how-to content, posts whose positive and negative parts balance, and posts with no stance on crypto |

- Read sarcasm and irony for what the author means ("Great, another 20% dip, love it" is negative).
- Crypto slang has a direction: "moon", "LFG", "WAGMI", "bullish", "pump" are positive; "rekt", "NGMI", "rug", "dump", "bearish" are negative.
- When one side clearly dominates, choose it. Choose neutral only when neither does.
- This definition matches the mapping used in H4: CryptoBERT's bullish, neutral and bearish map to positive, neutral and negative.

### 5.2 Topic (one or more)

Choose every topic that is a substantive part of the post, not a passing word. "Other" goes alone.

| Label | Covers |
| --- | --- |
| Price talk | Prices, charts, trading, positions, predictions, gains and losses, liquidations, market moves, fund flows |
| Regulation | Laws, regulators, courts, enforcement, taxes, government policy on crypto, ETF approvals or rejections |
| Security incident | Hacks, exploits, thefts, phishing, rug pulls, vulnerabilities, scam reports and warnings |
| Protocol or tech | Development, upgrades, forks, layer 2s, mining, nodes, validators and staking mechanics, fees and network performance, wallets as software |
| Launch or airdrop | New tokens, token sales and presales, airdrops, mints, listings on exchanges |
| Macro | Interest rates, inflation, central banks, the economy, stock markets, the dollar, tariffs and geopolitics |
| Other | None of the above: adoption, culture, memes, personal stories, NFTs as art, general opinion, off-topic posts |

### 5.3 Scam and promotion (one answer)

| Label | When |
| --- | --- |
| Organic | Ordinary discussion, news, analysis and opinion, including posts about the author's own trades and warnings about scams |
| Promotion | Tries to get readers to buy a token, join a group or channel, sign up, follow, or use a referral link or code; advertises a token, project, service or signal group; hype with no content ("next 100x gem, don't miss it") |
| Likely scam | Tries to get money, keys or wallet access through deception: giveaways that ask you to send crypto first, fake support or "fund recovery", "claim" or "verify your wallet" links, impersonation, guaranteed returns |

- Choose "Likely scam" over "Promotion" when a reader who follows the post would probably lose money to deception, not just to a bad investment.
- A post that only mentions a project is not promotion. It becomes promotion when it asks the reader to act.

### 5.4 Event reported (one answer)

An event is a specific real-world occurrence that the post reports or discusses as having happened, or as officially announced.

| Label | When |
| --- | --- |
| Hack | Theft, exploit or breach of a protocol, exchange, bridge, wallet or its users |
| Outage | An exchange, network, wallet or service is down, halted or degraded (for example, withdrawals paused or a chain halted) |
| Listing | An exchange lists or delists a token, or opens a new trading pair or product |
| Regulatory action | A regulator, court, government or police acts: a lawsuit, fine, ban, approval, arrest, sanction, or a law passed or signed |
| None | No such event, including predictions, rumors, questions, general remarks ("hacks are everywhere") and scam attempts in progress |

- If the post reports more than one event, choose the main one. If none is clearly the main one, use this order: hack, outage, regulatory action, listing.
- Events in the past count when the post reports them as news or discusses them specifically ("the Bybit hack last February"), not when they are only an example.

### 5.5 Assets referred to (one or more)

Which of the two study assets the post refers to. "Neither" goes alone.

| Label | When |
| --- | --- |
| Bitcoin | The asset, its network or its direct forms: BTC, $BTC, #bitcoin, sats, wrapped bitcoin, Lightning, Bitcoin ETFs |
| Ethereum | The asset, its network or its direct forms: ETH, ether, staked or wrapped ether, Ethereum ETFs, Ethereum upgrades |
| Neither | Neither of the two, including other chains and "crypto" in general |

- Clear implicit references count ("the orange coin" is Bitcoin).
- A token that runs on Ethereum is not Ethereum, unless the post is about the network itself.
- A ticker in a list counts ("$BTC $ETH $SOL" refers to both).

## 6. Quick reference (keys)

| Task | Keys |
| --- | --- |
| Sentiment | 1 negative · 2 neutral · 3 positive |
| Topic | q price · w regulation · e security · r protocol/tech · t launch/airdrop · y macro · o other |
| Scam and promotion | a organic · s promotion · d likely scam |
| Event | z none · x hack · c outage · v listing · b regulatory action |
| Assets | f Bitcoin · g Ethereum · h neither |
| Only if needed | 8 not English · 9 unreadable · 0 unsure |

## 7. Sample and split

The settings are in `config/labeling.toml`, and every batch stores their SHA-256.

- **Population.** In-study posts: English, not rule-flagged spam, not duplicates, collected live, from Bluesky and Mastodon.
- **Daily batches.** One batch of 500 posts per UTC day, drawn after the day ends (with a 60-minute margin so every post is normalized), until 2,000 posts are labeled. The days are the time strata.
- **Strata within a day.** Per platform, two disjoint strata: scam or promotion candidates (a contract or wallet address, or a match on the candidate patterns in the settings) and all other posts. Candidates get 10% of the batch, or fewer if the day has fewer. The rest comes from the other posts. Both parts are split across platforms in proportion to the day's in-study volume. Within a stratum, posts are drawn by simple random sampling. The batch is shuffled before labeling, so the strata cannot be told apart.
- **Weights.** Each batch records every stratum's population N and sample size n. Per-class measures use the sample as drawn. Measures that depend on prevalence (accuracy, calibration, Brier score) are also reported re-weighted by N/n.
- **Top-ups.** Flagged posts (section 4) leave the set. If the usable total falls short of 2,000, a top-up batch of the missing size is drawn from the newest labeled day.
- **Re-labeling.** 200 posts whose first label was last edited at least 14 days earlier, in proportion to platform, labeled blind (section 3, item 8). Agreement is Cohen's kappa per task. For topic and assets it is the kappa of each label as a yes/no decision, averaged over the labels, together with the share of posts with exactly the same set.
- **Split.** By receive time: the 500 oldest usable posts are the calibration set, and the rest is the test set. The split is fixed once, by `cryptonlp labeling freeze`, and never changes. Prompts, rubrics and calibration may use only the calibration set.
- **Deletions.** A post deleted at the source leaves the set, its labels and its Label Studio task. Losses are reported.

## 8. Changes

| Version | Date | Change |
| --- | --- | --- |
| 1.0 | ⟦date registered⟧ | First version |
