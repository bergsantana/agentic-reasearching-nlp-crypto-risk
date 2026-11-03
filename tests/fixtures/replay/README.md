# Frozen replay day

One recorded UTC day of Jetstream, Mastodon, trend and Binance messages, fed back through the
collectors by `cryptonlp replay run`. CI uses it from step 3b on (the thin end-to-end loop).

To make it:

1. Set `CRYPTONLP_RECORD_REPLAY=true` in `.env` and restart the collectors
   (`docker compose --profile collect up -d`). Messages go to `data/replay/<day>/`
   (git-ignored).
2. After a full UTC day has passed, freeze it:
   `uv run cryptonlp replay freeze 2026-11-05` writes `tests/fixtures/replay/2026-11-05/` with
   a `manifest.json` of message counts and SHA-256 hashes. It refuses a day that doesn't cover
   midnight to midnight.
3. Set `CRYPTONLP_RECORD_REPLAY=false` again.
4. Check it replays: `uv run cryptonlp replay run tests/fixtures/replay/2026-11-05`
   (test database only).

Before committing a day here, decide how to handle the real posts it contains: a public
repository would publish their text and author IDs, and would keep posts deleted at the source
in the git history, which deletion sync can't reach.
