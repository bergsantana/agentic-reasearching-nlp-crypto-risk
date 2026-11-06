"""Step 3: language filter, dedupe, entities, spam rules, pseudonymization."""

from __future__ import annotations

import stat
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from cryptonlp.normalize import review
from cryptonlp.normalize.config import (
    DedupeConfig,
    LanguageConfig,
    NormalizeConfig,
    load_normalize_config,
)
from cryptonlp.normalize.dedupe import NearDuplicateIndex, from_bytes, signature, to_bytes
from cryptonlp.normalize.entities import AssetMatcher, addresses, cashtags, hashtags
from cryptonlp.normalize.language import identify, primary_subtag
from cryptonlp.normalize.pipeline import Normalizer
from cryptonlp.normalize.pseudonym import (
    PseudonymKey,
    PseudonymKeyError,
    generate_key,
    load_key,
)
from cryptonlp.normalize.spam import SpamFilter
from cryptonlp.normalize.text import for_language, text_sha256
from cryptonlp.rawstore import CollectedPost, RawStore
from cryptonlp.schemas import Origin, RawPost, Source
from tests.factories import T0

KEY = PseudonymKey(bytes(range(32)))


@pytest.fixture(scope="module")
def cfg() -> NormalizeConfig:
    return load_normalize_config()


# --- config --------------------------------------------------------------------------------


def test_repo_config_loads_and_has_a_stable_hash(cfg: NormalizeConfig) -> None:
    assert cfg.dedupe.bands * cfg.dedupe.rows == cfg.dedupe.num_perm
    assert set(cfg.entities.assets) == {"BTC", "ETH"}
    assert cfg.sha256 == load_normalize_config().sha256
    assert len(cfg.spam.rules) >= 5


def test_bands_must_divide_num_perm() -> None:
    with pytest.raises(ValueError, match="bands"):
        DedupeConfig(num_perm=64, bands=10)


# --- language ------------------------------------------------------------------------------

LANG = LanguageConfig(min_letters=15)


def test_language_is_detected_from_the_prose() -> None:
    texts = [
        "Bitcoin falls below $77,700 as recent buyers slip into losses https://t.co/x",
        "Hackern gelang es, 6,3 Milliarden Dollar von Binance zu stehlen, indem sie logen.",
        "Binance retoma levantamentos de forma faseada após roubo de criptomoedas.",
    ]
    # Declared tags are often wrong; the text wins when there is enough of it.
    results = identify(texts, [["de"], [], ["en"]], LANG)
    assert [r.lang for r in results] == ["en", "de", "pt"]
    assert all(r.method == "detected" and r.confidence for r in results)


def test_short_or_ticker_only_text_falls_back_to_the_declared_tag() -> None:
    texts = ["🐳 20.3K $ETH ($46.4M) 0xceb6…66ea → #whale", "gm", "$BTC $ETH 🚀"]
    results = identify(texts, [["en-US"], [], []], LANG)
    expected = [("en", "declared"), ("und", "none"), ("und", "none")]
    assert [(r.lang, r.method) for r in results] == expected


def test_prose_for_language_detection() -> None:
    text = "Big news @alice.bsky.social: $BTC #Bitcoin 0xABCDEF up 5% https://x.com/a today"
    assert for_language(text) == "Big news : up today"
    assert primary_subtag("zh-CN") == "zh"


# --- entities ------------------------------------------------------------------------------


def test_cashtags_and_hashtags() -> None:
    text = "$btc and $ETH up, $6.43B stolen, US$6,3 bi, #Bitcoin #ETH #1 https://x.com/#frag $BTC"
    assert cashtags(text) == ("$BTC", "$ETH")
    assert hashtags(text) == ("bitcoin", "eth")


def test_assets_from_the_lexicon(cfg: NormalizeConfig) -> None:
    m = AssetMatcher(cfg.entities)
    assert m.assets("Ethereum and #bitcoin both up") == ("BTC", "ETH")
    assert m.assets("$ETH looks weak") == ("ETH",)
    assert m.assets("lightning-network fees") == ("BTC",)
    assert m.assets("ethics of something, bitcoiners") == ()
    assert m.assets("see https://example.com/bitcoin") == ()  # URLs are not text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "send to 0x52908400098527886E0F7030069857D2E4169EE7 now",
            ("evm:0x52908400098527886e0f7030069857d2e4169ee7",),
        ),
        (
            "BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4",
            ("btc:bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",),
        ),
        (
            "taproot bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",
            ("btc:bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",),
        ),
        ("genesis 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", ("btc:1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",)),
        ("p2sh 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy", ("btc:3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",)),
        (
            "mint EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            ("sol:EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",),
        ),
        ("bad checksum 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb", ()),
        ("bad bech32 bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5", ()),
        ("short hex 0x52908400098527886E0F7030069857D2E4169E", ()),
        ("lowercase only abcdefghijkmnopqrstuvwxyzabcdefghijk", ()),
        ("in a link https://solscan.io/token/EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", ()),
    ],
)
def test_addresses_are_validated(text: str, expected: tuple[str, ...]) -> None:
    assert addresses(text) == expected


# --- spam ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "has_url", "expected"),
    [
        ("Send 0.1 BTC to the address and receive 0.2 BTC back!", False, {"giveaway_send_receive"}),
        ("Double your ETH in 24 hours", False, {"double_your_coins"}),
        ("Claim your airdrop now: https://x.io", True, {"claim_with_link"}),
        ("Claim your airdrop", False, set()),  # no link to claim it at
        ("Connect your wallet to validate: https://x.io", True, {"wallet_connect_phish"}),
        ("Never share your seed phrase with anyone", False, set()),
        ("Scam alert: a fake site asks you to verify your wallet https://x.io", True, set()),
        ("Please enter your seed phrase to restore", False, {"seed_phrase_request"}),
        ("Guaranteed profit, 30% daily returns", False, {"guaranteed_returns"}),
        ("DM me on telegram for investment signals", False, {"dm_for_profit"}),
        ("Bitcoin falls below $77,700 after the Binance hack", True, set()),
    ],
)
def test_spam_rules(cfg: NormalizeConfig, text: str, has_url: bool, expected: set[str]) -> None:
    rules = SpamFilter(cfg.spam).rules_matched(text, has_url=has_url, n_cashtags=1, n_hashtags=0)
    assert set(rules) == expected


def test_tag_stuffing_is_spam(cfg: NormalizeConfig) -> None:
    f = SpamFilter(cfg.spam)
    limits = {"n_cashtags": cfg.spam.max_cashtags, "n_hashtags": cfg.spam.max_hashtags}
    assert f.rules_matched("x", has_url=False, **limits) == ()
    over = {k: v + 1 for k, v in limits.items()}
    assert f.rules_matched("x", has_url=False, **over) == (
        "too_many_cashtags",
        "too_many_hashtags",
    )


# --- pseudonymization ----------------------------------------------------------------------


def test_author_hash_is_keyed_and_per_source() -> None:
    h = KEY.author_hash("bluesky", "did:plc:alice")
    assert h == KEY.author_hash("bluesky", "did:plc:alice")
    assert len(h) == 64 and "alice" not in h
    assert h != KEY.author_hash("mastodon", "did:plc:alice")
    assert h != PseudonymKey(bytes(32)).author_hash("bluesky", "did:plc:alice")
    assert KEY.key_id != PseudonymKey(bytes(32)).key_id


def test_keygen_writes_a_private_key_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "pseudonym.key"
    key = generate_key(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_key(path).key_id == key.key_id
    with pytest.raises(PseudonymKeyError, match="never overwritten"):
        generate_key(path)
    with pytest.raises(PseudonymKeyError, match="keygen"):
        load_key(tmp_path / "missing.key")
    (tmp_path / "bad.key").write_text("not hex")
    with pytest.raises(PseudonymKeyError):
        load_key(tmp_path / "bad.key")


# --- near duplicates -----------------------------------------------------------------------

NEWS = (
    "Spot bitcoin ETFs post $643 million daily outflow after the Binance hack, "
    "the largest in months"
)


def test_near_copies_match_and_different_posts_do_not(cfg: NormalizeConfig) -> None:
    dd = cfg.dedupe
    a = signature(NEWS + " https://a.example/1", dd)
    b = signature(NEWS.replace("$643", "$644") + " https://b.example/2 @bot", dd)
    c = signature(
        "Ethereum developers schedule the next upgrade for early December on testnets", dd
    )
    assert a is not None and b is not None and c is not None
    assert signature("too short", dd) is None
    assert (from_bytes(to_bytes(a)) == a).all()

    index = NearDuplicateIndex(dd)
    index.add("a", T0, a, None)
    match = index.query(b, T0 + timedelta(minutes=1))
    assert match is not None and match.root == "a" and match.score >= dd.threshold
    assert index.query(c, T0 + timedelta(minutes=1)) is None
    # Never a duplicate of a post received later.
    assert index.query(b, T0 - timedelta(minutes=1)) is None


# --- the pipeline (test database) ----------------------------------------------------------


def raw(
    n: int,
    text: str,
    minutes: float,
    *,
    source: Source = Source.BLUESKY,
    author: str = "did:plc:alice",
    langs: tuple[str, ...] = (),
    origin: Origin = Origin.LIVE,
) -> CollectedPost:
    uri = (
        f"at://{author}/app.bsky.feed.post/{n}"
        if source is Source.BLUESKY
        else f"https://mastodon.social/users/x/statuses/{n}"
    )
    return CollectedPost(
        post=RawPost(
            source=source,
            uri=uri,
            author_id=author,
            text=text,
            created_at=T0,
            received_at=T0 + timedelta(minutes=minutes),
            langs=langs,
            origin=origin,
            collector_run_id="20261103T120000Z-0123456789ab",
        ),
        payload={"n": n},
    )


CORPUS = [
    raw(1, NEWS + " https://a.example/1", 0),
    raw(2, NEWS.upper() + "  https://a.example/1", 1, author="did:plc:bob"),  # exact
    raw(
        3,
        NEWS.replace("$643", "$644") + " https://b.example/2",
        2,
        source=Source.MASTODON,
        author="https://mastodon.social/users/x",
    ),  # near, cross-platform
    raw(4, "Hackern gelang es, Milliarden Dollar von Binance zu stehlen, sagt die Polizei.", 3),
    raw(5, "Double your BTC today! Send 0.1 BTC and receive 0.2 BTC back https://x.io", 4),
    raw(6, "Ethereum developers schedule the next upgrade for early December on testnets", 5),
    raw(7, "$BTC 🚀", 6, langs=("en",)),
    raw(8, NEWS + " https://c.example/3", 7, origin=Origin.BACKFILL),  # near copy, and backfill
]


def by_n(conn: psycopg.Connection) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        "SELECT uri, lang, lang_method, is_spam, spam_rules, duplicate_of, duplicate_kind, "
        "near_duplicate_score, in_study, author_hash, assets FROM clean.posts"
    ).fetchall()
    cols = [
        "uri",
        "lang",
        "lang_method",
        "is_spam",
        "spam_rules",
        "duplicate_of",
        "duplicate_kind",
        "score",
        "in_study",
        "author_hash",
        "assets",
    ]
    return {int(r[0].rsplit("/", 1)[1]): dict(zip(cols, r, strict=True)) for r in rows}


@pytest.fixture
def stored(db: psycopg.Connection, store: RawStore, run_id: str) -> RawStore:
    store.store_posts(db, CORPUS, run_id)
    return store


@pytest.mark.db
def test_pipeline_decisions(
    db: psycopg.Connection, stored: RawStore, cfg: NormalizeConfig, run_id: str
) -> None:
    stats = Normalizer(cfg, KEY, stored, run_id).run(db)
    assert (stats.normalized, stats.exact_duplicates, stats.near_duplicates) == (8, 1, 2)
    rows = by_n(db)
    uri1 = rows[1]["uri"]
    assert rows[1]["in_study"] and rows[1]["assets"] == ["BTC"]
    assert (rows[2]["duplicate_kind"], rows[2]["duplicate_of"]) == ("exact", uri1)
    assert (rows[3]["duplicate_kind"], rows[3]["duplicate_of"]) == ("near", uri1)
    assert rows[3]["score"] >= cfg.dedupe.threshold
    assert (rows[4]["lang"], rows[4]["in_study"]) == ("de", False)
    assert rows[5]["is_spam"] and "giveaway_send_receive" in rows[5]["spam_rules"]
    assert rows[6]["in_study"] and rows[6]["assets"] == ["ETH"]
    assert (rows[7]["lang"], rows[7]["lang_method"], rows[7]["in_study"]) == (
        "en",
        "declared",
        True,
    )
    assert rows[8]["duplicate_kind"] == "near" and not rows[8]["in_study"]
    assert rows[1]["author_hash"] == KEY.author_hash("bluesky", "did:plc:alice")
    assert db.execute("SELECT pending FROM monitoring.normalize_backlog").fetchone() == (0,)
    # Nothing left to do on the next run.
    assert Normalizer(cfg, KEY, stored, run_id).run(db).normalized == 0


@pytest.mark.db
def test_batching_does_not_change_any_decision(
    db: psycopg.Connection, stored: RawStore, cfg: NormalizeConfig, run_id: str
) -> None:
    Normalizer(cfg, KEY, stored, run_id).run(db)
    all_at_once = by_n(db)
    db.execute("TRUNCATE clean.labels, clean.embeddings, clean.minhash, clean.posts")
    stats = Normalizer(cfg, KEY, stored, run_id).run(db, batch_size=1)
    assert stats.batches == 8
    assert by_n(db) == all_at_once


@pytest.mark.db
def test_deletions_are_honored(
    db: psycopg.Connection, stored: RawStore, cfg: NormalizeConfig, run_id: str
) -> None:
    uris = {int(u.rsplit("/", 1)[1]): u for (u,) in db.execute("SELECT uri FROM raw.posts")}
    # Deleted before normalization: never enters clean.posts.
    db.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) "
        "VALUES ('bluesky', %s, 'stream', %s)",
        (uris[6], run_id),
    )
    Normalizer(cfg, KEY, stored, run_id).run(db)
    assert 6 not in by_n(db)
    # Deleted afterwards: removed at once, with its signature.
    db.execute(
        "INSERT INTO raw.deletions (source, uri, method, run_id) "
        "VALUES ('bluesky', %s, 'stream', %s)",
        (uris[1], run_id),
    )
    assert 1 not in by_n(db)
    assert db.execute(
        "SELECT count(*) FROM clean.minhash WHERE uri = %s", (uris[1],)
    ).fetchone() == (0,)
    assert db.execute("SELECT pending FROM monitoring.normalize_backlog").fetchone() == (0,)


@pytest.mark.db
def test_review_sample(
    db: psycopg.Connection, stored: RawStore, cfg: NormalizeConfig, run_id: str, tmp_path: Path
) -> None:
    Normalizer(cfg, KEY, stored, run_id).run(db)
    rows = review.sample(db, per_group=5)
    groups = {r["group"] for r in rows}
    assert {"kept_for_study", "not_english", "spam", "exact_duplicate", "near_duplicate"} <= groups
    near = next(r for r in rows if r["group"] == "near_duplicate")
    assert near["original_text"].startswith("Spot bitcoin")
    assert review.sample(db, per_group=5) == rows  # same seed, same sample
    path = tmp_path / "review.csv"
    review.write_csv(rows, path)
    assert path.read_text(encoding="utf-8").startswith("group,correct,note,")


def test_text_hash_ignores_case_and_spacing() -> None:
    assert text_sha256("Bitcoin  UP\n") == text_sha256("bitcoin up")
    assert text_sha256("bitcoin up") != text_sha256("bitcoin down")


@pytest.mark.db
def test_nul_characters_do_not_block_the_batch(
    db: psycopg.Connection, store: RawStore, cfg: NormalizeConfig, run_id: str
) -> None:
    # Seen live: a client that writes é as NUL + "e9". PostgreSQL rejects NUL in text.
    mangled = "La Cor\x00e9e du Nord promet une riposte \x00e0 la politique des \x00c9tats-Unis"
    store.store_posts(db, [raw(1, mangled, 0, langs=("fr\x00",)), raw(2, NEWS, 1)], run_id)
    stats = Normalizer(cfg, KEY, store, run_id).run(db)
    assert stats.normalized == 2
    row = db.execute(
        "SELECT text, langs_declared, text_sha256 FROM clean.posts WHERE uri LIKE '%/1'"
    ).fetchone()
    assert row is not None
    text, langs, digest = row
    assert text == mangled.replace("\x00", "") and langs == ["fr"]
    assert digest == text_sha256(text)
