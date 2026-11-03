from __future__ import annotations

import pytest

from cryptonlp.collectors.common import WatchFilter
from cryptonlp.config import WatchConfig, load_config

FILTER = WatchFilter(
    WatchConfig(terms=("bitcoin", "eth", "seed phrase", "rug pull"), cashtags=("BTC", "OP"))
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bitcoin is up", ("bitcoin",)),
        ("#Bitcoin and $BTC", ("bitcoin", "$btc")),
        ("ETH merge", ("eth",)),
        ("never share your seed phrase", ("seed phrase",)),
        ("total RUG-PULL", ("rug pull",)),
        ("$OP pumps", ("$op",)),
    ],
)
def test_matches(text: str, expected: tuple[str, ...]) -> None:
    assert FILTER.matches(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "bitcoinX is not a word match",
        "something ethical",
        "OP is an ordinary word without the $",  # bare tickers don't match
        "methane",
        "",
    ],
)
def test_no_match(text: str) -> None:
    assert FILTER.matches(text) == ()


def test_matches_are_unique_and_in_order() -> None:
    assert FILTER.matches("eth bitcoin ETH Bitcoin") == ("eth", "bitcoin")


def test_hash_ignores_order() -> None:
    a = WatchConfig(terms=("a", "b"), cashtags=("X",))
    b = WatchConfig(terms=("b", "a"), cashtags=("X",))
    c = WatchConfig(terms=("a", "c"), cashtags=("X",))
    assert a.sha256 == b.sha256 != c.sha256


def test_repo_config_loads() -> None:
    config = load_config()
    watch = WatchFilter(config.watch)
    assert watch.matches("Ethereum gas fees and $BTC") == ("ethereum", "btc", "$btc")
    assert config.binance.backfill_start.isoformat() == "2024-11-01"
