"""Step 6: classifier metrics, calibration and H4's test, against values worked out by hand."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from cryptonlp.classify import metrics as m

CLASSES = ("negative", "neutral", "positive")


def onehot_probs(pred: list[int], k: int = 3, p: float = 0.8) -> np.ndarray:
    out = np.full((len(pred), k), (1 - p) / (k - 1))
    out[np.arange(len(pred)), pred] = p
    return out


def test_single_label_by_hand() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = [0, 1, 1, 1, 2, 0]
    tm = m.single_label_metrics(CLASSES, y, onehot_probs(pred))
    # negative: tp 1, fp 1, fn 1 -> P = R = F = 0.5
    # neutral:  tp 2, fp 1, fn 0 -> P 2/3, R 1, F 0.8
    # positive: tp 1, fp 0, fn 1 -> P 1, R 0.5, F 2/3
    assert tm.per_class["negative"].f1 == pytest.approx(0.5)
    assert tm.per_class["neutral"].f1 == pytest.approx(0.8)
    assert tm.per_class["positive"].f1 == pytest.approx(2 / 3)
    assert tm.macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert tm.balanced_accuracy == pytest.approx((0.5 + 1 + 0.5) / 3)
    assert tm.accuracy == pytest.approx(4 / 6)
    # Brier per row: a right row is off by 0.2, 0.1, 0.1; a wrong one by 0.9 (truth), 0.8, 0.1.
    right = 0.2**2 + 2 * 0.1**2
    wrong = 0.1**2 + 0.9**2 + 0.8**2  # true class gets 0.1, predicted 0.8, other 0.1
    assert tm.brier == pytest.approx((4 * right + 2 * wrong) / 6)
    # every confidence is 0.8, accuracy 4/6: ECE = |4/6 - 0.8|
    assert tm.ece == pytest.approx(abs(4 / 6 - 0.8))
    assert tm.macro_f1 == pytest.approx(m.macro_f1(y, np.array(pred), 3))


def test_weights_count_posts_as_their_strata() -> None:
    y = np.array([0, 1])
    pred = [0, 0]
    w = np.array([1.0, 3.0])
    tm = m.single_label_metrics(CLASSES, y, onehot_probs(pred), weights=w)
    assert tm.accuracy == pytest.approx(0.25)
    # The same as repeating the second post three times.
    rep = m.single_label_metrics(CLASSES, np.array([0, 1, 1, 1]), onehot_probs([0, 0, 0, 0]))
    assert tm.macro_f1 == pytest.approx(rep.macro_f1)
    assert tm.brier == pytest.approx(rep.brier)


def test_absent_classes_stay_out_of_the_macro_average() -> None:
    y = np.array([0, 0, 1])
    tm = m.single_label_metrics(CLASSES, y, onehot_probs([0, 0, 1]))
    assert tm.macro_f1 == 1.0 and tm.per_class["positive"].support == 0


def test_temperature_fit_recovers_the_true_temperature() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, size=(4000, 3))
    true_t = 2.5
    p = m.softmax(logits, true_t)
    y = np.array([rng.choice(3, p=row) for row in p])
    t = m.fit_temperature(logits, y)
    assert t == pytest.approx(true_t, rel=0.1)
    assert m.nll(logits, y, t) <= m.nll(logits, y, 1.0)
    # Calibration never moves the argmax.
    assert (m.softmax(logits, t).argmax(1) == logits.argmax(1)).all()


def test_multi_label_other_is_predicted_when_no_topic_is() -> None:
    labels = ("a", "b", "other")
    y = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    probs = np.array([[0.9, 0.2, 0.08], [0.1, 0.3, 0.63], [0.7, 0.4, 0.18]])
    tm = m.multi_label_metrics(labels, y, probs, decision_cols=[0, 1])
    # a: tp 2; b: fn 1 (f1 0); other: tp 1.
    assert tm.per_class["a"].f1 == 1.0
    assert tm.per_class["b"].f1 == 0.0
    assert tm.per_class["other"].f1 == 1.0
    assert tm.macro_f1 == pytest.approx(2 / 3)
    assert tm.accuracy == pytest.approx(2 / 3)  # exact set match on rows 1 and 2


def test_noninferiority_bootstrap() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 600)
    good = np.where(rng.random(600) < 0.8, y, rng.integers(0, 3, 600))
    bad = np.where(rng.random(600) < 0.4, y, rng.integers(0, 3, 600))

    def test(a: np.ndarray, b: np.ndarray) -> m.NonInferiority:
        return m.noninferiority_bootstrap(
            y, a, b, 3, margin=0.05, alpha=0.05, replicates=500, seed=7
        )

    better = test(good, bad)
    assert better.supported and better.estimate > 0.2 and better.p_value < 0.01
    worse = test(bad, good)
    assert not worse.supported and worse.p_value > 0.5
    same = test(good, good)
    assert same.estimate == 0 and same.supported  # equal is within any positive margin
    again = test(good, bad)
    assert again == better  # the seed fixes it
    # Decision and p-value agree.
    for r in (better, worse, same):
        assert r.supported == (r.p_value < r.alpha) or abs(r.lower + r.margin) < 1e-3


def test_superiority_bootstrap_and_multilabel_macro_f1() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 800)
    good = np.where(rng.random(800) < 0.85, y, rng.integers(0, 3, 800))
    bad = np.where(rng.random(800) < 0.5, y, rng.integers(0, 3, 800))

    def f(pred: np.ndarray) -> Callable[[np.ndarray], float]:
        return lambda idx: m.macro_f1(y[idx], pred[idx], 3)

    clear = m.superiority_bootstrap(800, f(good), f(bad), alpha=0.025, replicates=300, seed=1)
    assert clear.better and clear.lower > 0
    tie = m.superiority_bootstrap(800, f(good), f(good), alpha=0.025, replicates=300, seed=1)
    assert not tie.better and tie.estimate == 0
    truth = np.array([[1, 0, 0], [0, 0, 1], [1, 1, 0]], dtype=bool)
    pred = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    assert m.multilabel_macro_f1(truth, pred) == pytest.approx(2 / 3)  # as multi_label_metrics


def test_fallback_rule() -> None:
    from cryptonlp.classify.bakeoff import decide_task
    from cryptonlp.schemas.labels import Task

    kw = {"min_macro_f1": 0.40, "min_scam_recall": 0.70}
    better = {"openjev": {"macro_f1": 0.7, "better": True}}
    close = {"openjev": {"macro_f1": 0.62, "better": False}}
    keep = decide_task(Task.SENTIMENT, 0.6, close, scam_recall=None, **kw)
    assert keep == {"decision": "keep_semif", "reasons": [], "use": "semif"}
    beaten = decide_task(Task.SENTIMENT, 0.6, better, scam_recall=None, **kw)
    assert beaten["decision"] == "fallback" and beaten["use"] == "openjev"
    low = decide_task(Task.EVENT, 0.35, close, scam_recall=None, **kw)
    assert low["use"] == "openjev" and "floor" in low["reasons"][0]
    alone = decide_task(Task.TOPIC, 0.35, {}, scam_recall=None, **kw)
    assert alone["decision"] == "fallback" and alone["use"] == "semif"  # nothing else scored
    scam = decide_task(Task.SCAM, 0.6, close, scam_recall=0.5, **kw)
    assert scam["decision"] == "fallback" and "recall" in scam["reasons"][0]
    ok_scam = decide_task(Task.SCAM, 0.6, close, scam_recall=0.8, **kw)
    assert ok_scam["decision"] == "keep_semif"
