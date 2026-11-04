"""Classifier metrics and calibration, in own code so every number is easy to audit.

Single-label tasks (sentiment, scam, event): per-class precision, recall and F1, macro F1,
balanced accuracy (mean recall), multi-class Brier score, and top-label expected calibration
error. Topic (multi-label): every label, "other" included, is a binary indicator; macro F1 and
balanced accuracy average the per-label values, Brier averages the squared errors over posts and
labels, and ECE pools the six yes/no topic decisions.

Every measure takes optional weights per post: the stratum weights N/n of the labeling sample
(claims M15), for the re-weighted estimates.

Calibration: one temperature per model and task, softmax(logits / T), T minimizing the log loss
on the calibration slice (SemIf's ``CALIBRATION.md``; Guo et al., 2017). It never changes the
chosen option, so F1 and balanced accuracy are the same before and after.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

Array = NDArray[np.float64]


def softmax(logits: Array, temperature: float = 1.0) -> Array:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: Array = e / e.sum(axis=-1, keepdims=True)
    return out


def _weights(n: int, weights: Array | None) -> Array:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be one non-negative number per post, not all zero")
    return w


# --- calibration ---------------------------------------------------------------------------


def nll(logits: Array, y: NDArray[np.int64], temperature: float) -> float:
    """Mean negative log-likelihood of the true options."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(-log_p[np.arange(len(y)), y].mean())


def fit_temperature(logits: Array, y: NDArray[np.int64]) -> float:
    """The temperature in [0.05, 20] that minimizes the log loss (searched on log T)."""
    if len(y) == 0:
        raise ValueError("no rows to calibrate on")
    res = minimize_scalar(
        lambda log_t: nll(logits, y, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(res.x))


def ece(
    confidence: Array, correct: Array, *, bins: int = 15, weights: Array | None = None
) -> float:
    """Expected calibration error: equal-width bins on the confidence, the weighted mean gap
    between each bin's accuracy and its mean confidence."""
    c = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    w = _weights(len(c), weights)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            wb = w[m]
            total += wb.sum() * abs(np.average(ok[m], weights=wb) - np.average(c[m], weights=wb))
    return float(total / w.sum())


def binary_ece(p: Array, y: Array, *, bins: int = 15, weights: Array | None = None) -> float:
    """ECE of yes/no decisions: confidence max(p, 1 - p), correct when the side chosen is y."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    choose = p > 0.5
    return ece(
        np.where(choose, p, 1 - p), (choose == (y > 0.5)).astype(float), bins=bins, weights=weights
    )


# --- single-label --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassStats:
    precision: float
    recall: float
    f1: float
    support: float  # weighted count of posts whose true label is this class


@dataclass(frozen=True)
class TaskMetrics:
    n: int
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    brier: float
    ece: float
    per_class: dict[str, ClassStats] = field(default_factory=dict)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def single_label_metrics(
    classes: Sequence[str],
    y_true: NDArray[np.int64],
    probs: Array,
    *,
    bins: int = 15,
    weights: Array | None = None,
) -> TaskMetrics:
    """``probs[i, k]``: probability of ``classes[k]`` for post i. The prediction is the argmax
    (first on ties). Classes absent from the truth are left out of macro F1 and balanced
    accuracy, as is usual; they still count in precision of the others."""
    n, k = probs.shape
    w = _weights(n, weights)
    pred = probs.argmax(axis=1)
    per: dict[str, ClassStats] = {}
    f1s, recalls = [], []
    for c in range(k):
        tp = float(w[(pred == c) & (y_true == c)].sum())
        fp = float(w[(pred == c) & (y_true != c)].sum())
        fn = float(w[(pred != c) & (y_true == c)].sum())
        p, r, f = _prf(tp, fp, fn)
        support = float(w[y_true == c].sum())
        per[classes[c]] = ClassStats(p, r, f, support)
        if support > 0:
            f1s.append(f)
            recalls.append(r)
    onehot = np.eye(k)[y_true]
    brier = float(np.average(((probs - onehot) ** 2).sum(axis=1), weights=w))
    correct = (pred == y_true).astype(float)
    return TaskMetrics(
        n=n,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        balanced_accuracy=float(np.mean(recalls)) if recalls else 0.0,
        accuracy=float(np.average(correct, weights=w)),
        brier=brier,
        ece=ece(probs.max(axis=1), correct, bins=bins, weights=w),
        per_class=per,
    )


def macro_f1(y_true: NDArray[np.int64], pred: NDArray[np.int64], k: int) -> float:
    """Unweighted macro F1 over the classes present in ``y_true`` (the bootstrap's statistic;
    the same definition as ``single_label_metrics``)."""
    f1s = []
    for c in range(k):
        if not (y_true == c).any():
            continue
        tp = float(((pred == c) & (y_true == c)).sum())
        fp = float(((pred == c) & (y_true != c)).sum())
        fn = float(((pred != c) & (y_true == c)).sum())
        f1s.append(_prf(tp, fp, fn)[2])
    return float(np.mean(f1s)) if f1s else 0.0


# --- multi-label (topic) -------------------------------------------------------------------


def multi_label_metrics(
    labels: Sequence[str],
    y_true: Array,
    probs: Array,
    *,
    decision_cols: Sequence[int],
    bins: int = 15,
    weights: Array | None = None,
) -> TaskMetrics:
    """``y_true[i, j]`` in {0, 1} and ``probs[i, j]`` = P(label j) for post i. A label is
    predicted when its probability passes 0.5, except the "other" column, which is predicted
    exactly when no decision column is (``decision_cols``: the topic columns with their own
    yes/no decision; the rest is "other")."""
    n, m = probs.shape
    w = _weights(n, weights)
    pred = np.zeros((n, m), dtype=bool)
    cols = list(decision_cols)
    pred[:, cols] = probs[:, cols] > 0.5
    for j in range(m):
        if j not in cols:
            pred[:, j] = ~pred[:, cols].any(axis=1)
    truth = y_true > 0.5
    per: dict[str, ClassStats] = {}
    f1s, bal = [], []
    for j, name in enumerate(labels):
        tp = float(w[pred[:, j] & truth[:, j]].sum())
        fp = float(w[pred[:, j] & ~truth[:, j]].sum())
        fn = float(w[~pred[:, j] & truth[:, j]].sum())
        tn = float(w[~pred[:, j] & ~truth[:, j]].sum())
        p, r, f = _prf(tp, fp, fn)
        support = float(w[truth[:, j]].sum())
        per[name] = ClassStats(p, r, f, support)
        if support > 0:
            f1s.append(f)
            spec = tn / (tn + fp) if tn + fp > 0 else 0.0
            bal.append((r + spec) / 2)
    brier = float(np.average(((probs - y_true) ** 2).mean(axis=1), weights=w))
    exact = (pred == truth).all(axis=1).astype(float)
    flat_w = np.repeat(w, len(cols))
    return TaskMetrics(
        n=n,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        balanced_accuracy=float(np.mean(bal)) if bal else 0.0,
        accuracy=float(np.average(exact, weights=w)),  # exact match of the whole label set
        brier=brier,
        ece=binary_ece(probs[:, cols].ravel(), y_true[:, cols].ravel(), bins=bins, weights=flat_w),
        per_class=per,
    )


def multilabel_macro_f1(truth: NDArray[np.bool_], pred: NDArray[np.bool_]) -> float:
    """Unweighted macro F1 over the label columns present in ``truth`` (the definition of
    ``multi_label_metrics``, for the bootstrap)."""
    f1s = []
    for j in range(truth.shape[1]):
        t, p = truth[:, j], pred[:, j]
        if not t.any():
            continue
        tp = float((p & t).sum())
        f1s.append(_prf(tp, float((p & ~t).sum()), float((~p & t).sum()))[2])
    return float(np.mean(f1s)) if f1s else 0.0


@dataclass(frozen=True)
class Superiority:
    estimate: float  # statistic of A minus statistic of B
    lower: float  # one-sided percentile lower bound at level alpha
    alpha: float
    better: bool  # lower > 0: A clearly beats B


def superiority_bootstrap(
    n: int,
    stat_a: Callable[[NDArray[np.int64]], float],
    stat_b: Callable[[NDArray[np.int64]], float],
    *,
    alpha: float,
    replicates: int,
    seed: int,
) -> Superiority:
    """Does A clearly beat B? ``stat_*(rows)`` computes each model's statistic on a set of row
    indices; posts are resampled with replacement and both are scored on the same resample."""
    rng = np.random.default_rng(seed)
    everyone = np.arange(n)
    diffs = np.empty(replicates)
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        diffs[b] = stat_a(idx) - stat_b(idx)
    lower = float(np.quantile(diffs, alpha))
    return Superiority(stat_a(everyone) - stat_b(everyone), lower, alpha, lower > 0)


# --- H4: paired bootstrap non-inferiority --------------------------------------------------


@dataclass(frozen=True)
class NonInferiority:
    estimate: float  # macro F1 of A minus macro F1 of B
    f1_a: float
    f1_b: float
    lower: float  # one-sided percentile lower bound at level alpha
    margin: float
    alpha: float
    p_value: float  # share of bootstrap differences at or below -margin
    supported: bool
    n: int
    replicates: int
    seed: int


def noninferiority_bootstrap(
    y_true: NDArray[np.int64],
    pred_a: NDArray[np.int64],
    pred_b: NDArray[np.int64],
    k: int,
    *,
    margin: float,
    alpha: float,
    replicates: int,
    seed: int,
) -> NonInferiority:
    """Is A's macro F1 at least B's minus ``margin``? Posts are resampled with replacement
    (both models scored on the same resample); the one-sided percentile lower bound of the
    difference must exceed -margin. The p-value, (1 + #{diff* <= -margin}) / (B + 1), is below
    alpha exactly when it does (up to the +1)."""
    n = len(y_true)
    rng = np.random.default_rng(seed)
    f_a, f_b = macro_f1(y_true, pred_a, k), macro_f1(y_true, pred_b, k)
    diffs = np.empty(replicates)
    for b in range(replicates):
        idx = rng.integers(0, n, n)
        diffs[b] = macro_f1(y_true[idx], pred_a[idx], k) - macro_f1(y_true[idx], pred_b[idx], k)
    lower = float(np.quantile(diffs, alpha))
    p = float((1 + (diffs <= -margin).sum()) / (replicates + 1))
    return NonInferiority(
        estimate=f_a - f_b,
        f1_a=f_a,
        f1_b=f_b,
        lower=lower,
        margin=margin,
        alpha=alpha,
        p_value=p,
        supported=lower > -margin,
        n=n,
        replicates=replicates,
        seed=seed,
    )
