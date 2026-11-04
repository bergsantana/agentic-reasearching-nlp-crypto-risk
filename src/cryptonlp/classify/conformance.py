"""Does SemIf on our vLLM server answer as SemIf's own code does?

SemIf's repository publishes its outputs for its 144 authored decisions on the same model
revision (``tests/fixtures/semif/``). We send the same decisions through our prompt builder
and the vLLM readout, and compare: the prompt hash (must be identical, row for row), the chosen
option, and the logits. The logits cannot be bit-identical: vLLM's BF16 kernels differ from
the transformers path SemIf ran, as SemIf's own fast paths differ from its reference by 5 to 6
argmaxes in 777 (Lee, 2026). So the check is on the chosen option, with the logit gap reported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cryptonlp.classify.prompt import Decision, Option
from cryptonlp.classify.semif import SemIfScorer, softmax
from cryptonlp.settings import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "semif" / "authored144.jsonl"
# Share of rows whose chosen option must match SemIf's reference.
MIN_AGREEMENT = 0.97


@dataclass(frozen=True)
class Conformance:
    rows: int
    prompt_hash_matches: int
    same_choice: int
    agreement: float
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    max_abs_prob_diff: float
    accuracy_ours: float
    accuracy_reference: float
    passed: bool
    mismatched_choices: list[str]


def load_fixture(path: Path = FIXTURE) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_decision(row: dict[str, Any]) -> Decision:
    return Decision(
        row["id"],
        row["state"],
        row["question"],
        tuple(Option(o["id"], o["description"]) for o in row["options"]),
    )


def check(scorer: SemIfScorer, rows: list[dict[str, Any]] | None = None) -> Conformance:
    rows = rows or load_fixture()
    scored = scorer.score([to_decision(r) for r in rows])
    hashes = same = right_ours = right_ref = 0
    diffs: list[float] = []
    prob_diffs: list[float] = []
    mismatched = []
    for r, s in zip(rows, scored, strict=True):
        ref = r["reference"]
        hashes += int(s.prompt_sha256 == ref["prompt_sha256"])
        ours = int(np.argmax(s.logits))
        theirs = int(np.argmax(ref["option_logits"]))
        same += int(ours == theirs)
        if ours != theirs:
            mismatched.append(r["id"])
        right_ours += int(ours == r["label"])
        right_ref += int(theirs == r["label"])
        d = np.abs(np.subtract(s.logits, ref["option_logits"]))
        diffs.extend(d.tolist())
        prob_diffs.append(
            float(np.max(np.abs(np.subtract(softmax(s.logits), ref["probabilities"]))))
        )
    n = len(rows)
    agreement = same / n
    return Conformance(
        rows=n,
        prompt_hash_matches=hashes,
        same_choice=same,
        agreement=agreement,
        max_abs_logit_diff=float(max(diffs)),
        mean_abs_logit_diff=float(np.mean(diffs)),
        max_abs_prob_diff=max(prob_diffs),
        accuracy_ours=right_ours / n,
        accuracy_reference=right_ref / n,
        passed=hashes == n and agreement >= MIN_AGREEMENT,
        mismatched_choices=mismatched,
    )


def as_dict(c: Conformance) -> dict[str, Any]:
    return asdict(c)
