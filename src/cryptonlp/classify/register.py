"""H4's registration: the rubric, the hypothesis and the bake-off's experiment package, appended
to the ledger in one transaction, before the 1,500 test posts are scored (claims P18).

- The rubric (``config/semif/rubric.toml``) as a ``rubric`` document, full text.
- H4, confirmatory, in the article's words, with its analysis plan: the non-inferiority test,
  margin, level, bootstrap, the CryptoBERT label mapping, the split, the fallback rule.
- The experiment package ``H4.bakeoff``: the code commit, the frozen labeled set, the SemIf spec
  (model revision, prompt hash, temperatures), the baselines' pinned revisions.

``cryptonlp classify bakeoff export --split test`` refuses until this package is in the ledger
with the SemIf spec the config holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg

from cryptonlp import ledger
from cryptonlp.classify.bakeoff import H4_PACKAGE, _sha256_file
from cryptonlp.classify.config import ClassifyConfig, _resolve
from cryptonlp.classify.errors import ClassifyError
from cryptonlp.schemas.common import Phase
from cryptonlp.schemas.ledger import (
    DataFile,
    Document,
    DocumentKind,
    ExperimentPackage,
    Hypothesis,
    HypothesisKind,
    Proposer,
)
from cryptonlp.settings import REPO_ROOT

# Verbatim from the article (Section 4.8). Results quote it word for word.
H4_STATEMENT = (
    "Após calibração, o SemIf atinge F1 macro em sentimento igual ou superior ao do CryptoBERT "
    "no conjunto de teste."
)
RUBRIC_KEY = "semif-rubric"


class RegistrationError(ClassifyError):
    pass


def commit_of(run_id: str) -> str:
    commit = run_id.split("-")[1]
    if commit == "nogit" or run_id.endswith("-dirty"):
        raise RegistrationError(
            "register from committed code: the package cites the commit (run ID " + run_id + ")"
        )
    return commit


def analysis_plan(config: ClassifyConfig) -> dict[str, Any]:
    h, fb, b = config.h4, config.fallback, config.bakeoff
    return {
        "estimand": "macro F1 of SemIf minus macro F1 of CryptoBERT, sentiment, test split, "
        "Bluesky and Mastodon pooled, unweighted",
        "test": "non-inferiority, paired bootstrap over posts (percentile, one-sided)",
        "decision": "H4 holds when the one-sided lower bound at level alpha exceeds -margin",
        "p_value": "(1 + #{bootstrap difference <= -margin}) / (replicates + 1)",
        "margin": h.margin,
        "alpha": h.alpha,
        "replicates": h.bootstrap,
        "seed": h.seed,
        "cryptobert_mapping": b.cryptobert.labels,
        "calibration": "one temperature per model and task, fitted on the 500-post calibration "
        "slice; it does not change macro F1",
        "split": "time split of the labeled set: oldest 500 calibration, newest 1,500 test",
        "secondary": "per-class precision, recall and F1, balanced accuracy, Brier score, ECE "
        "and latency per 1,000 posts, per platform, unweighted and re-weighted by strata, for "
        "SemIf, Wortega's OpenJev, CryptoBERT, FinBERT and VADER (descriptive)",
        "fallback_rule": {
            "scope": "per task, test split, Bluesky and Mastodon pooled, unweighted",
            "rules": [
                "another model scored on the task clearly beats SemIf: the one-sided lower bound "
                "at alpha of (its macro F1 - SemIf's), paired bootstrap over posts with the H4 "
                "replicates and seed, is above 0",
                f"SemIf's macro F1 is below {fb.min_macro_f1}",
                f"scam only: SemIf's recall of likely_scam is below {fb.min_scam_recall}",
            ],
            "alpha": fb.alpha,
            "min_macro_f1": fb.min_macro_f1,
            "min_scam_recall": fb.min_scam_recall,
            "then": "the task moves to a model fine-tuned on the Curator's labels; until then, "
            "the best model that clearly beat SemIf, else the best other model by macro F1",
        },
        "reproducibility": f"{b.repeat} test posts scored twice by SemIf; share of unchanged "
        "labels reported",
    }


def records(
    config: ClassifyConfig,
    *,
    labeled_set: Path,
    phase: Phase,
    run_id: str,
) -> tuple[Document, Hypothesis, dict[str, Any]]:
    """The rubric document, H4, and the package's fields (its hypothesis_seq comes later)."""
    if not config.h4.decided:
        raise RegistrationError(
            "config/classify.toml has h4.decided = false: choose the margin and the fallback "
            "threshold (both still proposals), then set decided = true"
        )
    if not labeled_set.exists():
        raise RegistrationError(f"{labeled_set} does not exist (`cryptonlp labeling freeze`)")
    document = Document.from_file(
        _resolve(config.semif.rubric),
        repo_root=REPO_ROOT,
        document_key=RUBRIC_KEY,
        kind=DocumentKind.RUBRIC,
        title="SemIf rubric: criteria and option descriptions for the four tasks",
        version=config.rubric.version,
        phase=phase,
        run_id=run_id,
    )
    hypothesis = Hypothesis(
        hypothesis_key="H4",
        kind=HypothesisKind.CONFIRMATORY,
        statement=H4_STATEMENT,
        analysis_plan=analysis_plan(config),
        proposed_by=Proposer.RESEARCHER,
        phase=phase,
        run_id=run_id,
    )
    shown = labeled_set.resolve()
    if shown.is_relative_to(REPO_ROOT):
        shown = shown.relative_to(REPO_ROOT)
    b = config.bakeoff
    package = {
        "package_key": f"{H4_PACKAGE}.{config.rubric.version}.{phase.value}",
        "description": "H4 classifier bake-off: SemIf, Wortega's OpenJev, CryptoBERT, FinBERT "
        "and VADER on the human-labeled test split",
        "code_commit": commit_of(run_id),
        "data": (DataFile(path=str(shown), sha256=_sha256_file(labeled_set)),),
        "model_revision": f"{config.semif.model}@{config.semif.revision}",
        "prompt_sha256": config.prompt_sha256,
        "config": {
            "semif_spec": config.semif_spec,
            "semif_version": config.semif_version,
            "chat_template_sha256": config.chat_template.sha256,
            "baselines": {
                name: getattr(b, name).model_dump(mode="json")
                for name in ("cryptobert", "finbert", "openjev")
            },
            "vader": "vaderSentiment, compound cut-offs 0.05",
            "h4": config.h4.model_dump(mode="json"),
            "fallback": config.fallback.model_dump(mode="json"),
        },
        "phase": phase,
        "run_id": run_id,
    }
    return document, hypothesis, package


def register(
    conn: psycopg.Connection,
    config: ClassifyConfig,
    *,
    labeled_set: Path,
    phase: Phase,
    run_id: str,
) -> list[ledger.Entry]:
    """Append the three records in one transaction (all or none)."""
    document, hypothesis, fields = records(
        config, labeled_set=labeled_set, phase=phase, run_id=run_id
    )
    with conn.transaction():
        doc = ledger.append(conn, document)
        hyp = ledger.append(conn, hypothesis)
        pkg = ledger.append(conn, ExperimentPackage(hypothesis_seq=hyp.seq, **fields))
    return [doc, hyp, pkg]
