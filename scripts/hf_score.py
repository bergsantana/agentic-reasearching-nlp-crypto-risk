"""Bake-off scorer for the transformers baselines: CryptoBERT, FinBERT and Wortega's OpenJev.

Runs inside the vLLM image (Compose service ``hf-scorer``), whose PyTorch supports the Blackwell
card, so the app image never needs PyTorch. Standalone on purpose: only the standard library,
torch and transformers, no ``cryptonlp`` import.

    docker compose run --rm hf-scorer <cryptobert|finbert|openjev> data/bakeoff/<run ID>

Reads ``inputs.jsonl`` ({"item_id", "text"}) and, for OpenJev, ``rubric.json`` from the run
directory; models and pinned revisions from ``config/classify.toml`` ([bakeoff.<model>]).
Writes ``<model>.jsonl`` and ``<model>-meta.json`` there, and nothing else.

- CryptoBERT, FinBERT: the classifier's raw logits per post, with its own label names, as its
  model card runs it (truncated to ``max_tokens``).
- OpenJev 4B v5: for every rubric decision, each option becomes one NLI hypothesis over the post,
  ``The answer to "<question>" is <option>: <description>`` (its ``code/openjev_decide.py``), and
  the output is P(entailment) per option, scored as its ``modeling_openjev.py`` does
  (``Premise: ...\\nHypothesis: ...``, right padding, last-token pooling, softmax over its three
  classes).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import tomllib
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OPENJEV_TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'
ENTAILMENT = "entailment"


def load_config(name: str) -> dict:
    with (ROOT / "config" / "classify.toml").open("rb") as f:
        return tomllib.load(f)["bakeoff"][name]


def read_inputs(run: Path) -> list[dict]:
    rows = [json.loads(line) for line in (run / "inputs.jsonl").read_text().splitlines() if line]
    if not rows:
        sys.exit(f"{run}/inputs.jsonl is empty")
    return rows


def load(cfg: dict, device: str):
    kw = {"revision": cfg["revision"]}
    if cfg.get("subfolder"):
        kw["subfolder"] = cfg["subfolder"]
    tok = AutoTokenizer.from_pretrained(cfg["repo"], **kw)
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" and device == "cuda" else torch.float32
    # Straight onto the GPU: a 4B model staged in RAM would pass Docker's memory cap here.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["repo"], dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True, **kw
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def classify_texts(tok, model, texts: list[str], max_tokens: int, device: str, batch: int):
    """Raw logits per text, in input order (batched by length for speed)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        enc = tok(
            [texts[i] for i in idx],
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.float().cpu().tolist()
        for i, row in zip(idx, logits, strict=True):
            out[i] = row
    return out


@torch.inference_mode()
def openjev_entailment(
    tok, model, pairs: list[tuple[str, str]], max_tokens: int, device: str, batch: int
) -> list[float]:
    """P(entailment) per (premise, hypothesis), as OpenJevCrossEncoder.predict computes it."""
    template = (
        getattr(model.config, "nli_template", None)
        or "Premise: {premise}\nHypothesis: {hypothesis}"
    )
    if getattr(model.config, "openjev_mlp_head", 0):
        sys.exit("this checkpoint has an MLP head; hf_score.py scores only the linear head")
    labels = {v: int(k) for k, v in model.config.id2label.items()}
    ent = labels[ENTAILMENT]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    text_cfg = model.config.get_text_config()
    if text_cfg.pad_token_id is None:
        text_cfg.pad_token_id = tok.pad_token_id
    backbone = getattr(model, model.base_model_prefix)
    texts = [template.format(premise=p.strip(), hypothesis=h.strip()) for p, h in pairs]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[float] = [float("nan")] * len(texts)
    for start in range(0, len(order), batch):
        idx = order[start : start + batch]
        enc = tok(
            [texts[i] for i in idx],
            truncation=True,
            max_length=max_tokens,
            padding=True,
            return_tensors="pt",
        ).to(device)
        h = backbone(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), last]
        probs = torch.softmax(model.score(pooled).float(), -1).cpu()
        for i, row in zip(idx, probs, strict=True):
            out[i] = float(row[ent])
    return out


# Free memory each model needs before it may start, in GiB (GPU, RAM). OpenJev 4B holds 8.5 GiB
# of BF16 weights; 4 Nov 2026, scorers next to the vLLM servers pushed Docker past its RAM cap
# and Postgres into crash recovery, so the scorer refuses instead of trying.
NEEDS_GIB = {"openjev": (10.5, 3.0), "cryptobert": (1.5, 2.0), "finbert": (1.5, 2.0)}


def preflight(name: str, device: str) -> None:
    gpu, ram = NEEDS_GIB[name]
    with open("/proc/meminfo") as f:
        info = dict(line.split(":", 1) for line in f)
    avail = int(info["MemAvailable"].split()[0]) / 2**20
    if avail < ram:
        sys.exit(f"only {avail:.1f} GiB of RAM free, {name} needs {ram}: stop something first")
    if device == "cuda":
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < gpu:
            sys.exit(
                f"only {free:.1f} GiB free on the GPU, {name} needs {gpu}: stop the vLLM servers "
                "first (docker compose --profile gpu stop)"
            )


def give_back(*paths: Path) -> None:
    """The container runs as root (the shared Hugging Face cache is root's); hand the outputs
    to the host user (OWNER_UID/OWNER_GID, set by Compose)."""
    uid, gid = os.environ.get("OWNER_UID"), os.environ.get("OWNER_GID")
    if os.geteuid() == 0 and uid and gid:
        for path in paths:
            os.chown(path, int(uid), int(gid))


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"cryptobert", "finbert", "openjev"}:
        sys.exit(__doc__)
    name, run = sys.argv[1], Path(sys.argv[2])
    if not run.is_absolute():
        run = ROOT / run
    cfg = load_config(name)
    rows = read_inputs(run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preflight(name, device)
    started = time.perf_counter()
    tok, model = load(cfg, device)
    loaded = time.perf_counter()
    texts = [r["text"] for r in rows]
    results: list[dict] = []
    if name == "openjev":
        rubric = json.loads((run / "rubric.json").read_text())
        pairs, keys = [], []
        for r in rows:
            for d in rubric["decisions"]:
                for o in d["options"]:
                    hyp = OPENJEV_TEMPLATE.format(
                        instr=d["question"], label=o["id"], crit=o["description"]
                    )
                    pairs.append((r["text"], hyp))
                    keys.append((r["item_id"], d["id"], o["id"]))
        ent = openjev_entailment(tok, model, pairs, cfg["max_tokens"], device, batch=16)
        grouped: dict[tuple, dict] = {}
        for (item_id, decision, option), p in zip(keys, ent, strict=True):
            g = grouped.setdefault(
                (item_id, decision),
                {"item_id": item_id, "decision": decision, "option_ids": [], "entailment": []},
            )
            g["option_ids"].append(option)
            g["entailment"].append(p)
        results = list(grouped.values())
    else:
        labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        logits = classify_texts(tok, model, texts, cfg["max_tokens"], device, batch=64)
        results = [
            {"item_id": r["item_id"], "labels": labels, "logits": lg}
            for r, lg in zip(rows, logits, strict=True)
        ]
    done = time.perf_counter()
    tmp = run / f".{name}.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in results))
    os.replace(tmp, run / f"{name}.jsonl")
    meta = {
        "model": name,
        "repo": cfg["repo"],
        "subfolder": cfg.get("subfolder"),
        "revision": cfg["revision"],
        "max_tokens": cfg["max_tokens"],
        "posts": len(rows),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "dtype": str(next(model.parameters()).dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "load_seconds": loaded - started,
        "score_seconds": done - loaded,
    }
    (run / f"{name}-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    give_back(run / f"{name}.jsonl", run / f"{name}-meta.json")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
