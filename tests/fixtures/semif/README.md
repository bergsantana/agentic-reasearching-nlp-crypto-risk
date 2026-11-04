# SemIf reference rows

`authored144.jsonl` joins two files of the SemIf repository (Lee, 2026;
https://github.com/TheoLeeCJ/SemIf-OpenJev, commit `23cf1f39fc9534fe81437200959b6dfc7106e45a`, MIT licence,
copyright 2026 TheoLeeCJ):

| File in SemIf | SHA-256 |
| --- | --- |
| `benchmarks/data/authored144.jsonl` (the project-authored decisions) | `8162d1c73f925af64453f1ec05ef36d583b3815bf698e60f0d454bd11537e079` |
| `results/raw/predictions/direct-authored144.jsonl` (its direct-logit outputs on Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16) | `7c966285d2bbee3afd04d0bb454ce6a9d88336c7d5968b1fddde99ac7e83c321` |

Each row keeps the decision (`id`, `state`, `question`, `options`, `label`) and, under `reference`,
SemIf's own output for it: the answer-letter logits, the probabilities, the prompt SHA-256 and the
token count. No post text from this study is here.

Two uses:

- `tests/test_semif.py` renders every prompt with our code and checks its SHA-256 against SemIf's, so
  the prompt the study sends is byte for byte the one SemIf builds.
- `cryptonlp classify conformance` scores the rows on the vLLM server and compares the logits and the
  chosen option with SemIf's, so the vLLM readout can be trusted to be SemIf's.
