"""Stand-ins with the real interfaces, so the whole system runs before the real parts exist.
Each later step swaps one out (tech/build-plan.md, "Testing along the way")."""

from cryptonlp.stubs.classifier import STUB_VERSION, StubClassifier

__all__ = ["STUB_VERSION", "StubClassifier"]
