"""Classification: the interface every classifier implements, and labels in ``clean.labels``.

``StubClassifier`` (``cryptonlp.stubs``) implements it now; SemIf replaces it in step 6 without
other changes.
"""

from cryptonlp.classify.base import Classifier, PostText
from cryptonlp.classify.store import classify_pending, write_labels

__all__ = ["Classifier", "PostText", "classify_pending", "write_labels"]
