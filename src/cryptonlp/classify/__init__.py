"""Classification: the interface every classifier implements, and labels in ``clean.labels``.

``StubClassifier`` (``cryptonlp.stubs``) was the step 3b stand-in; ``SemIfClassifier``
(``cryptonlp.classify.semif``) is the study's classifier from step 6. ``config/classify.toml``
chooses which one the hourly loop uses.
"""

from cryptonlp.classify.base import Classifier, PostText
from cryptonlp.classify.store import classify_pending, write_labels

__all__ = ["Classifier", "PostText", "classify_pending", "write_labels"]
