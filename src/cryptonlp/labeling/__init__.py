"""Step 4: the human-labeled set (sampling, Label Studio, agreement, the time split).

The labeling manual is ``manual.md`` in this package; register it in the ledger before the
first batch is labeled (``cryptonlp ledger register-document``).
"""

from cryptonlp.labeling.config import LabelingConfig, load_labeling_config
from cryptonlp.labeling.sample import PASS1, RELABEL, Batch, DrawError, draw_pass1, draw_relabel
from cryptonlp.labeling.studio import Studio, StudioError

MANUAL = __file__.replace("__init__.py", "manual.md")

__all__ = [
    "MANUAL",
    "PASS1",
    "RELABEL",
    "Batch",
    "DrawError",
    "LabelingConfig",
    "Studio",
    "StudioError",
    "draw_pass1",
    "draw_relabel",
    "load_labeling_config",
]
