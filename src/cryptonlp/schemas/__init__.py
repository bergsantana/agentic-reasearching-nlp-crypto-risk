"""Shared schemas. Every component and every stand-in exchanges these, so a stand-in can be
swapped for the real component without other changes."""

from cryptonlp.schemas.common import (
    BAR,
    Asset,
    Origin,
    Phase,
    Record,
    Source,
    floor_to_bar,
)
from cryptonlp.schemas.features import HourlyFeatures
from cryptonlp.schemas.findings import AgentRole, Finding
from cryptonlp.schemas.forecasts import Arm, Forecast, Target
from cryptonlp.schemas.labels import TASK_LABELS, Annotator, Label, Task
from cryptonlp.schemas.market import Candle, CandleSource, Interval, Symbol
from cryptonlp.schemas.posts import NormalizedPost, RawPost

__all__ = [
    "BAR",
    "TASK_LABELS",
    "AgentRole",
    "Annotator",
    "Arm",
    "Asset",
    "Candle",
    "CandleSource",
    "Finding",
    "Forecast",
    "HourlyFeatures",
    "Interval",
    "Label",
    "NormalizedPost",
    "Origin",
    "Phase",
    "RawPost",
    "Record",
    "Source",
    "Symbol",
    "Target",
    "Task",
    "floor_to_bar",
]
