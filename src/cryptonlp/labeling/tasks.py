"""The five labeling tasks as the annotator sees them, and annotations read back as labels.

One definition serves both directions: the Label Studio labeling interface is generated from
it, and annotations are checked against it. The label values are those of
``cryptonlp.schemas.labels``; the manual (``manual.md``) defines each one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import quoteattr

from cryptonlp.schemas.labels import MULTI_LABEL_TASKS, NO_ENTITY, TASK_LABELS, Task


@dataclass(frozen=True)
class Option:
    value: str  # stored (Label Studio's alias)
    shown: str  # what the annotator reads
    hotkey: str


@dataclass(frozen=True)
class TaskSpec:
    task: Task
    question: str
    options: tuple[Option, ...]
    # An option that cannot be chosen with any other (multi-label tasks only).
    exclusive: str | None = None

    @property
    def multiple(self) -> bool:
        return self.task in MULTI_LABEL_TASKS


# In the order the annotator answers them (manual, sections 5.1 to 5.5).
TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        Task.SENTIMENT,
        "1. Sentiment toward the crypto assets, projects or market discussed",
        (
            Option("negative", "Negative", "1"),
            Option("neutral", "Neutral", "2"),
            Option("positive", "Positive", "3"),
        ),
    ),
    TaskSpec(
        Task.TOPIC,
        "2. Topics (one or more)",
        (
            Option("price_talk", "Price talk", "q"),
            Option("regulation", "Regulation", "w"),
            Option("security_incident", "Security incident", "e"),
            Option("protocol_tech", "Protocol or tech", "r"),
            Option("launch_airdrop", "Launch or airdrop", "t"),
            Option("macro", "Macro", "y"),
            Option("other", "Other", "o"),
        ),
        exclusive="other",
    ),
    TaskSpec(
        Task.SCAM,
        "3. Scam and promotion",
        (
            Option("organic", "Organic", "a"),
            Option("promotion", "Promotion", "s"),
            Option("likely_scam", "Likely scam", "d"),
        ),
    ),
    TaskSpec(
        Task.EVENT,
        "4. Event reported",
        (
            Option("none", "None", "z"),
            Option("hack", "Hack", "x"),
            Option("outage", "Outage", "c"),
            Option("listing", "Listing", "v"),
            Option("regulatory_action", "Regulatory action", "b"),
        ),
    ),
    TaskSpec(
        Task.ENTITY,
        "5. Assets referred to (one or more)",
        (
            Option("token:BTC", "Bitcoin", "f"),
            Option("token:ETH", "Ethereum", "g"),
            Option(NO_ENTITY, "Neither", "h"),
        ),
        exclusive=NO_ENTITY,
    ),
)
SPECS = {spec.task: spec for spec in TASKS}

# Not a task: a post the manual says not to label (section 4). Optional, so it costs no click.
PROBLEMS = (
    Option("not_english", "Not English", "8"),
    Option("unreadable", "Unreadable or empty", "9"),
)
UNCERTAIN = Option("uncertain", "Unsure about at least one answer", "0")


def _choices(name: str, options: tuple[Option, ...], *, multiple: bool, required: bool) -> str:
    rows = "\n".join(
        f"    <Choice value={quoteattr(o.shown)} alias={quoteattr(o.value)} "
        f"hotkey={quoteattr(o.hotkey)}/>"
        for o in options
    )
    return (
        f'  <Choices name="{name}" toName="text" choice="{"multiple" if multiple else "single"}" '
        f'showInline="true" required="{str(required).lower()}">\n{rows}\n  </Choices>'
    )


def label_config() -> str:
    """The Label Studio labeling interface: the post text and the five tasks. Nothing else is
    shown (no platform, author, time or model output), so the label rests on the text alone."""
    parts = ["<View>", '  <Text name="text" value="$text"/>']
    for spec in TASKS:
        parts.append(f'  <Header value={quoteattr(spec.question)} size="5"/>')
        parts.append(_choices(spec.task.value, spec.options, multiple=spec.multiple, required=True))
    parts.append('  <Header value="Only if needed" size="6"/>')
    parts.append(_choices("problem", PROBLEMS, multiple=False, required=False))
    parts.append(_choices("uncertain", (UNCERTAIN,), multiple=True, required=False))
    parts.append("</View>")
    return "\n".join(parts) + "\n"


@dataclass(frozen=True)
class ReadAnnotation:
    """An annotation checked against the tasks. ``labels`` is set only for a usable post."""

    labels: dict[str, list[str]] | None
    problem: str | None
    uncertain: bool
    error: str | None


def _canonical(options: tuple[Option, ...]) -> dict[str, str]:
    # Label Studio stores the alias; accept the shown text too, in case a version stores that.
    return {o.value: o.value for o in options} | {o.shown: o.value for o in options}


def read_result(result: list[dict[str, Any]]) -> ReadAnnotation:
    """Read a Label Studio result list into labels, or say why it can't be used."""
    picked: dict[str, list[str]] = {}
    for region in result:
        name = region.get("from_name")
        choices = region.get("value", {}).get("choices")
        if not isinstance(name, str) or not isinstance(choices, list):
            return ReadAnnotation(None, None, False, f"unexpected result region: {region!r}")
        picked.setdefault(name, []).extend(str(c) for c in choices)

    uncertain = bool(picked.pop("uncertain", []))
    problems = [_canonical(PROBLEMS).get(p) for p in picked.pop("problem", [])]
    if problems:
        if len(problems) != 1 or problems[0] is None:
            return ReadAnnotation(None, None, uncertain, f"invalid problem flag: {problems}")
        return ReadAnnotation(None, problems[0], uncertain, None)

    labels: dict[str, list[str]] = {}
    for spec in TASKS:
        raw = picked.pop(spec.task.value, [])
        names = _canonical(spec.options)
        values = [names.get(v) for v in raw]
        if not values:
            return ReadAnnotation(None, None, uncertain, f"no answer for {spec.task.value}")
        if None in values:
            return ReadAnnotation(None, None, uncertain, f"unknown {spec.task.value} value: {raw}")
        clean = sorted(set(v for v in values if v is not None))
        if not spec.multiple and len(clean) != 1:
            return ReadAnnotation(None, None, uncertain, f"{spec.task.value} takes one answer")
        if spec.exclusive in clean and len(clean) > 1:
            return ReadAnnotation(
                None, None, uncertain, f"{spec.task.value}: {spec.exclusive!r} goes alone"
            )
        labels[spec.task.value] = clean
    if picked:
        return ReadAnnotation(None, None, uncertain, f"unknown fields: {sorted(picked)}")
    return ReadAnnotation(labels, None, uncertain, None)


def _check_specs() -> None:
    """The interface must offer exactly the label sets of the schema (import-time check)."""
    for spec in TASKS:
        values = {o.value for o in spec.options}
        if spec.task is Task.ENTITY:
            continue
        if values != TASK_LABELS[spec.task]:
            raise AssertionError(f"{spec.task}: interface {values} != schema")
    keys = [o.hotkey for s in TASKS for o in s.options] + [o.hotkey for o in PROBLEMS]
    if len(keys) + 1 != len({*keys, UNCERTAIN.hotkey}):
        raise AssertionError("hotkeys must be unique")


_check_specs()
