# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-model judge panel statistics for collected Tier 3 rewards.

With a judge panel the verifier stores one ``details[metric]["panel"]`` block
per LLM metric in each reward (see ``aggregate_panel`` in
``eval_core.llm_judge``). The collector passes the scoreable reward rows of
every condition whose execution succeeded, and this module derives the
``<agent>/judge_panel.json`` report:

- Per-judge arm means and lift, on the same trials as the headline lift. A
  metric the verifier skipped (no ground truth or expected behaviors) has a
  judge-independent score and no panel block; that score counts for every
  judge, and ``n_skipped`` counts those trials. Otherwise a judge's metric
  mean covers the trials it scored successfully: ``n`` counts them and
  ``failed`` counts the trial metrics it failed.
- Inter-judge agreement per metric, pooled over both arms: Fleiss' kappa on the
  boolean verdicts (accuracy criteria, behaviors, goal ``achieved``) and
  Krippendorff's alpha (interval metric) on member scores.
- Whether the judges' LLM lifts point the same way. A zero lift is compatible
  with either direction; only an up and a down lift together disagree.
- The judges from the agent's own model family: their mean LLM lift against
  the other judges', and the lift re-aggregated without them next to a
  reference that drops as many other-family judges instead. Under vote and
  median a smaller panel can change the lift by itself (two judges tie where
  three form a majority), so that lift is compared with the reference, not
  with the headline lift.
- The trial metrics whose judges disagreed by at least the panel threshold.

Every reward row is one judgment unit for agreement and disagreement. Means
weight each logical trial once: a multi-step fallback trial averages its step
rows first, like the collector's ``_logical_attempt_rewards``. A row with no
detail for a metric, such as a native multi-step trial whose Harbor result
supplies the aggregate reward, keeps no per-judge verdict and counts toward no
panel statistic.

Panel blocks are verifier output, so they are validated and bounded here; a
malformed block is ignored instead of trusted.
"""

from __future__ import annotations

import copy
import functools
import heapq
import itertools
import math
import re
import statistics
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from skillevaluator.tier3.eval_core.llm_judge import (
    _JUDGE_PANEL_CRITERIA,
    _JUDGE_PANEL_PROVIDERS,
    DEFAULT_JUDGE_PANEL_DISAGREEMENT,
    JUDGE_PANEL_AGGREGATIONS,
    JUDGE_PANEL_MAX_MEMBERS,
    _model_family,
    aggregate_panel,
)
from skillevaluator.tier3.harbor.metrics import metric_value
from skillevaluator.utils.redaction import redact_sensitive_text

JUDGE_PANEL_SCHEMA_VERSION = "1.0"
JUDGE_PANEL_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
MAX_DISAGREEMENT_CASES = 50

_CONDITIONS = ("with_skill", "without_skill")
_DETERMINISTIC_METRICS = ("security", "skill_execution", "skill_efficiency")
# One run uses one panel; these bounds only stop malformed rewards from
# inflating the report.
_MAX_JUDGES = 16
# A valid panel (at most five judges) has at most four reference subsets.
_MAX_REFERENCE_SUBSETS = 32
_MAX_BEHAVIOR_POSITIONS = 256
_MAX_IDENTITY_CHARS = 256
_MAX_REASON_CHARS = 512
_MAX_TEXT_SCAN_CHARS = 65_536
_SIGN_TOLERANCE = 1e-9
# Sums of squared deviations below this are float noise around identical scores.
_ZERO_VARIANCE = 1e-12
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class PanelRewardRow:
    """One scoreable reward row: a whole trial, or one step of a multi-step fallback trial."""

    condition: str
    trial_id: str
    entry_id: str
    reward: Mapping[str, Any]
    step: str | None = None


class KappaResult(NamedTuple):
    """Fleiss' kappa with the number of rated items and the mean observed agreement."""

    kappa: float | None
    items: int
    observed_agreement: float | None


class AlphaResult(NamedTuple):
    """Krippendorff's alpha with the number of pairable units."""

    alpha: float | None
    units: int


@dataclass(frozen=True)
class _Member:
    label: str
    provider: str
    model: str
    ok: bool
    score: float | None
    reason: Any
    entry: Mapping[str, Any]


@dataclass(frozen=True)
class _Block:
    aggregation: str
    quorum: int | None
    threshold: float
    disagreement: bool
    members: tuple[_Member, ...]


_Judged = tuple[PanelRewardRow, dict[str, _Block | None]]


class _ArmLift(NamedTuple):
    """Re-aggregated arm means (rounded per metric) with their unrounded LLM and overall means."""

    means: dict[str, dict[str, float | None]]
    llm: dict[str, float | None]
    overall: dict[str, float | None]


def fleiss_kappa(items: Iterable[Sequence[Hashable]]) -> KappaResult:
    """Return Fleiss' kappa for categorical ratings, allowing a different number of raters per item.

    Each item lists the categories its raters chose. An item's observed
    agreement is the share of agreeing rater pairs among its own raters, and
    category proportions pool every rating. Items with fewer than two ratings
    are excluded. Kappa is undefined (``None``) when expected agreement is 1,
    that is when every rating fell into one category; observed agreement is
    still reported.
    """
    agreement_total = 0.0
    item_count = 0
    rating_count = 0
    category_totals: Counter[Hashable] = Counter()
    for ratings in items:
        raters = len(ratings)
        if raters < 2:
            continue
        counts = Counter(ratings)
        agreement_total += sum(count * (count - 1) for count in counts.values()) / (raters * (raters - 1))
        item_count += 1
        rating_count += raters
        category_totals.update(counts)
    if not item_count:
        return KappaResult(None, 0, None)
    observed = agreement_total / item_count
    if len(category_totals) < 2:
        return KappaResult(None, item_count, observed)
    expected = sum((count / rating_count) ** 2 for count in category_totals.values())
    return KappaResult((observed - expected) / (1.0 - expected), item_count, observed)


def krippendorff_alpha_interval(units: Iterable[Sequence[float]]) -> AlphaResult:
    """Return Krippendorff's alpha with the interval metric, allowing missing ratings.

    Each unit lists the values assigned by the raters who rated it; absent
    raters are simply missing. Units with fewer than two values cannot be
    paired and are excluded. With ``n`` pairable values,
    ``alpha = 1 - D_o / D_e`` where ``D_o`` averages squared differences within
    units (each unit weighted by ``1 / (m_u - 1)``) and ``D_e`` averages them
    over every pair of values. Alpha is undefined (``None``) when ``D_e`` is 0.
    """
    pairable: list[Sequence[float]] = [values for values in units if len(values) >= 2]
    if not pairable:
        return AlphaResult(None, 0)
    # Over ordered pairs, sum (a - b)^2 = 2 * m * sum((v - mean)^2); centering keeps it stable.
    within = 0.0
    for values in pairable:
        unit_mean = statistics.fmean(values)
        within += 2 * len(values) * math.fsum((value - unit_mean) ** 2 for value in values) / (len(values) - 1)
    pooled = [value for values in pairable for value in values]
    pooled_mean = statistics.fmean(pooled)
    squared_deviations = math.fsum((value - pooled_mean) ** 2 for value in pooled)
    if squared_deviations <= _ZERO_VARIANCE:
        return AlphaResult(None, len(pairable))
    total = len(pooled)
    return AlphaResult(1.0 - (total - 1) * within / (2 * total * squared_deviations), len(pairable))


def build_judge_panel_report(
    rows: Iterable[PanelRewardRow],
    *,
    arm_scores: Mapping[str, Mapping[str, Any]] | None = None,
    agent_model: str | None = None,
) -> dict[str, Any] | None:
    """Build the ``judge_panel.json`` report, or return ``None`` when no row carries a panel block.

    ``rows`` holds the scoreable rewards of the conditions that may publish
    quality. ``arm_scores`` maps each condition to the collector's metric means,
    whose deterministic metrics complete the overall score of the lift without
    same-family judges.
    """
    all_rows = list(rows)
    trials = {
        condition: _judged_trials([row for row in all_rows if row.condition == condition]) for condition in _CONDITIONS
    }
    blocks = [
        block
        for condition in _CONDITIONS
        for trial in trials[condition]
        for _row, row_blocks in trial
        for block in row_blocks.values()
        if block is not None
    ]
    if not blocks:
        return None

    judges: dict[str, _Member] = {}
    for block in blocks:
        for member in block.members:
            if member.label not in judges and len(judges) < _MAX_JUDGES:
                judges[member.label] = member
    agent_family = _model_family(None, agent_model)
    families = {label: _model_family(member.provider, member.model) for label, member in judges.items()}
    same_family = [label for label, family in families.items() if family != "unknown" and family == agent_family]

    per_judge = _per_judge(trials, judges, families)
    lift_signs = {
        label: _sign(lift) for label, entry in per_judge.items() if (lift := entry["lift"]["llm_overall"]) is not None
    }
    # A flat judge (often a ceiling effect) sees no harm; only up and down together disagree.
    lift_sign_consistent = not {1, -1} <= set(lift_signs.values()) if len(lift_signs) >= 2 else None
    cases, truncated = _disagreement_cases(trials)
    return {
        "schema_version": JUDGE_PANEL_SCHEMA_VERSION,
        "metrics": list(JUDGE_PANEL_METRICS),
        "aggregation": blocks[0].aggregation,
        "quorum": blocks[0].quorum,
        "disagreement_threshold": blocks[0].threshold,
        "judges": [
            {
                "judge": label,
                "provider": member.provider,
                "model": member.model,
                "family": families[label],
                "same_family": label in same_family,
            }
            for label, member in judges.items()
        ],
        "agent_model": agent_model,
        "agent_family": agent_family,
        "per_judge": per_judge,
        "agreement": {metric: _agreement(trials, metric) for metric in JUDGE_PANEL_METRICS},
        "lift_sign_consistent": lift_sign_consistent,
        "lift_signs": lift_signs,
        "same_family_judges": same_family,
        "lift_excluding_same_family": _lift_excluding_same_family(
            trials,
            excluded=same_family,
            remaining=[label for label in judges if label not in same_family],
            arm_scores=arm_scores or {},
        ),
        "same_family_comparison": _same_family_comparison(per_judge, same_family),
        "disagreement_cases": cases,
        "disagreement_cases_truncated": truncated,
    }


def judge_panel_result_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``result.json`` copy of a report: disagreement cases become their count."""
    summary = {key: copy.deepcopy(value) for key, value in report.items() if key != "disagreement_cases"}
    cases = report.get("disagreement_cases")
    summary["disagreement_case_count"] = (len(cases) if isinstance(cases, list) else 0) + int(
        report.get("disagreement_cases_truncated") or 0
    )
    return summary


def panel_behavior_failures(detail: Mapping[str, Any]) -> list[str]:
    """Explain each failed or tied behavior of an aggregated panel ``behavior_check`` detail.

    An aggregated behavior's reason is only the vote tally ("1/3 judges observed
    this behavior"), so each explanation adds the reason of the first successful
    judge that did not observe the behavior: a majority judge for a failed
    behavior, or the failing side of a tie. A tie counts as 0.5, so it is listed
    and marked. Each explanation is bounded to 512 characters.
    """
    panel = detail.get("panel")
    members = panel.get("members") if isinstance(panel, Mapping) else None
    member_results = [
        member["results"]
        for member in (members[:_MAX_JUDGES] if isinstance(members, list) else [])
        if isinstance(member, Mapping) and member.get("status") == "ok" and isinstance(member.get("results"), list)
    ]
    results = detail.get("results")
    explanations: list[str] = []
    for index, result in enumerate(results[:_MAX_BEHAVIOR_POSITIONS] if isinstance(results, list) else []):
        if not isinstance(result, Mapping) or "passed" not in result:
            continue
        passed = result["passed"]
        if passed is not False and passed is not None:
            continue
        tally = result["reason"].strip() if isinstance(result.get("reason"), str) else ""
        if passed is None and tally:
            tally = f"{tally} (tied)"
        rationale = next(
            (
                text
                for entries in member_results
                if index < len(entries)
                and isinstance(entries[index], Mapping)
                and entries[index].get("passed") is False
                and isinstance(reason := entries[index].get("reason"), str)
                and (text := reason.strip())
            ),
            "",
        )
        text = f"{tally}: {rationale}" if tally and rationale else tally or rationale
        if text:
            explanations.append(text[:_MAX_REASON_CHARS])
    return explanations


# ---------------------------------------------------------------------------
# Panel block parsing
# ---------------------------------------------------------------------------


def _judged_trials(rows: list[PanelRewardRow]) -> list[list[_Judged]]:
    """Group rows into logical trials and parse each row's panel blocks.

    A row without a step (Harbor's authoritative or single-step reward) stands
    for its whole trial; otherwise every step row of the trial counts.
    """
    groups: dict[str, list[PanelRewardRow]] = {}
    for row in rows:
        groups.setdefault(row.trial_id, []).append(row)
    trials: list[list[_Judged]] = []
    for group in groups.values():
        authoritative = next((row for row in group if not row.step), None)
        selected = [authoritative] if authoritative is not None else group
        trials.append(
            [(row, {metric: _panel_block(row.reward, metric) for metric in JUDGE_PANEL_METRICS}) for row in selected]
        )
    return trials


def _panel_block(reward: Mapping[str, Any], metric: str) -> _Block | None:
    details = reward.get("details")
    detail = details.get(metric) if isinstance(details, Mapping) else None
    panel = detail.get("panel") if isinstance(detail, Mapping) else None
    if not isinstance(panel, Mapping):
        return None
    aggregation = panel.get("aggregation")
    raw_members = panel.get("members")
    if (
        aggregation not in JUDGE_PANEL_AGGREGATIONS
        or not isinstance(raw_members, list)
        or not 1 <= len(raw_members) <= JUDGE_PANEL_MAX_MEMBERS
    ):
        return None
    members: list[_Member] = []
    for raw_member in raw_members:
        member = _panel_member(raw_member)
        if member is None or any(existing.label == member.label for existing in members):
            return None
        members.append(member)
    quorum = panel.get("quorum")
    threshold = _unit_score(panel.get("disagreement_threshold"))
    return _Block(
        aggregation=str(aggregation),
        quorum=quorum if isinstance(quorum, int) and not isinstance(quorum, bool) and quorum >= 1 else None,
        threshold=DEFAULT_JUDGE_PANEL_DISAGREEMENT if threshold is None else threshold,
        disagreement=panel.get("disagreement") is True,
        members=tuple(members),
    )


def _panel_member(raw: Any) -> _Member | None:
    if not isinstance(raw, Mapping):
        return None
    provider = raw.get("provider")
    model = raw.get("model")
    if provider not in _JUDGE_PANEL_PROVIDERS or not _is_identity(model):
        return None
    identity = {"label": f"{provider}:{model}", "provider": str(provider), "model": model, "entry": raw}
    status = raw.get("status")
    if status == "ok":
        score = _unit_score(raw.get("score"))
        if score is None:
            return None
        return _Member(ok=True, score=score, reason=raw.get("reason"), **identity)
    if status == "error":
        return _Member(ok=False, score=None, reason=raw.get("reason"), **identity)
    return None


def _is_identity(value: Any) -> bool:
    """Accept a model id the verifier could have been configured with, and nothing redaction would rewrite."""
    return isinstance(value, str) and 0 < len(value) <= _MAX_IDENTITY_CHARS and _is_clean_identity(value)


@functools.lru_cache(maxsize=256)
def _is_clean_identity(value: str) -> bool:
    # Cached: a run repeats the same few model ids in every panel block.
    return (
        value.isprintable()
        and not any(character.isspace() for character in value)
        and redact_sensitive_text(value) == value
    )


def _unit_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    # The range check is False for NaN, so it also rejects non-finite values.
    return numeric if 0.0 <= numeric <= 1.0 else None


# ---------------------------------------------------------------------------
# Per-judge means and lift
# ---------------------------------------------------------------------------


def _per_judge(
    trials: Mapping[str, list[list[_Judged]]],
    judges: Mapping[str, _Member],
    families: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    values = {
        label: {condition: {metric: [] for metric in JUDGE_PANEL_METRICS} for condition in _CONDITIONS}
        for label in judges
    }
    judged = {
        label: {condition: dict.fromkeys(JUDGE_PANEL_METRICS, 0) for condition in _CONDITIONS} for label in judges
    }
    skipped = {condition: dict.fromkeys(JUDGE_PANEL_METRICS, 0) for condition in _CONDITIONS}
    failed = {label: dict.fromkeys(_CONDITIONS, 0) for label in judges}
    for condition in _CONDITIONS:
        for trial in trials[condition]:
            for metric in JUDGE_PANEL_METRICS:
                skip_scores = [
                    score
                    for row, blocks in trial
                    if blocks[metric] is None and (score := _skip_score(row, metric)) is not None
                ]
                verdicts: dict[str, list[_Member]] = {}
                for _row, blocks in trial:
                    block = blocks[metric]
                    for member in block.members if block is not None else ():
                        verdicts.setdefault(member.label, []).append(member)
                if not verdicts:
                    # No judge saw this trial metric, so its skip score counts for every judge.
                    if skip_scores:
                        skipped[condition][metric] += 1
                        for label in judges:
                            values[label][condition][metric].append(statistics.fmean(skip_scores))
                    continue
                for label, members in verdicts.items():
                    if label not in judges:
                        continue
                    # A judge that failed any step of the trial has no score for it.
                    if all(member.ok for member in members):
                        judged[label][condition][metric] += 1
                        scores = [member.score for member in members if member.score is not None]
                        values[label][condition][metric].append(statistics.fmean([*scores, *skip_scores]))
                    else:
                        failed[label][condition] += 1

    per_judge: dict[str, dict[str, Any]] = {}
    for label, member in judges.items():
        arms = {condition: _arm_means(values[label][condition]) for condition in _CONDITIONS}
        per_judge[label] = {
            "provider": member.provider,
            "model": member.model,
            "family": families[label],
            "with_skill": arms["with_skill"],
            "without_skill": arms["without_skill"],
            "lift": {
                **{
                    metric: _delta(arms["with_skill"][metric], arms["without_skill"][metric])
                    for metric in JUDGE_PANEL_METRICS
                },
                "llm_overall": _delta(_llm_overall(arms["with_skill"]), _llm_overall(arms["without_skill"])),
            },
            "n": judged[label],
            "n_skipped": {condition: dict(skipped[condition]) for condition in _CONDITIONS},
            "failed": failed[label],
        }
    return per_judge


def _skip_score(row: PanelRewardRow, metric: str) -> float | None:
    """Return the row's judge-independent score when the verifier skipped this metric, else ``None``.

    The verifier's skip result is a ``details[metric]`` mapping without a panel
    block. A row with no detail for the metric, such as a native multi-step
    trial's Harbor aggregate reward, is not a skip: the panel produced that
    score, but no per-judge verdict remains to attribute it.
    """
    details = row.reward.get("details")
    detail = details.get(metric) if isinstance(details, Mapping) else None
    if not isinstance(detail, Mapping) or "panel" in detail:
        return None
    return metric_value(dict(row.reward), metric)


def _arm_means(values: Mapping[str, list[float]]) -> dict[str, float | None]:
    """Round each metric mean like the collector's ``average_metrics`` and add their LLM overall."""
    means: dict[str, float | None] = {
        metric: round(statistics.fmean(values[metric]), 4) if values[metric] else None for metric in JUDGE_PANEL_METRICS
    }
    means["llm_overall"] = _round(_llm_overall(means))
    return means


def _llm_overall(means: Mapping[str, float | None]) -> float | None:
    """Mean of the three LLM metric means, only when all three exist."""
    values = [means.get(metric) for metric in JUDGE_PANEL_METRICS]
    if any(value is None for value in values):
        return None
    return math.fsum(value for value in values if value is not None) / len(values)


def _delta(with_skill: float | None, without_skill: float | None) -> float | None:
    if with_skill is None or without_skill is None:
        return None
    return round(with_skill - without_skill, 4)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _sign(value: float) -> int:
    if value > _SIGN_TOLERANCE:
        return 1
    if value < -_SIGN_TOLERANCE:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def _agreement(trials: Mapping[str, list[list[_Judged]]], metric: str) -> dict[str, Any]:
    items: list[list[bool]] = []
    units: list[list[float]] = []
    for condition in _CONDITIONS:
        for trial in trials[condition]:
            for _row, blocks in trial:
                block = blocks[metric]
                if block is None:
                    continue
                items.extend(_block_verdicts(metric, block))
                units.append([member.score for member in block.members if member.score is not None])
    kappa = fleiss_kappa(items)
    alpha = krippendorff_alpha_interval(units)
    return {
        "fleiss_kappa": _round(kappa.kappa),
        "kappa_items": kappa.items,
        "krippendorff_alpha": _round(alpha.alpha),
        "alpha_units": alpha.units,
        "observed_agreement": _round(kappa.observed_agreement),
    }


def _block_verdicts(metric: str, block: _Block) -> list[list[bool]]:
    """Return one ballot per rated item: each accuracy criterion, each behavior, or the goal verdict."""
    ok_entries = [member.entry for member in block.members if member.ok]
    if metric == "accuracy":
        voters = [criteria for entry in ok_entries if (criteria := _complete_criteria(entry.get("criteria")))]
        return [[voter[key] for voter in voters] for key in _JUDGE_PANEL_CRITERIA] if voters else []
    if metric == "behavior_check":
        behaviors = [passed for entry in ok_entries if (passed := _behavior_verdicts(entry.get("results"))) is not None]
        positions = max((len(passed) for passed in behaviors), default=0)
        return [[passed[index] for passed in behaviors if index < len(passed)] for index in range(positions)]
    ballot = [achieved for entry in ok_entries if isinstance(achieved := entry.get("achieved"), bool)]
    return [ballot] if ballot else []


def _complete_criteria(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping) or set(value) != set(_JUDGE_PANEL_CRITERIA):
        return None
    if not all(isinstance(value[key], bool) for key in _JUDGE_PANEL_CRITERIA):
        return None
    return {key: value[key] for key in _JUDGE_PANEL_CRITERIA}


def _behavior_verdicts(value: Any) -> list[bool] | None:
    if not isinstance(value, list):
        return None
    results = value[:_MAX_BEHAVIOR_POSITIONS]
    if not all(isinstance(item, Mapping) and isinstance(item.get("passed"), bool) for item in results):
        return None
    return [item["passed"] for item in results]


# ---------------------------------------------------------------------------
# Lift without same-family judges
# ---------------------------------------------------------------------------


def _lift_excluding_same_family(
    trials: Mapping[str, list[list[_Judged]]],
    *,
    excluded: list[str],
    remaining: list[str],
    arm_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not excluded:
        return None
    if not remaining:
        return {"excluded": list(excluded), "judges": [], "available": False}

    lift = _reaggregated_lift(trials, frozenset(excluded), arm_scores)
    reference = _reference_lift(trials, size=len(excluded), others=remaining, arm_scores=arm_scores)
    llm_delta = _delta(lift.llm["with_skill"], lift.llm["without_skill"])
    return {
        "excluded": list(excluded),
        "judges": list(remaining),
        "available": True,
        "metrics": {
            metric: {
                "with_skill": lift.means["with_skill"][metric],
                "without_skill": lift.means["without_skill"][metric],
                "delta": _delta(lift.means["with_skill"][metric], lift.means["without_skill"][metric]),
            }
            for metric in JUDGE_PANEL_METRICS
        },
        "llm_overall": {
            "with_skill": _round(lift.llm["with_skill"]),
            "without_skill": _round(lift.llm["without_skill"]),
            "delta": llm_delta,
        },
        "overall": {
            "with_skill": _round(lift.overall["with_skill"]),
            "without_skill": _round(lift.overall["without_skill"]),
            "delta": _delta(lift.overall["with_skill"], lift.overall["without_skill"]),
        },
        "reference": reference,
        # Positive when equal-size panels that keep the same-family judges show a larger lift than this one.
        "same_family_gap": _delta(reference["llm_overall"]["delta"], llm_delta),
    }


def _reference_lift(
    trials: Mapping[str, list[list[_Judged]]],
    *,
    size: int,
    others: list[str],
    arm_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Average the lift re-aggregated without each ``size``-judge subset of the other-family judges.

    Dropping as many other-family judges as there are same-family ones leaves
    a panel of the same size, so this reference and the lift without
    same-family judges are re-voted by equally many judges and differ only in
    which judges were dropped. With fewer other-family judges than same-family
    ones no subset exists and every delta is ``None``.
    """
    subsets = list(itertools.islice(itertools.combinations(others, size), _MAX_REFERENCE_SUBSETS))
    lifts = [_reaggregated_lift(trials, frozenset(subset), arm_scores) for subset in subsets]

    def mean_delta(pairs: Iterable[tuple[float | None, float | None]]) -> float | None:
        deltas = [
            with_skill - without_skill
            for with_skill, without_skill in pairs
            if with_skill is not None and without_skill is not None
        ]
        return _round(statistics.fmean(deltas)) if deltas else None

    return {
        "subsets": len(subsets),
        "metrics": {
            metric: {
                "delta": mean_delta(
                    (lift.means["with_skill"][metric], lift.means["without_skill"][metric]) for lift in lifts
                )
            }
            for metric in JUDGE_PANEL_METRICS
        },
        "llm_overall": {"delta": mean_delta((lift.llm["with_skill"], lift.llm["without_skill"]) for lift in lifts)},
        "overall": {"delta": mean_delta((lift.overall["with_skill"], lift.overall["without_skill"]) for lift in lifts)},
    }


def _same_family_comparison(
    per_judge: Mapping[str, Mapping[str, Any]],
    same_family: list[str],
) -> dict[str, float | None] | None:
    """Compare the same-family judges' mean LLM lift with the other judges' mean, free of re-voting effects.

    ``None`` when either side has no LLM-overall lift: no judge shares the
    agent's family, every judge does, or one side's judges have no lift.
    """

    def mean_lift(labels: Iterable[str]) -> float | None:
        lifts = [lift for label in labels if (lift := per_judge[label]["lift"]["llm_overall"]) is not None]
        return _round(statistics.fmean(lifts)) if lifts else None

    same = mean_lift(same_family)
    others = mean_lift(label for label in per_judge if label not in same_family)
    if same is None or others is None:
        return None
    return {"same_family_mean_llm_lift": same, "other_judges_mean_llm_lift": others, "gap": _delta(same, others)}


def _reaggregated_lift(
    trials: Mapping[str, list[list[_Judged]]],
    excluded: frozenset[str],
    arm_scores: Mapping[str, Mapping[str, Any]],
) -> _ArmLift:
    means = {condition: _reaggregated_means(trials[condition], excluded) for condition in _CONDITIONS}
    return _ArmLift(
        means=means,
        llm={condition: _llm_overall(means[condition]) for condition in _CONDITIONS},
        overall={condition: _overall(arm_scores.get(condition), means[condition]) for condition in _CONDITIONS},
    )


def _reaggregated_means(trials: list[list[_Judged]], excluded: frozenset[str]) -> dict[str, float | None]:
    """Recompute arm means with the ``excluded`` judges removed from every panel.

    A skipped metric keeps its judge-independent score, as in the per-judge
    means, and a row without any detail for the metric counts for no mean.
    """
    values: dict[str, list[float]] = {metric: [] for metric in JUDGE_PANEL_METRICS}
    for trial in trials:
        for metric in JUDGE_PANEL_METRICS:
            row_values = []
            for row, blocks in trial:
                block = blocks[metric]
                value = _skip_score(row, metric) if block is None else _reaggregate(metric, block, excluded)
                if value is not None:
                    row_values.append(value)
            if row_values:
                values[metric].append(statistics.fmean(row_values))
    return {
        metric: round(statistics.fmean(values[metric]), 4) if values[metric] else None for metric in JUDGE_PANEL_METRICS
    }


def _reaggregate(metric: str, block: _Block, excluded: frozenset[str]) -> float | None:
    """Re-run the verifier's aggregation over the members not in ``excluded``, with quorum 1."""
    entries = [
        (member.provider, member.model, member.entry) for member in block.members if member.label not in excluded
    ]
    if not entries:
        return None
    try:
        result = aggregate_panel(
            metric,
            entries,
            aggregation=block.aggregation,
            quorum=1,
            disagreement_threshold=block.threshold,
        )
    except (TypeError, ValueError, statistics.StatisticsError):
        return None
    if str(result.get("status", "")).casefold() == "error":
        return None
    return _unit_score(result.get("score"))


def _overall(arm: Mapping[str, Any] | None, llm_means: Mapping[str, float | None]) -> float | None:
    """Mean of the six default metrics: original deterministic means plus re-aggregated LLM means."""
    deterministic = [metric_value(dict(arm), metric) if arm else None for metric in _DETERMINISTIC_METRICS]
    values = [*deterministic, *(llm_means[metric] for metric in JUDGE_PANEL_METRICS)]
    if any(value is None for value in values):
        return None
    return math.fsum(value for value in values if value is not None) / len(values)


# ---------------------------------------------------------------------------
# Disagreement cases
# ---------------------------------------------------------------------------


def _disagreement_cases(trials: Mapping[str, list[list[_Judged]]]) -> tuple[list[dict[str, Any]], int]:
    """Return the flagged trial metrics with the largest spreads first, capped, plus how many were cut."""
    flagged: list[tuple[float, int, str, PanelRewardRow, str, _Block]] = []
    for condition in _CONDITIONS:
        for trial in trials[condition]:
            for row, blocks in trial:
                for metric in JUDGE_PANEL_METRICS:
                    block = blocks[metric]
                    if block is None or not block.disagreement:
                        continue
                    flagged.append((_spread(block), len(flagged), condition, row, metric, block))
    # Ties keep collection order; reasons are only sanitized for the kept cases.
    kept = heapq.nsmallest(MAX_DISAGREEMENT_CASES, flagged, key=lambda item: (-item[0], item[1]))
    cases = [_case(condition, row, metric, block, spread) for spread, _index, condition, row, metric, block in kept]
    return cases, len(flagged) - len(cases)


def _spread(block: _Block) -> float:
    scores = [member.score for member in block.members if member.score is not None]
    return round(max(scores) - min(scores), 4) if scores else 0.0


def _case(condition: str, row: PanelRewardRow, metric: str, block: _Block, spread: float) -> dict[str, Any]:
    case: dict[str, Any] = {
        "entry_id": _safe_text(row.entry_id, _MAX_IDENTITY_CHARS),
        "trial_id": _safe_text(row.trial_id, _MAX_IDENTITY_CHARS),
    }
    if row.step:
        case["step"] = _safe_text(row.step, _MAX_IDENTITY_CHARS)
    case.update(
        {
            "condition": condition,
            "metric": metric,
            "spread": spread,
            "scores": {member.label: member.score for member in block.members},
            "reasons": {member.label: _safe_text(member.reason, _MAX_REASON_CHARS) for member in block.members},
        }
    )
    return case


def _safe_text(value: Any, max_len: int) -> str:
    """Return redacted, bounded, single-line text; judge reasons quote untrusted agent output."""
    if not isinstance(value, str):
        return ""
    text = redact_sensitive_text(value[:_MAX_TEXT_SCAN_CHARS], max_len=max_len)
    return _CONTROL_CHARACTERS.sub(" ", text).strip()
