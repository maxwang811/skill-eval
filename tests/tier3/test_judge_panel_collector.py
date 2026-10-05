# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collector statistics for the cross-model judge panel.

Hand-built rewards carry ``details[metric]["panel"]`` blocks shaped like the
verifier's ``aggregate_panel`` output. The tests pin per-judge arm means and
lift, inter-judge agreement (Fleiss' kappa and Krippendorff's alpha), the lift
sign check, same-family exclusion and its like-for-like reference, disagreement
cases, and the collector artifacts, including the regressions that keep panel
data out of custom metrics and leave every pre-existing artifact unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from skillevaluator.tier3.eval_core.llm_judge import JUDGE_PANEL_AGGREGATIONS, _model_family, aggregate_panel
from skillevaluator.tier3.harbor.collector import GENERATED_AGENT_ARTIFACTS, collect_harbor_results
from skillevaluator.tier3.harbor.judge_panel_stats import (
    MAX_DISAGREEMENT_CASES,
    PanelRewardRow,
    build_judge_panel_report,
    fleiss_kappa,
    judge_panel_result_summary,
    krippendorff_alpha_interval,
)
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRIC_SET, DEFAULT_METRICS, extract_custom_metrics

CRITERIA = ("SKILL_IDENTIFIED", "ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE")
OPENAI = ("openai", "gpt-5.6-sol")
ANTHROPIC = ("anthropic", "claude-opus-5")
NVIDIA = ("nv_build", "nvidia/nemotron-3-super-120b-a12b")
O_LABEL, A_LABEL, N_LABEL = (f"{provider}:{model}" for provider, model in (OPENAI, ANTHROPIC, NVIDIA))
AGENT_MODEL = "nvidia/nvidia/nemotron-3-super-120b-a12b"
LLM_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
WITH_DETERMINISTIC = (1.0, 1.0, 0.8)
WITHOUT_DETERMINISTIC = (1.0, 0.0, 0.5)


# ---------------------------------------------------------------------------
# Hand-built panel fixtures
# ---------------------------------------------------------------------------


def _identity(judge: tuple[str, str]) -> dict[str, str]:
    provider, model = judge
    return {"provider": provider, "model": model, "family": _model_family(provider, model)}


def _ok(judge: tuple[str, str], score: float, *, reason: str = "judged", **fields: Any) -> dict[str, Any]:
    return {**_identity(judge), "status": "ok", "score": score, "reason": reason, **fields}


def _err(judge: tuple[str, str], reason: str = "judge timed out") -> dict[str, Any]:
    return {**_identity(judge), "status": "error", "reason": reason}


def _accuracy(judge: tuple[str, str], *criteria: int, reason: str = "judged") -> dict[str, Any]:
    flags = dict(zip(CRITERIA, (bool(flag) for flag in criteria), strict=True))
    return _ok(judge, round(sum(criteria) / 5, 4), reason=reason, criteria=flags)


def _goal(judge: tuple[str, str], score: float, achieved: bool) -> dict[str, Any]:
    return _ok(judge, score, achieved=achieved, method="custom")


def _behavior(judge: tuple[str, str], *passed: int) -> dict[str, Any]:
    results = [
        {"step": index + 1, "passed": bool(flag), "reason": "observed" if flag else "missing"}
        for index, flag in enumerate(passed)
    ]
    return _ok(judge, round(sum(passed) / len(passed), 4), results=results)


def _panel_detail(
    members: list[dict[str, Any]],
    *,
    aggregation: str = "vote",
    quorum: int = 2,
    threshold: float = 0.4,
    disagreement: bool | None = None,
) -> dict[str, Any]:
    """Return one LLM metric's ``details`` entry with a contract-shaped panel block."""
    ok_scores = [member["score"] for member in members if member["status"] == "ok"]
    spread = round(max(ok_scores) - min(ok_scores), 4) if ok_scores else None
    flagged = (spread is not None and spread >= threshold - 1e-9) if disagreement is None else disagreement
    return {
        "score": round(sum(ok_scores) / len(ok_scores), 4) if ok_scores else None,
        "reason": f"panel {aggregation} ({len(ok_scores)}/{len(members)} judges)",
        "panel": {
            "aggregation": aggregation,
            "quorum": quorum,
            "members": members,
            "spread": spread,
            "agreement": None,
            "disagreement": flagged,
            "disagreement_threshold": threshold,
            "failed_members": len(members) - len(ok_scores),
        },
    }


def _reward(
    entry_id: str,
    *,
    accuracy: list[dict[str, Any]] | None = None,
    goal: list[dict[str, Any]] | None = None,
    behavior: list[dict[str, Any]] | None = None,
    deterministic: tuple[float, float, float] = WITH_DETERMINISTIC,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a scoreable default reward whose LLM metrics carry panel blocks.

    A metric without members is a skipped judge (no ground truth): score 1.0
    and no panel block, exactly like the verifier's skip result.
    """
    details: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for metric, members in (("accuracy", accuracy), ("goal_accuracy", goal), ("behavior_check", behavior)):
        if members is None:
            details[metric] = {"score": 1.0, "reason": "No ground_truth -- skipped"}
        else:
            details[metric] = _panel_detail(members)
        scores[metric] = details[metric]["score"]
    security, skill_execution, skill_efficiency = deterministic
    reward: dict[str, Any] = {
        "entry_id": entry_id,
        "metric_set": DEFAULT_METRIC_SET,
        "security": security,
        "skill_execution": skill_execution,
        "skill_efficiency": skill_efficiency,
        **scores,
        "details": details,
    }
    reward.update(extra or {})
    return reward


def _voted_reward(
    entry_id: str,
    *,
    accuracy: list[dict[str, Any]] | None = None,
    goal: list[dict[str, Any]] | None = None,
    behavior: list[dict[str, Any]] | None = None,
    aggregation: str = "vote",
    deterministic: tuple[float, float, float] = WITH_DETERMINISTIC,
) -> dict[str, Any]:
    """Return a reward whose judged metrics the real ``aggregate_panel`` scored from member entries.

    A metric without members is skipped exactly like the verifier skips it:
    score 1.0 and no panel block.
    """
    skips = {
        "accuracy": {"score": 1.0, "reason": "No ground_truth -- skipped"},
        "goal_accuracy": {"score": 1.0, "reason": "No ground_truth -- skipped"},
        "behavior_check": {"score": 1.0, "reason": "No expected_behavior defined", "results": []},
    }
    details: dict[str, Any] = {}
    for metric, members in (("accuracy", accuracy), ("goal_accuracy", goal), ("behavior_check", behavior)):
        if members is None:
            details[metric] = dict(skips[metric])
            continue
        details[metric] = aggregate_panel(
            metric,
            [(member["provider"], member["model"], member) for member in members],
            aggregation=aggregation,
            expected_count=len(members[0]["results"]) if metric == "behavior_check" else None,
        )
        assert details[metric].get("status") != "error", details[metric]
    security, skill_execution, skill_efficiency = deterministic
    return {
        "entry_id": entry_id,
        "metric_set": DEFAULT_METRIC_SET,
        "security": security,
        "skill_execution": skill_execution,
        "skill_efficiency": skill_efficiency,
        **{metric: detail["score"] for metric, detail in details.items()},
        "details": details,
    }


def _without_panel(reward: dict[str, Any]) -> dict[str, Any]:
    """Return the same reward with every panel block removed and all scores unchanged."""
    stripped = json.loads(json.dumps(reward))
    for detail in stripped["details"].values():
        if isinstance(detail, dict):
            detail.pop("panel", None)
    return stripped


def _rows(condition: str, rewards: list[dict[str, Any]]) -> list[PanelRewardRow]:
    return [
        PanelRewardRow(
            condition=condition,
            trial_id=f"{reward['entry_id']}__attempt",
            entry_id=reward["entry_id"],
            reward=reward,
        )
        for reward in rewards
    ]


def _with_skill_rewards() -> list[dict[str, Any]]:
    return [
        _reward(
            "case-1",
            accuracy=[
                _accuracy(OPENAI, 1, 1, 1, 1, 0),
                _accuracy(ANTHROPIC, 1, 1, 1, 1, 1),
                _accuracy(NVIDIA, 1, 1, 1, 0, 0),
            ],
            goal=[_goal(OPENAI, 1.0, True), _goal(ANTHROPIC, 1.0, True), _goal(NVIDIA, 0.0, False)],
            behavior=[_behavior(OPENAI, 1, 1), _behavior(ANTHROPIC, 1, 0), _behavior(NVIDIA, 0, 1)],
        ),
        _reward(
            "case-2",
            accuracy=[_accuracy(OPENAI, 1, 1, 1, 0, 0), _accuracy(ANTHROPIC, 1, 1, 1, 1, 0), _err(NVIDIA)],
            goal=[_goal(OPENAI, 0.8, True), _goal(ANTHROPIC, 1.0, True), _goal(NVIDIA, 1.0, True)],
            behavior=[_behavior(OPENAI, 1, 0), _behavior(ANTHROPIC, 1, 1), _behavior(NVIDIA, 1, 1)],
        ),
    ]


def _without_skill_rewards() -> list[dict[str, Any]]:
    return [
        _reward(
            "case-1",
            accuracy=[
                _accuracy(OPENAI, 1, 1, 0, 0, 0),
                _accuracy(ANTHROPIC, 1, 1, 1, 0, 0),
                _accuracy(NVIDIA, 1, 1, 1, 1, 0),
            ],
            goal=[_goal(OPENAI, 0.0, False), _goal(ANTHROPIC, 0.0, False), _goal(NVIDIA, 1.0, True)],
            behavior=[_behavior(OPENAI, 0, 0), _behavior(ANTHROPIC, 1, 0), _behavior(NVIDIA, 1, 1)],
            deterministic=WITHOUT_DETERMINISTIC,
        ),
        _reward(
            "case-2",
            accuracy=[
                _accuracy(OPENAI, 1, 0, 0, 0, 0),
                _accuracy(ANTHROPIC, 1, 1, 0, 0, 0),
                _accuracy(NVIDIA, 1, 1, 1, 1, 1),
            ],
            goal=[_goal(OPENAI, 0.0, False), _goal(ANTHROPIC, 0.5, False), _goal(NVIDIA, 1.0, True)],
            behavior=[_behavior(OPENAI, 0, 1), _behavior(ANTHROPIC, 0, 0), _behavior(NVIDIA, 1, 1)],
            deterministic=WITHOUT_DETERMINISTIC,
        ),
    ]


def _arm_scores() -> dict[str, dict[str, float]]:
    """Return the deterministic arm means the collector would publish for the fixtures."""
    return {
        "with_skill": dict(zip(DEFAULT_METRICS[:3], WITH_DETERMINISTIC, strict=True)),
        "without_skill": dict(zip(DEFAULT_METRICS[:3], WITHOUT_DETERMINISTIC, strict=True)),
    }


def _panel_report(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"arm_scores": _arm_scores(), "agent_model": AGENT_MODEL}
    kwargs.update(overrides)
    rows = [*_rows("with_skill", _with_skill_rewards()), *_rows("without_skill", _without_skill_rewards())]
    report = build_judge_panel_report(rows, **kwargs)
    assert report is not None
    return report


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------

# Fleiss (1971): 10 subjects, 14 raters each, 5 categories. Each row counts how
# many raters assigned the subject to categories 1..5.
FLEISS_1971_TABLE = (
    (0, 0, 0, 0, 14),
    (0, 2, 6, 4, 2),
    (0, 0, 3, 5, 6),
    (0, 3, 9, 2, 0),
    (2, 2, 8, 1, 1),
    (7, 7, 0, 0, 0),
    (3, 2, 6, 3, 0),
    (2, 5, 3, 2, 2),
    (6, 5, 2, 1, 0),
    (0, 2, 2, 3, 7),
)


def test_fleiss_kappa_matches_the_fleiss_1971_textbook_table() -> None:
    items = [
        [category for category, count in enumerate(row, start=1) for _ in range(count)] for row in FLEISS_1971_TABLE
    ]

    result = fleiss_kappa(items)

    # P-bar = 0.3780, P_e = 0.2128, kappa = (0.3780 - 0.2128) / (1 - 0.2128) = 0.2099.
    assert result.items == 10
    assert result.observed_agreement == pytest.approx(0.378022, abs=1e-6)
    assert result.kappa == pytest.approx(0.209931, abs=1e-6)


@pytest.mark.parametrize(
    ("items", "expected_kappa", "expected_items", "expected_observed"),
    [
        # Unanimous items in both categories: P-bar = 1, P_e = 0.5^2 + 0.5^2 = 0.5, kappa = 1.
        ([[True, True, True], [False, False, False]], 1.0, 2, 1.0),
        # Variable raters: P_1 = (2*1 + 1*0) / (3*2) = 1/3, P_2 = 0 / (2*1) = 0, P-bar = 1/6.
        # Pooled p_T = 3/5, p_F = 2/5, P_e = 0.52, kappa = (1/6 - 0.52) / 0.48 = -0.736111.
        ([[True, True, False], [True, False]], -0.736111, 2, 1 / 6),
        # Every rating is True: P_e == 1, so kappa is undefined but agreement is still reported.
        ([[True, True], [True, True, True]], None, 2, 1.0),
        # Single-rating items cannot measure agreement and are excluded.
        ([[True], [False], []], None, 0, None),
        ([[True, True], [False]], None, 1, 1.0),
    ],
    ids=("perfect", "variable-raters", "single-category", "no-pairable-items", "single-rater-item-excluded"),
)
def test_fleiss_kappa_hand_computed_boolean_cases(
    items: list[list[bool]],
    expected_kappa: float | None,
    expected_items: int,
    expected_observed: float | None,
) -> None:
    result = fleiss_kappa(items)

    assert result.items == expected_items
    if expected_kappa is None:
        assert result.kappa is None
    else:
        assert result.kappa == pytest.approx(expected_kappa, abs=1e-6)
    if expected_observed is None:
        assert result.observed_agreement is None
    else:
        assert result.observed_agreement == pytest.approx(expected_observed, abs=1e-9)


def test_krippendorff_alpha_interval_hand_computed_with_missing_values() -> None:
    # Units are trials; each lists the scores of the judges that rated it. The
    # fourth unit has one rating and the fifth none (missing data), so only the
    # first three units are pairable: n = 7 values.
    #   values: 1.0 0.8 | 0.2 0.4 | 1.0 1.0 0.6     sum = 5.0, sum of squares = 4.2
    #   all ordered pairs:  sum (a-b)^2 = 2*n*sumsq - 2*sum^2 = 58.8 - 50 = 8.8
    #   within units, the ordered-pair sum divided by (m_u - 1) is
    #     2 x 0.2^2 / 1 = 0.08 for the first,  2 x 0.2^2 / 1 = 0.08 for the second,
    #     and 2 x (0.4^2 + 0.4^2) / 2 = 0.32 for the third unit, 0.48 in total
    #   D_o = 0.48 / n,  D_e = 8.8 / (n * (n - 1))
    #   alpha = 1 - D_o / D_e = 1 - (n - 1) * 0.48 / 8.8 = 1 - 6 * 0.48 / 8.8 = 0.672727
    result = krippendorff_alpha_interval([[1.0, 0.8], [0.2, 0.4], [1.0, 1.0, 0.6], [0.0], []])

    assert result.units == 3
    assert result.alpha == pytest.approx(0.672727, abs=1e-6)


def test_krippendorff_alpha_is_undefined_without_expected_disagreement() -> None:
    # Every pairable value is identical, so D_e = 0 and alpha is undefined.
    undefined = krippendorff_alpha_interval([[1.0, 1.0], [1.0, 1.0, 1.0], [0.0]])
    assert undefined.alpha is None
    assert undefined.units == 2

    empty = krippendorff_alpha_interval([[0.5], []])
    assert empty.alpha is None
    assert empty.units == 0


def test_agreement_pools_criteria_items_and_trial_score_units() -> None:
    # Accuracy kappa items are the 5 criteria of each trial. Trial 2 has two
    # usable raters because nemotron failed:
    #   T1: C1 TTT, C2 TTT, C3 TTF, C4 TTF, C5 FTF;  T2: C1 TT, C2 FT, C3-5 FF
    #   P_i = 1, 1, 1/3, 1/3, 1/3, 1, 0, 1, 1, 1 -> P-bar = 7/10
    #   14 of 25 ratings are True -> P_e = 0.56^2 + 0.44^2 = 0.5072
    #   kappa = (0.7 - 0.5072) / (1 - 0.5072) = 0.391234
    # Accuracy alpha units are the trial scores [0.8, 1.0, 0.4] and [0.2, 0.4]:
    #   n = 5 values, sum = 2.8, sum of squares = 2.0
    #   all ordered pairs: 2*n*sumsq - 2*sum^2 = 20 - 15.68 = 4.32
    #   within: u1 2 * (0.2^2 + 0.4^2 + 0.6^2) / 2 = 0.56;  u2 2 * 0.2^2 / 1 = 0.08;  total 0.64
    #   alpha = 1 - (n - 1) * 0.64 / 4.32 = 0.407407
    rewards = [
        _reward(
            "case-1",
            accuracy=[
                _accuracy(OPENAI, 1, 1, 1, 1, 0),
                _accuracy(ANTHROPIC, 1, 1, 1, 1, 1),
                _accuracy(NVIDIA, 1, 1, 0, 0, 0),
            ],
        ),
        _reward(
            "case-2",
            accuracy=[_accuracy(OPENAI, 1, 0, 0, 0, 0), _accuracy(ANTHROPIC, 1, 1, 0, 0, 0), _err(NVIDIA)],
        ),
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    accuracy = report["agreement"]["accuracy"]
    assert accuracy == {
        "fleiss_kappa": 0.3912,
        "kappa_items": 10,
        "krippendorff_alpha": 0.4074,
        "alpha_units": 2,
        "observed_agreement": 0.7,
    }
    # Skipped metrics carry no panel block and therefore no agreement data.
    for metric in ("goal_accuracy", "behavior_check"):
        assert report["agreement"][metric] == {
            "fleiss_kappa": None,
            "kappa_items": 0,
            "krippendorff_alpha": None,
            "alpha_units": 0,
            "observed_agreement": None,
        }


def test_agreement_uses_goal_achieved_and_behavior_positions() -> None:
    # Goal items are the boolean achieved verdicts: T1 TTF, T2 TT.
    #   P = 1/3, 1 -> P-bar = 2/3; 4 of 5 True -> P_e = 0.68; kappa = (2/3 - 0.68) / 0.32 = -0.041667
    # Behavior items are per-position verdicts: T1 pos1 TTF, pos2 TFT; T2 pos1 TT, pos2 FT.
    #   P = 1/3, 1/3, 1, 0 -> P-bar = 5/12; 7 of 10 True -> P_e = 0.58; kappa = (5/12 - 0.58) / 0.42 = -0.388889
    rewards = [
        _reward(
            "case-1",
            goal=[_goal(OPENAI, 1.0, True), _goal(ANTHROPIC, 0.9, True), _goal(NVIDIA, 0.1, False)],
            behavior=[_behavior(OPENAI, 1, 1), _behavior(ANTHROPIC, 1, 0), _behavior(NVIDIA, 0, 1)],
        ),
        _reward(
            "case-2",
            goal=[_goal(OPENAI, 1.0, True), _goal(ANTHROPIC, 1.0, True), _err(NVIDIA)],
            behavior=[_behavior(OPENAI, 1, 0), _behavior(ANTHROPIC, 1, 1), _err(NVIDIA)],
        ),
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    goal = report["agreement"]["goal_accuracy"]
    assert goal["kappa_items"] == 2
    assert goal["fleiss_kappa"] == pytest.approx(-0.0417, abs=1e-4)
    assert goal["observed_agreement"] == pytest.approx(0.6667, abs=1e-4)
    behavior = report["agreement"]["behavior_check"]
    assert behavior["kappa_items"] == 4
    assert behavior["fleiss_kappa"] == pytest.approx(-0.3889, abs=1e-4)
    assert behavior["observed_agreement"] == pytest.approx(0.4167, abs=1e-4)


# ---------------------------------------------------------------------------
# Per-judge means, lift, and sign consistency
# ---------------------------------------------------------------------------


def test_per_judge_arm_means_counts_failures_and_lift() -> None:
    report = _panel_report()

    assert report["schema_version"] == "1.0"
    assert report["metrics"] == list(LLM_METRICS)
    assert report["aggregation"] == "vote"
    assert report["quorum"] == 2
    assert [judge["judge"] for judge in report["judges"]] == [O_LABEL, A_LABEL, N_LABEL]
    assert list(report["per_judge"]) == [O_LABEL, A_LABEL, N_LABEL]

    openai = report["per_judge"][O_LABEL]
    assert openai["provider"] == "openai"
    assert openai["model"] == "gpt-5.6-sol"
    assert openai["family"] == "openai"
    assert openai["with_skill"] == {
        "accuracy": 0.7,
        "goal_accuracy": 0.9,
        "behavior_check": 0.75,
        "llm_overall": 0.7833,
    }
    assert openai["without_skill"] == {
        "accuracy": 0.3,
        "goal_accuracy": 0.0,
        "behavior_check": 0.25,
        "llm_overall": 0.1833,
    }
    assert openai["lift"] == {"accuracy": 0.4, "goal_accuracy": 0.9, "behavior_check": 0.5, "llm_overall": 0.6}
    assert openai["n"] == {
        "with_skill": dict.fromkeys(LLM_METRICS, 2),
        "without_skill": dict.fromkeys(LLM_METRICS, 2),
    }
    assert openai["n_skipped"] == {
        "with_skill": dict.fromkeys(LLM_METRICS, 0),
        "without_skill": dict.fromkeys(LLM_METRICS, 0),
    }
    assert openai["failed"] == {"with_skill": 0, "without_skill": 0}

    anthropic = report["per_judge"][A_LABEL]
    assert anthropic["with_skill"] == {
        "accuracy": 0.9,
        "goal_accuracy": 1.0,
        "behavior_check": 0.75,
        "llm_overall": 0.8833,
    }
    assert anthropic["without_skill"] == {
        "accuracy": 0.5,
        "goal_accuracy": 0.25,
        "behavior_check": 0.25,
        "llm_overall": 0.3333,
    }
    assert anthropic["lift"] == {"accuracy": 0.4, "goal_accuracy": 0.75, "behavior_check": 0.5, "llm_overall": 0.55}

    # Nemotron failed accuracy on case-2: that trial is excluded from its mean
    # and counted once as a failure for the with-skill arm.
    nemotron = report["per_judge"][N_LABEL]
    assert nemotron["family"] == "nvidia"
    assert nemotron["with_skill"] == {
        "accuracy": 0.6,
        "goal_accuracy": 0.5,
        "behavior_check": 0.75,
        "llm_overall": 0.6167,
    }
    assert nemotron["without_skill"] == {
        "accuracy": 0.9,
        "goal_accuracy": 1.0,
        "behavior_check": 1.0,
        "llm_overall": 0.9667,
    }
    assert nemotron["lift"] == {
        "accuracy": -0.3,
        "goal_accuracy": -0.5,
        "behavior_check": -0.25,
        "llm_overall": -0.35,
    }
    assert nemotron["n"]["with_skill"] == {"accuracy": 1, "goal_accuracy": 2, "behavior_check": 2}
    assert nemotron["failed"] == {"with_skill": 1, "without_skill": 0}


def test_llm_overall_is_missing_only_when_a_judge_failed_a_judged_metric() -> None:
    # Behavior was skipped (no expected behaviors): its judge-independent 1.0
    # counts for every judge. gpt-5.6-sol failed the judged goal metric, so it
    # has no goal mean and therefore no LLM overall.
    rewards = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 0)],
            goal=[_err(OPENAI), _goal(ANTHROPIC, 0.8, True)],
        )
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    openai = report["per_judge"][O_LABEL]
    assert openai["with_skill"] == {
        "accuracy": 1.0,
        "goal_accuracy": None,
        "behavior_check": 1.0,
        "llm_overall": None,
    }
    assert openai["n"]["with_skill"] == {"accuracy": 1, "goal_accuracy": 0, "behavior_check": 0}
    assert openai["n_skipped"]["with_skill"] == {"accuracy": 0, "goal_accuracy": 0, "behavior_check": 1}
    assert openai["failed"]["with_skill"] == 1
    # (0.8 + 0.8 + 1.0) / 3 = 0.8667
    assert report["per_judge"][A_LABEL]["with_skill"] == {
        "accuracy": 0.8,
        "goal_accuracy": 0.8,
        "behavior_check": 1.0,
        "llm_overall": 0.8667,
    }
    # No baseline was collected, so there is no lift and no sign to compare.
    assert openai["without_skill"] == dict.fromkeys((*LLM_METRICS, "llm_overall"))
    assert openai["lift"] == dict.fromkeys((*LLM_METRICS, "llm_overall"))
    assert report["lift_signs"] == {}
    assert report["lift_sign_consistent"] is None


def _split_verdict_rewards(*, behaviors: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return two cases per arm where gpt-5.6-sol and claude-opus-5 see the skill help and nemotron sees it hurt.

    Every judge observes the one expected behavior in both arms; without
    behaviors the verifier skips behavior_check with the same 1.0 score.
    """

    def reward(entry_id: str, *, with_skill: bool) -> dict[str, Any]:
        verdicts = {OPENAI: with_skill, ANTHROPIC: with_skill, NVIDIA: not with_skill}
        return _voted_reward(
            entry_id,
            accuracy=[_accuracy(judge, *([int(yes)] * 5)) for judge, yes in verdicts.items()],
            goal=[_goal(judge, float(yes), yes) for judge, yes in verdicts.items()],
            behavior=[_behavior(judge, 1) for judge in verdicts] if behaviors else None,
            deterministic=WITH_DETERMINISTIC if with_skill else WITHOUT_DETERMINISTIC,
        )

    return (
        [reward(entry_id, with_skill=True) for entry_id in ("case-1", "case-2")],
        [reward(entry_id, with_skill=False) for entry_id in ("case-1", "case-2")],
    )


@pytest.mark.parametrize("behaviors", [False, True], ids=("behavior-skipped-everywhere", "behavior-judged"))
def test_lift_sign_check_runs_when_a_metric_is_skipped_in_every_trial(behaviors: bool) -> None:
    with_rewards, without_rewards = _split_verdict_rewards(behaviors=behaviors)

    report = build_judge_panel_report(
        [*_rows("with_skill", with_rewards), *_rows("without_skill", without_rewards)],
        agent_model=AGENT_MODEL,
    )

    assert report is not None
    # behavior_check is 1.0 in both arms either way, so each LLM lift is
    # (+-1 + +-1 + 0) / 3: the skipped metric no longer hides the sign flip.
    assert {label: entry["lift"]["llm_overall"] for label, entry in report["per_judge"].items()} == {
        O_LABEL: 0.6667,
        A_LABEL: 0.6667,
        N_LABEL: -0.6667,
    }
    assert report["lift_signs"] == {O_LABEL: 1, A_LABEL: 1, N_LABEL: -1}
    assert report["lift_sign_consistent"] is False
    openai = report["per_judge"][O_LABEL]
    assert openai["lift"]["behavior_check"] == 0.0
    assert openai["n"]["with_skill"]["behavior_check"] == (2 if behaviors else 0)
    assert openai["n_skipped"]["with_skill"]["behavior_check"] == (0 if behaviors else 2)


@pytest.mark.parametrize(
    "skipped",
    [("behavior_check",), ("accuracy", "goal_accuracy")],
    ids=("case-2-without-assertions", "case-2-without-ground-truth"),
)
def test_unanimous_panel_per_judge_lift_matches_lift_excluding_same_family_with_skipped_metrics(
    tmp_path: Path,
    skipped: tuple[str, ...],
) -> None:
    def reward(entry_id: str, *, with_skill: bool) -> dict[str, Any]:
        # Every judge returns the same verdicts, so no judge can show self-preference.
        skip = skipped if entry_id == "case-2" else ()
        return _voted_reward(
            entry_id,
            accuracy=None
            if "accuracy" in skip
            else [_accuracy(judge, 1, 1, *([int(with_skill)] * 3)) for judge in (OPENAI, ANTHROPIC, NVIDIA)],
            goal=None
            if "goal_accuracy" in skip
            else [_goal(judge, float(with_skill), with_skill) for judge in (OPENAI, ANTHROPIC, NVIDIA)],
            behavior=None
            if "behavior_check" in skip
            else [_behavior(judge, int(with_skill), int(with_skill)) for judge in (OPENAI, ANTHROPIC, NVIDIA)],
            deterministic=WITH_DETERMINISTIC if with_skill else WITHOUT_DETERMINISTIC,
        )

    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    _write_jobs(
        jobs_dir,
        with_rewards=[reward(entry_id, with_skill=True) for entry_id in ("case-1", "case-2")],
        without_rewards=[reward(entry_id, with_skill=False) for entry_id in ("case-1", "case-2")],
    )

    _collect(jobs_dir, results_dir)

    headline = json.loads((results_dir / "opencode" / "lift.json").read_text(encoding="utf-8"))
    report = json.loads((results_dir / "opencode" / "judge_panel.json").read_text(encoding="utf-8"))
    excluded = report["lift_excluding_same_family"]
    for label, entry in report["per_judge"].items():
        for metric in LLM_METRICS:
            assert entry["lift"][metric] == excluded["metrics"][metric]["delta"] == headline[metric]["delta"], (
                label,
                metric,
            )
            assert entry["n"]["with_skill"][metric] == (1 if metric in skipped else 2)
            assert entry["n_skipped"]["with_skill"][metric] == (1 if metric in skipped else 0)
        assert entry["lift"]["llm_overall"] == excluded["llm_overall"]["delta"], label
    assert report["lift_signs"] == dict.fromkeys((O_LABEL, A_LABEL, N_LABEL), 1)
    assert report["lift_sign_consistent"] is True
    assert excluded["same_family_gap"] == 0.0
    assert report["same_family_comparison"]["gap"] == 0.0


def test_native_root_rows_count_for_no_judge() -> None:
    # A native multi-step trial whose Harbor result supplies the aggregate reward
    # keeps only Harbor's numbers. Its LLM scores came from the panel, so they are
    # not judge-independent, and no per-judge verdict survives to attribute.
    native = {
        "entry_id": "case-2",
        "accuracy": 0.0,
        "goal_accuracy": 0.0,
        "behavior_check": 0.0,
        "details": {"harbor_rewards": {"accuracy": 0.0, "goal_accuracy": 0.0, "behavior_check": 0.0}},
    }
    judged = _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(NVIDIA, 1, 1, 0, 0, 0)])

    report = build_judge_panel_report(_rows("with_skill", [judged, native]), agent_model=AGENT_MODEL)

    assert report is not None
    openai = report["per_judge"][O_LABEL]
    assert openai["with_skill"] == {"accuracy": 1.0, "goal_accuracy": 1.0, "behavior_check": 1.0, "llm_overall": 1.0}
    assert openai["n"]["with_skill"] == {"accuracy": 1, "goal_accuracy": 0, "behavior_check": 0}
    assert openai["n_skipped"]["with_skill"] == {"accuracy": 0, "goal_accuracy": 1, "behavior_check": 1}
    excluded = report["lift_excluding_same_family"]
    assert {metric: excluded["metrics"][metric]["with_skill"] for metric in LLM_METRICS} == dict.fromkeys(
        LLM_METRICS, 1.0
    )


def test_lift_sign_consistency_true_false_and_undefined() -> None:
    flipped = _panel_report()
    assert flipped["lift_signs"] == {O_LABEL: 1, A_LABEL: 1, N_LABEL: -1}
    assert flipped["lift_sign_consistent"] is False

    agreeing_with = [
        _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 0)]),
    ]
    agreeing_without = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 0, 0, 0, 0), _accuracy(ANTHROPIC, 1, 1, 0, 0, 0)],
            deterministic=WITHOUT_DETERMINISTIC,
        ),
    ]
    consistent = build_judge_panel_report(
        [*_rows("with_skill", agreeing_with), *_rows("without_skill", agreeing_without)],
        agent_model=AGENT_MODEL,
    )
    assert consistent is not None
    # Goal and behavior were skipped (score 1.0, no panel): their judge-independent
    # score counts for every judge, so each LLM lift is the accuracy lift / 3.
    assert {label: entry["lift"]["llm_overall"] for label, entry in consistent["per_judge"].items()} == {
        O_LABEL: 0.2667,
        A_LABEL: 0.1333,
    }
    assert consistent["lift_signs"] == {O_LABEL: 1, A_LABEL: 1}
    assert consistent["lift_sign_consistent"] is True

    complete_with = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 0)],
            goal=[_goal(OPENAI, 1.0, True), _goal(ANTHROPIC, 0.8, True)],
            behavior=[_behavior(OPENAI, 1), _behavior(ANTHROPIC, 1)],
        )
    ]
    complete_without = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 0, 0, 0, 0), _accuracy(ANTHROPIC, 1, 1, 0, 0, 0)],
            goal=[_goal(OPENAI, 0.0, False), _goal(ANTHROPIC, 0.2, False)],
            behavior=[_behavior(OPENAI, 0), _behavior(ANTHROPIC, 1)],
            deterministic=WITHOUT_DETERMINISTIC,
        )
    ]
    agreeing = build_judge_panel_report(
        [*_rows("with_skill", complete_with), *_rows("without_skill", complete_without)],
        agent_model=AGENT_MODEL,
    )
    assert agreeing is not None
    assert agreeing["lift_signs"] == {O_LABEL: 1, A_LABEL: 1}
    assert agreeing["lift_sign_consistent"] is True

    single_judge = build_judge_panel_report(
        [
            *_rows("with_skill", [_reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1)])]),
            *_rows("without_skill", [_reward("case-1", accuracy=[_accuracy(OPENAI, 0, 0, 0, 0, 0)])]),
        ],
        agent_model=AGENT_MODEL,
    )
    assert single_judge is not None
    assert single_judge["lift_sign_consistent"] is None


def _signed_lift_rows(signs: dict[tuple[str, str], int]) -> list[PanelRewardRow]:
    """Return one trial per arm where each judge's accuracy lift has the given sign (goal and behavior skipped)."""
    flags = {1: ((1,) * 5, (0,) * 5), 0: ((1,) * 5, (1,) * 5), -1: ((0,) * 5, (1,) * 5)}
    with_reward = _reward("case-1", accuracy=[_accuracy(judge, *flags[sign][0]) for judge, sign in signs.items()])
    without_reward = _reward(
        "case-1",
        accuracy=[_accuracy(judge, *flags[sign][1]) for judge, sign in signs.items()],
        deterministic=WITHOUT_DETERMINISTIC,
    )
    return [*_rows("with_skill", [with_reward]), *_rows("without_skill", [without_reward])]


@pytest.mark.parametrize(
    ("signs", "consistent"),
    [
        ({OPENAI: 0, ANTHROPIC: 1}, True),
        ({OPENAI: 0, ANTHROPIC: -1}, True),
        ({OPENAI: 0, ANTHROPIC: 0}, True),
        ({OPENAI: 1, ANTHROPIC: 1}, True),
        ({OPENAI: 1, ANTHROPIC: -1}, False),
        ({OPENAI: 0, ANTHROPIC: 1, NVIDIA: -1}, False),
    ],
    ids=("flat-up", "flat-down", "flat-flat", "up-up", "up-down", "flat-up-down"),
)
def test_zero_lift_is_compatible_with_either_sign(signs: dict[tuple[str, str], int], consistent: bool) -> None:
    # A flat judge (often a ceiling effect: 1.0 in both arms) sees no harm, so
    # only judges pointing up and down at the same time make the lift inconclusive.
    report = build_judge_panel_report(_signed_lift_rows(signs), agent_model=AGENT_MODEL)

    assert report is not None
    assert report["lift_signs"] == {f"{provider}:{model}": sign for (provider, model), sign in signs.items()}
    assert report["lift_sign_consistent"] is consistent


def test_multistep_fallback_trial_is_weighted_once_per_logical_trial() -> None:
    step_one = _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 0, 0, 0), _accuracy(ANTHROPIC, 1, 1, 0, 0, 0)])
    step_two = _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 1)])
    single = _reward("case-2", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _err(ANTHROPIC)])
    rows = [
        PanelRewardRow("with_skill", "case-1__attempt", "case-1", step_one, step="step-1"),
        PanelRewardRow("with_skill", "case-1__attempt", "case-1", step_two, step="step-2"),
        PanelRewardRow("with_skill", "case-2__attempt", "case-2", single),
    ]

    report = build_judge_panel_report(rows, agent_model=AGENT_MODEL)

    assert report is not None
    # case-1 averages its steps (0.4 + 1.0) / 2 = 0.7 before the arm mean
    # (0.7 + 1.0) / 2 = 0.85; a per-row mean would be 0.8.
    assert report["per_judge"][O_LABEL]["with_skill"]["accuracy"] == 0.85
    assert report["per_judge"][O_LABEL]["n"]["with_skill"]["accuracy"] == 2
    # Each step is its own agreement unit.
    assert report["agreement"]["accuracy"]["alpha_units"] == 2
    assert report["per_judge"][A_LABEL]["with_skill"]["accuracy"] == 0.7
    assert report["per_judge"][A_LABEL]["failed"]["with_skill"] == 1


def test_authoritative_trial_row_supersedes_its_step_rows() -> None:
    # Like the collector's logical rewards, a row without a step stands for the whole trial.
    rows = [
        PanelRewardRow(
            "with_skill",
            "case-1__attempt",
            "case-1",
            _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 1)]),
        ),
        PanelRewardRow(
            "with_skill",
            "case-1__attempt",
            "case-1",
            _reward("case-1", accuracy=[_accuracy(OPENAI, 0, 0, 0, 0, 0), _accuracy(ANTHROPIC, 0, 0, 0, 0, 0)]),
            step="step-1",
        ),
    ]

    report = build_judge_panel_report(rows, agent_model=AGENT_MODEL)

    assert report is not None
    assert report["per_judge"][O_LABEL]["with_skill"]["accuracy"] == 1.0
    assert report["agreement"]["accuracy"]["alpha_units"] == 1


def test_judge_failing_any_step_fails_the_logical_trial() -> None:
    rows = [
        PanelRewardRow(
            "with_skill",
            "case-1__attempt",
            "case-1",
            _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 1)]),
            step="step-1",
        ),
        PanelRewardRow(
            "with_skill",
            "case-1__attempt",
            "case-1",
            _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _err(ANTHROPIC)]),
            step="step-2",
        ),
    ]

    report = build_judge_panel_report(rows, agent_model=AGENT_MODEL)

    assert report is not None
    assert report["per_judge"][A_LABEL]["with_skill"]["accuracy"] is None
    assert report["per_judge"][A_LABEL]["n"]["with_skill"]["accuracy"] == 0
    assert report["per_judge"][A_LABEL]["failed"]["with_skill"] == 1


# ---------------------------------------------------------------------------
# Same-family judges
# ---------------------------------------------------------------------------


def test_same_family_judge_is_detected_from_the_model_id() -> None:
    report = _panel_report()

    assert report["agent_model"] == AGENT_MODEL
    assert report["agent_family"] == "nvidia"
    assert report["same_family_judges"] == [N_LABEL]
    assert report["judges"] == [
        {"judge": O_LABEL, "provider": "openai", "model": "gpt-5.6-sol", "family": "openai", "same_family": False},
        {
            "judge": A_LABEL,
            "provider": "anthropic",
            "model": "claude-opus-5",
            "family": "anthropic",
            "same_family": False,
        },
        {
            "judge": N_LABEL,
            "provider": "nv_build",
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "family": "nvidia",
            "same_family": True,
        },
    ]


@pytest.mark.parametrize("agent_model", [None, "", "mystery-model", "acme/house-model-7b"])
def test_unknown_family_never_counts_as_same_family(agent_model: str | None) -> None:
    gateway = ("openai-compatible", "acme/house-model-7b")
    rewards = [
        _reward("case-1", accuracy=[_accuracy(gateway, 1, 1, 1, 1, 1), _accuracy(OPENAI, 1, 1, 1, 1, 0)]),
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=agent_model)

    assert report is not None
    assert report["agent_family"] == "unknown"
    assert report["judges"][0]["family"] == "unknown"
    assert report["same_family_judges"] == []
    assert all(judge["same_family"] is False for judge in report["judges"])
    assert report["lift_excluding_same_family"] is None
    assert report["same_family_comparison"] is None


def test_lift_excluding_same_family_reaggregates_the_remaining_judges() -> None:
    report = _panel_report()

    # Without nemotron the two remaining judges are re-voted with quorum 1. A
    # criterion or behavior they split counts as 0.5, and a goal majority takes
    # the median score of its side. Per trial (case-1, case-2):
    #   with:    accuracy 0.9, 0.7 -> 0.8;  goal 1.0, 0.9 -> 0.95;  behavior 0.75, 0.75 -> 0.75
    #   without: accuracy 0.5, 0.3 -> 0.4;  goal 0.0, 0.25 -> 0.125;  behavior 0.25, 0.25 -> 0.25
    #   llm_overall with = 2.5 / 3 = 0.8333, without = 0.775 / 3 = 0.2583, delta 0.575
    #   overall with    = (1.0 + 1.0 + 0.8 + 0.8 + 0.95 + 0.75) / 6 = 0.8833
    #   overall without = (1.0 + 0.0 + 0.5 + 0.4 + 0.125 + 0.25) / 6 = 0.3792
    # The reference drops one other-family judge instead, keeping a two-judge panel:
    #   without gpt-5.6-sol (claude-opus-5 + nemotron), per arm with / without:
    #     accuracy 0.8 / 0.7;  goal 0.75 / 0.5;  behavior 0.75 / 0.625
    #     llm_overall 0.7667 - 0.6083 = 0.1583;  overall 0.85 - 0.5542 = 0.2958
    #   without claude-opus-5 (gpt-5.6-sol + nemotron):
    #     accuracy 0.65 / 0.6;  goal 0.7 / 0.5;  behavior 0.75 / 0.625
    #     llm_overall 0.7 - 0.575 = 0.125;  overall 0.8167 - 0.5375 = 0.2792
    #   averaged: accuracy 0.075, goal 0.225, behavior 0.125, llm_overall 0.1417, overall 0.2875
    #   same_family_gap = 0.1417 - 0.575 = -0.4333: nemotron lowers the lift
    assert report["lift_excluding_same_family"] == {
        "excluded": [N_LABEL],
        "judges": [O_LABEL, A_LABEL],
        "available": True,
        "metrics": {
            "accuracy": {"with_skill": 0.8, "without_skill": 0.4, "delta": 0.4},
            "goal_accuracy": {"with_skill": 0.95, "without_skill": 0.125, "delta": 0.825},
            "behavior_check": {"with_skill": 0.75, "without_skill": 0.25, "delta": 0.5},
        },
        "llm_overall": {"with_skill": 0.8333, "without_skill": 0.2583, "delta": 0.575},
        "overall": {"with_skill": 0.8833, "without_skill": 0.3792, "delta": 0.5042},
        "reference": {
            "subsets": 2,
            "metrics": {
                "accuracy": {"delta": 0.075},
                "goal_accuracy": {"delta": 0.225},
                "behavior_check": {"delta": 0.125},
            },
            "llm_overall": {"delta": 0.1417},
            "overall": {"delta": 0.2875},
        },
        "same_family_gap": -0.4333,
    }
    # Per-judge LLM lifts: nemotron -0.35 against (0.6 + 0.55) / 2 = 0.575.
    assert report["same_family_comparison"] == {
        "same_family_mean_llm_lift": -0.35,
        "other_judges_mean_llm_lift": 0.575,
        "gap": -0.925,
    }


# Each tuple is one with-skill trial's verdicts by (gpt-5.6-sol, claude-opus-5, nemotron).
BALANCED_DISSENT = ((False, True, True), (True, False, True), (True, True, False))
NEMOTRON_NEVER_DISSENTS = ((False, True, True), (True, False, True), (True, True, True))


def _dissent_report(with_votes: tuple[tuple[bool, ...], ...], *, aggregation: str, agent_model: str) -> dict[str, Any]:
    """Score three trials per arm; every judge says no on every item of the baseline."""

    def reward(entry_id: str, votes: tuple[bool, ...], deterministic: tuple[float, float, float]) -> dict[str, Any]:
        judges = dict(zip((OPENAI, ANTHROPIC, NVIDIA), votes, strict=True))
        return _voted_reward(
            entry_id,
            accuracy=[_accuracy(judge, *([int(yes)] * 5)) for judge, yes in judges.items()],
            goal=[_goal(judge, float(yes), yes) for judge, yes in judges.items()],
            behavior=[_behavior(judge, int(yes)) for judge, yes in judges.items()],
            aggregation=aggregation,
            deterministic=deterministic,
        )

    with_rewards = [reward(f"case-{index}", votes, WITH_DETERMINISTIC) for index, votes in enumerate(with_votes, 1)]
    without_rewards = [
        reward(f"case-{index}", (False, False, False), WITHOUT_DETERMINISTIC) for index in range(1, len(with_votes) + 1)
    ]
    rows = [*_rows("with_skill", with_rewards), *_rows("without_skill", without_rewards)]
    report = build_judge_panel_report(rows, arm_scores=_arm_scores(), agent_model=agent_model)
    assert report is not None
    headline = sum(
        sum(reward[metric] for reward in with_rewards) / len(with_rewards)
        - sum(reward[metric] for reward in without_rewards) / len(without_rewards)
        for metric in LLM_METRICS
    )
    report["_headline_llm_lift"] = round(headline / len(LLM_METRICS), 4)
    return report


@pytest.mark.parametrize("aggregation", JUDGE_PANEL_AGGREGATIONS)
@pytest.mark.parametrize(
    ("agent_model", "same_family"),
    [(AGENT_MODEL, N_LABEL), ("gpt-5.6-sol", O_LABEL), ("claude-opus-5", A_LABEL)],
    ids=("nemotron-agent", "placebo-openai-agent", "placebo-anthropic-agent"),
)
def test_same_family_comparison_is_neutral_for_exchangeable_judges(
    aggregation: str,
    agent_model: str,
    same_family: str,
) -> None:
    # Zero self-preference: each judge dissents in exactly one with-skill trial.
    report = _dissent_report(BALANCED_DISSENT, aggregation=aggregation, agent_model=agent_model)

    assert report["same_family_judges"] == [same_family]
    assert {entry["lift"]["llm_overall"] for entry in report["per_judge"].values()} == {0.6667}
    assert report["same_family_comparison"] == {
        "same_family_mean_llm_lift": 0.6667,
        "other_judges_mean_llm_lift": 0.6667,
        "gap": 0.0,
    }
    excluded = report["lift_excluding_same_family"]
    # A two-judge re-vote ties where three judges formed a majority, so under vote
    # and median the excluded lift falls below the headline with no self-preference
    # at all. The reference drops one other-family judge instead and falls just as far.
    assert report["_headline_llm_lift"] == (0.6667 if aggregation == "mean" else 1.0)
    assert excluded["llm_overall"]["delta"] == 0.6667
    assert excluded["reference"]["subsets"] == 2
    assert excluded["reference"]["llm_overall"]["delta"] == 0.6667
    assert excluded["reference"]["overall"]["delta"] == excluded["overall"]["delta"]
    assert excluded["same_family_gap"] == 0.0


@pytest.mark.parametrize("aggregation", JUDGE_PANEL_AGGREGATIONS)
def test_same_family_comparison_flags_a_biased_same_family_judge(aggregation: str) -> None:
    # Nemotron never dissents in the with-skill arm, while the other judges each dissent once.
    report = _dissent_report(NEMOTRON_NEVER_DISSENTS, aggregation=aggregation, agent_model=AGENT_MODEL)

    # Per-judge LLM lifts: nemotron 1.0 against 0.6667 for the others.
    assert report["same_family_comparison"] == {
        "same_family_mean_llm_lift": 1.0,
        "other_judges_mean_llm_lift": 0.6667,
        "gap": 0.3333,
    }
    excluded = report["lift_excluding_same_family"]
    # Without nemotron: (0.5 + 0.5 + 1.0) / 3; without either other judge instead:
    # (1.0 + 0.5 + 1.0) / 3 or (0.5 + 1.0 + 1.0) / 3, both 0.8333.
    assert excluded["llm_overall"]["delta"] == 0.6667
    assert excluded["reference"]["llm_overall"]["delta"] == 0.8333
    assert excluded["same_family_gap"] == 0.1666


@pytest.mark.parametrize(
    ("with_accuracy", "without_accuracy"),
    [
        pytest.param(
            [_accuracy(NVIDIA, 1, 1, 1, 1, 1), _accuracy(("nv_build", "nemotron-mini"), 1, 1, 1, 0, 0)],
            [_accuracy(NVIDIA, 0, 0, 0, 0, 0), _accuracy(("nv_build", "nemotron-mini"), 0, 0, 0, 0, 0)],
            id="every-judge-same-family",
        ),
        pytest.param(
            [_accuracy(NVIDIA, 1, 1, 1, 1, 1), _err(OPENAI)],
            [_accuracy(NVIDIA, 0, 0, 0, 0, 0), _accuracy(OPENAI, 0, 0, 0, 0, 0)],
            id="other-judge-without-a-lift",
        ),
    ],
)
def test_same_family_comparison_needs_a_lift_on_both_sides(
    with_accuracy: list[dict[str, Any]], without_accuracy: list[dict[str, Any]]
) -> None:
    # Goal and behavior are skipped; nemotron's accuracy lift of 1.0 gives it an LLM lift of 0.3333.
    rows = [
        *_rows("with_skill", [_reward("case-1", accuracy=with_accuracy)]),
        *_rows("without_skill", [_reward("case-1", accuracy=without_accuracy, deterministic=WITHOUT_DETERMINISTIC)]),
    ]

    report = build_judge_panel_report(rows, arm_scores=_arm_scores(), agent_model=AGENT_MODEL)

    assert report is not None
    assert N_LABEL in report["same_family_judges"]
    assert report["per_judge"][N_LABEL]["lift"]["llm_overall"] == 0.3333
    # The other side is empty: every judge shares the family, or the other judge failed every trial.
    assert report["same_family_comparison"] is None


def test_reference_needs_as_many_other_family_judges_as_same_family_ones() -> None:
    # Two same-family judges and one other: no two-judge subset of the others exists.
    rewards = [
        _reward(
            "case-1",
            accuracy=[
                _accuracy(NVIDIA, 1, 1, 1, 1, 1),
                _accuracy(("nv_build", "nemotron-mini"), 1, 1, 0, 0, 0),
                _accuracy(OPENAI, 1, 1, 1, 0, 0),
            ],
        ),
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    excluded = report["lift_excluding_same_family"]
    assert excluded["available"] is True
    assert excluded["metrics"]["accuracy"]["with_skill"] == 0.6
    assert excluded["reference"] == {
        "subsets": 0,
        "metrics": {metric: {"delta": None} for metric in LLM_METRICS},
        "llm_overall": {"delta": None},
        "overall": {"delta": None},
    }
    assert excluded["same_family_gap"] is None
    # No baseline, so neither side has a lift to compare either.
    assert report["same_family_comparison"] is None


def test_lift_excluding_same_family_keeps_judge_independent_skip_scores() -> None:
    # Goal and behavior were skipped (score 1.0, no panel). Their trial values do
    # not depend on any judge, so the re-aggregated means keep them.
    with_rows = [
        _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(NVIDIA, 0, 0, 0, 0, 0)]),
    ]
    without_rows = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 1, 0, 0, 0), _accuracy(NVIDIA, 1, 1, 1, 1, 1)],
            deterministic=WITHOUT_DETERMINISTIC,
        ),
    ]

    report = build_judge_panel_report(
        [*_rows("with_skill", with_rows), *_rows("without_skill", without_rows)],
        arm_scores=_arm_scores(),
        agent_model=AGENT_MODEL,
    )

    assert report is not None
    excluded = report["lift_excluding_same_family"]
    assert excluded["metrics"]["accuracy"] == {"with_skill": 1.0, "without_skill": 0.4, "delta": 0.6}
    assert excluded["metrics"]["goal_accuracy"] == {"with_skill": 1.0, "without_skill": 1.0, "delta": 0.0}
    assert excluded["llm_overall"] == {"with_skill": 1.0, "without_skill": 0.8, "delta": 0.2}


def test_lift_excluding_same_family_uses_quorum_one_for_the_remaining_judges() -> None:
    # With nemotron excluded only gpt-5.6-sol succeeded on case-1. The original
    # quorum of 2 would leave no score; quorum 1 keeps the remaining verdict.
    with_rows = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 1, 1, 0, 0), _err(ANTHROPIC), _accuracy(NVIDIA, 1, 1, 1, 1, 1)],
        ),
        _reward(
            "case-2",
            accuracy=[
                _accuracy(OPENAI, 1, 1, 1, 1, 1),
                _accuracy(ANTHROPIC, 1, 1, 1, 1, 1),
                _accuracy(NVIDIA, 1, 1, 1, 1, 1),
            ],
        ),
    ]

    report = build_judge_panel_report(_rows("with_skill", with_rows), agent_model=AGENT_MODEL)

    assert report is not None
    accuracy = report["lift_excluding_same_family"]["metrics"]["accuracy"]
    assert accuracy == {"with_skill": 0.8, "without_skill": None, "delta": None}


def test_lift_excluding_same_family_drops_trials_without_a_remaining_verdict() -> None:
    with_rows = [
        _reward("case-1", accuracy=[_err(OPENAI), _accuracy(NVIDIA, 1, 1, 1, 1, 1)]),
        _reward("case-2", accuracy=[_accuracy(OPENAI, 1, 1, 0, 0, 0), _accuracy(NVIDIA, 1, 1, 1, 1, 1)]),
    ]

    report = build_judge_panel_report(_rows("with_skill", with_rows), agent_model=AGENT_MODEL)

    assert report is not None
    assert report["lift_excluding_same_family"]["metrics"]["accuracy"]["with_skill"] == 0.4


def test_lift_excluding_same_family_when_every_judge_shares_the_agent_family() -> None:
    rewards = [
        _reward(
            "case-1",
            accuracy=[_accuracy(NVIDIA, 1, 1, 1, 1, 1), _accuracy(("nv_build", "nemotron-mini"), 1, 1, 0, 0, 0)],
        ),
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    assert report["same_family_judges"] == [N_LABEL, "nv_build:nemotron-mini"]
    assert report["lift_excluding_same_family"] == {
        "excluded": [N_LABEL, "nv_build:nemotron-mini"],
        "judges": [],
        "available": False,
    }


def test_lift_excluding_same_family_is_none_without_same_family_judges() -> None:
    report = _panel_report(agent_model="openai-compatible/acme-agent")

    assert report["agent_family"] == "unknown"
    assert report["same_family_judges"] == []
    assert report["lift_excluding_same_family"] is None
    assert report["same_family_comparison"] is None


# ---------------------------------------------------------------------------
# Disagreement cases
# ---------------------------------------------------------------------------


def test_disagreement_cases_list_each_judge_score_and_reason() -> None:
    report = _panel_report()

    # Ten (trial, metric) blocks have a spread of at least 0.4; the largest
    # spreads come first and ties keep collection order.
    cases = report["disagreement_cases"]
    assert report["disagreement_cases_truncated"] == 0
    assert [(case["condition"], case["entry_id"], case["metric"], case["spread"]) for case in cases] == [
        ("with_skill", "case-1", "goal_accuracy", 1.0),
        ("without_skill", "case-1", "goal_accuracy", 1.0),
        ("without_skill", "case-1", "behavior_check", 1.0),
        ("without_skill", "case-2", "goal_accuracy", 1.0),
        ("without_skill", "case-2", "behavior_check", 1.0),
        ("without_skill", "case-2", "accuracy", 0.8),
        ("with_skill", "case-1", "behavior_check", 0.5),
        ("with_skill", "case-2", "behavior_check", 0.5),
        ("with_skill", "case-1", "accuracy", 0.4),
        ("without_skill", "case-1", "accuracy", 0.4),
    ]
    first = cases[0]
    assert first["trial_id"] == "case-1__attempt"
    assert "step" not in first
    assert first["scores"] == {O_LABEL: 1.0, A_LABEL: 1.0, N_LABEL: 0.0}
    assert first["reasons"] == {O_LABEL: "judged", A_LABEL: "judged", N_LABEL: "judged"}


def test_disagreement_cases_are_capped_with_a_truncation_count_and_bounded_reasons() -> None:
    marker = "synthetic-panel-secret-123456"
    long_reason = f"Authorization: Bearer {marker}\x00\x1b " + "the judge disagreed " * 400
    rewards = [
        _reward(
            f"case-{index:03d}",
            accuracy=[
                _accuracy(OPENAI, 1, 1, 1, 1, 1, reason=long_reason),
                _accuracy(ANTHROPIC, 0, 0, 0, 0, 0, reason="nothing matched"),
            ],
        )
        for index in range(MAX_DISAGREEMENT_CASES + 10)
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    assert MAX_DISAGREEMENT_CASES == 50
    assert len(report["disagreement_cases"]) == 50
    assert report["disagreement_cases_truncated"] == 10
    for case in report["disagreement_cases"]:
        reason = case["reasons"][O_LABEL]
        assert len(reason) <= 512
        assert marker not in reason
        assert not any(ord(character) < 32 or ord(character) == 127 for character in reason)
        assert case["reasons"][A_LABEL] == "nothing matched"


def test_failed_member_appears_in_a_disagreement_case_with_its_error() -> None:
    rewards = [
        _reward(
            "case-1",
            accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 0, 0, 0, 0, 0), _err(NVIDIA, "HTTP 503")],
        )
    ]

    report = build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL)

    assert report is not None
    [case] = report["disagreement_cases"]
    assert case["scores"] == {O_LABEL: 1.0, A_LABEL: 0.0, N_LABEL: None}
    assert case["reasons"][N_LABEL] == "HTTP 503"


# ---------------------------------------------------------------------------
# Input validation and summaries
# ---------------------------------------------------------------------------


def test_rewards_without_panel_blocks_produce_no_report() -> None:
    rewards = [_without_panel(reward) for reward in _with_skill_rewards()]

    assert build_judge_panel_report(_rows("with_skill", rewards), agent_model=AGENT_MODEL) is None
    assert build_judge_panel_report([], agent_model=AGENT_MODEL) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda panel: panel.update(members="openai:gpt"),
        lambda panel: panel.update(members=[]),
        lambda panel: panel.update(aggregation="majority"),
        lambda panel: panel["members"][0].update(score=float("nan")),
        lambda panel: panel["members"][0].update(score=1.5),
        lambda panel: panel["members"][0].update(score=True),
        lambda panel: panel["members"][0].update(status="pending"),
        lambda panel: panel["members"][0].update(model="gpt 5"),
        lambda panel: panel["members"][0].update(model=""),
        lambda panel: panel["members"][0].update(provider=7),
        lambda panel: panel["members"][1].update(provider="openai", model="gpt-5.6-sol"),
        lambda panel: panel.update(members=[dict(panel["members"][0]) for _ in range(6)]),
    ],
    ids=(
        "members-not-list",
        "no-members",
        "unknown-aggregation",
        "nan-score",
        "score-out-of-range",
        "bool-score",
        "unknown-status",
        "model-whitespace",
        "empty-model",
        "provider-not-string",
        "duplicate-member",
        "too-many-members",
    ),
)
def test_malformed_panel_blocks_are_ignored(mutate) -> None:
    reward = _reward("case-1", accuracy=[_accuracy(OPENAI, 1, 1, 1, 1, 1), _accuracy(ANTHROPIC, 1, 1, 1, 1, 0)])
    mutate(reward["details"]["accuracy"]["panel"])

    assert build_judge_panel_report(_rows("with_skill", [reward]), agent_model=AGENT_MODEL) is None


def test_result_summary_replaces_cases_with_their_count() -> None:
    report = _panel_report()

    summary = judge_panel_result_summary(report)

    assert "disagreement_cases" not in summary
    assert summary["disagreement_case_count"] == 10
    assert summary["disagreement_cases_truncated"] == 0
    assert {key: value for key, value in summary.items() if key != "disagreement_case_count"} == {
        key: value for key, value in report.items() if key != "disagreement_cases"
    }
    assert "disagreement_cases" in report


def test_panel_blocks_never_become_custom_metrics() -> None:
    for reward in [*_with_skill_rewards(), *_without_skill_rewards()]:
        assert extract_custom_metrics(reward) == {}
        assert extract_custom_metrics({**reward, "domain_quality": 0.4}) == {"domain_quality": 0.4}


# ---------------------------------------------------------------------------
# Collector wiring on fake Harbor job directories
# ---------------------------------------------------------------------------


def _write_complete_job_result(job_dir: Path, trial_names: list[str]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": len(trial_names),
                "stats": {
                    "n_completed_trials": len(trial_names),
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "agent__model___harbor-tasks": {
                            "n_trials": len(trial_names),
                            "n_errors": 0,
                            "reward_stats": {"reward": {"0.5": trial_names}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_verifier_outputs(job_dir: Path, trial_name: str, reward: dict[str, Any]) -> None:
    """Write the verifier's numeric reward.json and rich sidecar like templates/eval.py."""
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    numeric = {
        key: value for key, value in reward.items() if isinstance(value, int | float) and not isinstance(value, bool)
    }
    metric_scores = [reward.get(metric) for metric in DEFAULT_METRICS]
    # A failed judge leaves its metric out and the verifier writes overall 0.0.
    numeric["overall"] = 0.0 if None in metric_scores else round(sum(metric_scores) / len(metric_scores), 4)
    (verifier / "reward.json").write_text(json.dumps(numeric), encoding="utf-8")
    (verifier / "skill_evaluator_reward.json").write_text(json.dumps(reward), encoding="utf-8")


def _write_jobs(
    jobs_dir: Path,
    *,
    with_rewards: list[dict[str, Any]],
    without_rewards: list[dict[str, Any]] | None,
) -> None:
    for variant, rewards in (("with", with_rewards), ("without", without_rewards)):
        if rewards is None:
            continue
        job_dir = jobs_dir / f"demo-opencode-{variant}"
        trial_names = [f"{reward['entry_id']}__attempt" for reward in rewards]
        for trial_name, reward in zip(trial_names, rewards, strict=True):
            _write_verifier_outputs(job_dir, trial_name, reward)
        _write_complete_job_result(job_dir, trial_names)


def _collect(jobs_dir: Path, results_dir: Path, *, skip_baseline: bool = False) -> dict[str, Any]:
    return collect_harbor_results(
        skill_name="demo",
        agents=["opencode"],
        output_dir=results_dir,
        jobs_dir=jobs_dir,
        skip_baseline=skip_baseline,
        expected_cases=2,
        expected_case_ids=["case-1", "case-2"],
        expected_trials=2,
        agent_models={"opencode": {"model": AGENT_MODEL, "source": "test"}},
    )


def _generated_files(results_dir: Path) -> dict[str, bytes]:
    """Snapshot every collector output except per-trial copies of the raw rewards."""
    return {
        path.relative_to(results_dir).as_posix(): path.read_bytes()
        for path in sorted(results_dir.rglob("*"))
        if path.is_file() and "trials" not in path.relative_to(results_dir).parts
    }


def test_collect_writes_judge_panel_artifact_and_result_summary(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    _write_jobs(jobs_dir, with_rewards=_with_skill_rewards(), without_rewards=_without_skill_rewards())

    result = _collect(jobs_dir, results_dir)

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    artifact_path = results_dir / "opencode" / "judge_panel.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(_panel_report()))
    assert artifact == expected

    summary = agent["judge_panel"]
    assert "disagreement_cases" not in summary
    assert summary["disagreement_case_count"] == 10
    assert summary["per_judge"] == artifact["per_judge"]
    assert summary["agreement"] == artifact["agreement"]
    assert summary["lift_excluding_same_family"] == artifact["lift_excluding_same_family"]
    assert "judge_panel.json" in GENERATED_AGENT_ARTIFACTS


def test_panel_blocks_leave_every_existing_collector_output_unchanged(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    custom = {"domain_quality": 0.4}
    with_rewards = [{**reward, **custom} for reward in _with_skill_rewards()]
    without_rewards = [{**reward, "domain_quality": 0.2} for reward in _without_skill_rewards()]
    plain_jobs = tmp_path / "plain-jobs"
    panel_jobs = tmp_path / "panel-jobs"
    _write_jobs(
        plain_jobs,
        with_rewards=[_without_panel(reward) for reward in with_rewards],
        without_rewards=[_without_panel(reward) for reward in without_rewards],
    )
    _write_jobs(panel_jobs, with_rewards=with_rewards, without_rewards=without_rewards)

    plain_result = _collect(plain_jobs, results_dir)
    plain_files = _generated_files(results_dir)
    panel_result = _collect(panel_jobs, results_dir)
    panel_files = _generated_files(results_dir)

    assert "judge_panel" not in plain_result["agents"]["opencode"]
    assert "opencode/judge_panel.json" not in plain_files
    assert panel_files.pop("opencode/judge_panel.json")
    assert panel_files == plain_files
    assert json.loads(panel_files["opencode/custom_lift.json"]) == {
        "domain_quality": {"with_skill": 0.4, "without_skill": 0.2, "delta": 0.2, "direction": "up"}
    }
    panel_agent = dict(panel_result["agents"]["opencode"])
    assert panel_agent.pop("judge_panel")
    assert {**panel_result, "agents": {"opencode": panel_agent}} == plain_result

    # Re-collecting without panel data removes the stale generated artifact.
    stale_result = _collect(plain_jobs, results_dir)
    assert not (results_dir / "opencode" / "judge_panel.json").exists()
    assert "judge_panel" not in stale_result["agents"]["opencode"]
    assert _generated_files(results_dir) == plain_files


def test_baseline_skip_reports_with_skill_panel_statistics_only(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    _write_jobs(jobs_dir, with_rewards=_with_skill_rewards(), without_rewards=None)

    result = _collect(jobs_dir, results_dir, skip_baseline=True)

    assert result["execution_status"] == "succeeded"
    artifact = json.loads((results_dir / "opencode" / "judge_panel.json").read_text(encoding="utf-8"))
    openai = artifact["per_judge"][O_LABEL]
    assert openai["with_skill"]["accuracy"] == 0.7
    assert openai["without_skill"] == dict.fromkeys((*LLM_METRICS, "llm_overall"))
    assert openai["lift"]["llm_overall"] is None
    assert artifact["lift_sign_consistent"] is None
    assert artifact["lift_excluding_same_family"]["metrics"]["accuracy"] == {
        "with_skill": 0.8,
        "without_skill": None,
        "delta": None,
    }
    assert all(case["condition"] == "with_skill" for case in artifact["disagreement_cases"])


def test_failed_condition_contributes_no_panel_statistics(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    rewards = _with_skill_rewards()
    failed = dict(rewards[1])
    failed["evaluation_status"] = "failed"
    failed["evaluation_errors"] = {"accuracy": "Judge panel quorum not met for accuracy: 1/3 judges succeeded"}
    failed["accuracy"] = None
    _write_jobs(jobs_dir, with_rewards=[rewards[0], failed], without_rewards=None)

    result = _collect(jobs_dir, results_dir, skip_baseline=True)

    assert result["execution_status"] == "failed"
    assert result["agents"]["opencode"]["with_skill"] == {}
    assert "judge_panel" not in result["agents"]["opencode"]
    assert not (results_dir / "opencode" / "judge_panel.json").exists()
