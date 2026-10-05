# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``compare`` shows judge-panel agreement only for runs that used a judge panel."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from skillevaluator.tier3 import commands as tier3_commands

# Fleiss' kappa / Krippendorff's alpha, as the column header spells them.
_KAPPA_ALPHA = "\u03ba/\u03b1"

_WITH_SKILL = {
    "security": 1.0,
    "skill_execution": 0.9,
    "skill_efficiency": 0.7,
    "accuracy": 0.8,
    "goal_accuracy": 1.0,
    "behavior_check": 0.75,
}
_WITHOUT_SKILL = {
    "security": 0.9,
    "skill_execution": 0.5,
    "skill_efficiency": 0.6,
    "accuracy": 0.4,
    "goal_accuracy": 0.5,
    "behavior_check": 0.25,
}


@pytest.fixture
def console(monkeypatch: pytest.MonkeyPatch) -> Console:
    recording = Console(file=io.StringIO(), record=True, width=200)
    monkeypatch.setattr(tier3_commands, "console", recording)
    return recording


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "demo" / "20260709_010000"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text("{}", encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name, "agents": {}}), encoding="utf-8")
    (tmp_path / "demo").mkdir(exist_ok=True)
    return run_dir


def _agent(run_dir: Path, agent: str, *, baseline: bool = True) -> Path:
    agent_dir = run_dir / agent
    variants = [("with-skill", _WITH_SKILL)] + ([("without-skill", _WITHOUT_SKILL)] if baseline else [])
    for variant, scores in variants:
        (agent_dir / variant).mkdir(parents=True)
        (agent_dir / variant / "summary.json").write_text(
            json.dumps({"execution_status": "succeeded", "scores": scores, "num_trials": 2}),
            encoding="utf-8",
        )
    return agent_dir


def _judge_panel(agreement: dict[str, object]) -> dict[str, object]:
    """A ``<agent>/judge_panel.json`` payload in the collector's 1.0 schema."""
    return {
        "schema_version": "1.0",
        "metrics": ["accuracy", "goal_accuracy", "behavior_check"],
        "aggregation": "vote",
        "quorum": 2,
        "judges": [
            {
                "judge": "openai:gpt-5.6-sol",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "family": "openai",
                "same_family": False,
            }
        ],
        "agent_model": "nvidia/nvidia/nemotron-3-super-120b-a12b",
        "agent_family": "nvidia",
        "per_judge": {},
        "agreement": agreement,
        "lift_sign_consistent": True,
        "lift_signs": {"openai:gpt-5.6-sol": 1},
        "same_family_judges": [],
        "lift_excluding_same_family": None,
        "disagreement_cases": [],
        "disagreement_cases_truncated": 0,
    }


def _agreement(kappa: object, alpha: object) -> dict[str, object]:
    return {
        "fleiss_kappa": kappa,
        "kappa_items": 30,
        "krippendorff_alpha": alpha,
        "alpha_units": 6,
        "observed_agreement": 0.9,
    }


_AGREEMENT = {
    "accuracy": _agreement(0.61, 0.7),
    "goal_accuracy": _agreement(None, 0.5),
    "behavior_check": _agreement(None, None),
}


def _write_panel(agent_dir: Path, payload: object) -> None:
    (agent_dir / "judge_panel.json").write_text(json.dumps(payload), encoding="utf-8")


def _compare(tmp_path: Path, console: Console) -> str:
    assert tier3_commands.compare_results(tmp_path / "demo", results_dir=tmp_path / "results") == 0
    return console.export_text()


def _cells(line: str) -> list[str]:
    """Split one rendered table line into cells, dropping the panel border."""
    return line.replace("│", " ").split()


def _row(text: str, label: str) -> list[str]:
    rows = [_cells(line) for line in text.splitlines() if _cells(line)[:1] == [label]]
    assert len(rows) == 1, text
    return rows[0]


def test_compare_adds_a_judge_agreement_column_for_a_panel_run(tmp_path: Path, console: Console) -> None:
    run_dir = _run_dir(tmp_path)
    _write_panel(_agent(run_dir, "opencode"), _judge_panel(_AGREEMENT))

    text = _compare(tmp_path, console)

    # Rich bottom-aligns the two-line headers: "judges" sits above the kappa/alpha line.
    lines = text.splitlines()
    header = next(index for index, line in enumerate(lines) if "Evaluator" in line)
    assert _cells(lines[header - 1])[-1] == "judges"
    assert _cells(lines[header])[-1] == _KAPPA_ALPHA
    assert _row(text, "accuracy") == ["accuracy", "0.80", "+0.40", "0.61/0.70"]
    assert _row(text, "goal_accuracy") == ["goal_accuracy", "1.00", "+0.50", "-/0.50"]
    assert _row(text, "behavior_check") == ["behavior_check", "0.75", "+0.50", "-"]
    # Deterministic metrics and the overall row leave the agreement cell blank.
    assert _row(text, "security") == ["security", "1.00", "+0.10"]
    assert _row(text, "skill_execution") == ["skill_execution", "0.90", "+0.40"]
    assert _row(text, "Overall") == ["Overall", "0.86", "+0.33"]


def test_compare_places_the_column_after_only_the_panel_agents_columns(tmp_path: Path, console: Console) -> None:
    run_dir = _run_dir(tmp_path)
    _agent(run_dir, "codex")
    _write_panel(_agent(run_dir, "opencode", baseline=False), _judge_panel(_AGREEMENT))

    text = _compare(tmp_path, console)

    assert text.count(_KAPPA_ALPHA) == 1
    # codex: score + lift; opencode: score + judges (it has no baseline column).
    assert _row(text, "accuracy") == ["accuracy", "0.80", "+0.40", "0.80", "0.61/0.70"]
    assert _row(text, "security") == ["security", "1.00", "+0.10", "1.00"]
    assert _row(text, "Overall") == ["Overall", "0.86", "+0.33", "0.86"]


def test_compare_renders_non_numeric_agreement_as_unavailable(tmp_path: Path, console: Console) -> None:
    run_dir = _run_dir(tmp_path)
    agreement = {
        "accuracy": _agreement(True, "0.9"),
        "goal_accuracy": _agreement(float("nan"), 0.25),
        "behavior_check": "not-a-mapping",
    }
    _write_panel(_agent(run_dir, "opencode"), _judge_panel(agreement))

    text = _compare(tmp_path, console)

    assert _row(text, "accuracy")[-1] == "-"
    assert _row(text, "goal_accuracy")[-1] == "-/0.25"
    assert _row(text, "behavior_check")[-1] == "-"


def test_compare_shows_unavailable_agreement_when_the_panel_has_none(tmp_path: Path, console: Console) -> None:
    run_dir = _run_dir(tmp_path)
    _write_panel(_agent(run_dir, "opencode"), {"schema_version": "1.0"})

    text = _compare(tmp_path, console)

    assert _KAPPA_ALPHA in text
    for metric in ("accuracy", "goal_accuracy", "behavior_check"):
        assert _row(text, metric)[-1] == "-"


@pytest.mark.parametrize("variant", ["non-object", "malformed", "symlink", "directory"])
def test_compare_output_is_unchanged_without_a_usable_judge_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    run_dir = _run_dir(tmp_path)
    agent_dir = _agent(run_dir, "opencode")

    baseline_console = Console(file=io.StringIO(), record=True, width=200)
    monkeypatch.setattr(tier3_commands, "console", baseline_console)
    baseline = _compare(tmp_path, baseline_console)
    assert _KAPPA_ALPHA not in baseline
    assert "judges" not in baseline

    panel_path = agent_dir / "judge_panel.json"
    if variant == "non-object":
        panel_path.write_text("[]\n", encoding="utf-8")
    elif variant == "malformed":
        panel_path.write_text("{not json", encoding="utf-8")
    elif variant == "symlink":
        # A linked artifact is never followed out of the run directory.
        target = tmp_path / "elsewhere.json"
        target.write_text(json.dumps(_judge_panel(_AGREEMENT)), encoding="utf-8")
        panel_path.symlink_to(target)
    else:
        panel_path.mkdir()

    again_console = Console(file=io.StringIO(), record=True, width=200)
    monkeypatch.setattr(tier3_commands, "console", again_console)
    assert _compare(tmp_path, again_console) == baseline
