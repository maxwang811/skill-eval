# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge Panel section of the canonical Tier 3 report.

Reports render through the real path: ``judge_panel.json`` on disk, the
bounded report loader, ``build_agent_eval_payload``, and ``HTMLReporter``.
Without panel data the payload and the rendered HTML must be byte-identical to
a report built without the feature.
"""

from __future__ import annotations

import base64
import copy
import html as html_lib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import MockUrllibResponse
from tests.tier3.test_judge_panel_collector import (
    AGENT_MODEL,
    ANTHROPIC,
    LLM_METRICS,
    OPENAI,
    _collect,
    _panel_report,
    _signed_lift_rows,
    _split_verdict_rewards,
    _with_skill_rewards,
    _without_skill_rewards,
    _write_complete_job_result,
    _write_jobs,
)

from skillevaluator.evaluation import tier3_report
from skillevaluator.evaluation.tier3_report import build_agent_eval_payload, render_agent_eval_html_report
from skillevaluator.models import ValidationResult
from skillevaluator.reporting import HTMLReporter
from skillevaluator.reporting import html as html_module
from skillevaluator.tier3.eval_core.llm_judge import aggregate_panel
from skillevaluator.tier3.harbor import report, report_data
from skillevaluator.tier3.harbor.judge_panel_stats import build_judge_panel_report
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS

O_LABEL = "openai:gpt-5.6-sol"
A_LABEL = "anthropic:claude-opus-5"
N_LABEL = "nv_build:nvidia/nemotron-3-super-120b-a12b"
SCRIPT_REASON = "<script>alert(1)</script>"
_EVAL_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)
_JUDGE_ENDPOINTS = {
    "https://api.openai.com/v1/chat/completions": "openai",
    "https://api.anthropic.com/v1/messages": "anthropic",
    "https://integrate.api.nvidia.com/v1/chat/completions": "nv_build",
}
_JUDGE_KEYS = {
    "openai": ("OPENAI_API_KEY", "openai-panel-judge-key"),
    "anthropic": ("ANTHROPIC_API_KEY", "anthropic-panel-judge-key"),
    "nv_build": ("NVIDIA_API_KEY", "nvidia-panel-judge-key"),
}
_VERIFIER_ENV_TO_CLEAR = (
    "SKILL_EVAL_LLM_BASE_URL",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_LLM_MODEL",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "LLM_JUDGE_MODEL",
    "SKILL_EVAL_JUDGE_MODEL",
    "LLM_JUDGE_FALLBACK_MODELS",
    "SKILL_EVAL_JUDGE_PANEL_AGGREGATION",
    "SKILL_EVAL_JUDGE_PANEL_QUORUM",
    "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT",
)
# Every template addition sits in a block that renders nothing without panel
# data; stripping these blocks must reproduce the pre-feature template output.
_TEMPLATE_PANEL_BLOCK = re.compile(
    r"\s*\{%- if tier3\.get\('judge_panel'\) %\}\{# judge-panel:(?P<name>[a-z-]+) #\}.*?\{# /judge-panel:(?P=name) #\}",
    re.DOTALL,
)


def _metric_values(accuracy: float | None, goal: float | None, behavior: float | None) -> dict[str, float | None]:
    values = {"accuracy": accuracy, "goal_accuracy": goal, "behavior_check": behavior}
    present = [value for value in values.values() if value is not None]
    values["llm_overall"] = round(sum(present) / 3, 4) if len(present) == 3 else None
    return values


def _judge_entry(provider: str, model: str, family: str, with_skill: tuple, without_skill: tuple) -> dict[str, Any]:
    with_values = _metric_values(*with_skill)
    without_values = _metric_values(*without_skill)
    return {
        "provider": provider,
        "model": model,
        "family": family,
        "with_skill": with_values,
        "without_skill": without_values,
        "lift": {
            metric: round(with_values[metric] - without_values[metric], 4)
            if with_values[metric] is not None and without_values[metric] is not None
            else None
            for metric in with_values
        },
        "n": {
            "with_skill": {"accuracy": 2, "goal_accuracy": 2, "behavior_check": 2},
            "without_skill": {"accuracy": 2, "goal_accuracy": 2, "behavior_check": 2},
        },
        "n_skipped": {
            "with_skill": {"accuracy": 0, "goal_accuracy": 0, "behavior_check": 0},
            "without_skill": {"accuracy": 0, "goal_accuracy": 0, "behavior_check": 0},
        },
        "failed": {"with_skill": 0, "without_skill": 0},
    }


def _panel_artifact(**overrides: Any) -> dict[str, Any]:
    """Return a contract-shaped ``judge_panel.json`` with hand-picked agreement values."""
    panel: dict[str, Any] = {
        "schema_version": "1.0",
        "metrics": ["accuracy", "goal_accuracy", "behavior_check"],
        "aggregation": "vote",
        "quorum": 2,
        "disagreement_threshold": 0.4,
        "judges": [
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
        ],
        "agent_model": AGENT_MODEL,
        "agent_family": "nvidia",
        "per_judge": {
            O_LABEL: _judge_entry("openai", "gpt-5.6-sol", "openai", (0.7, 0.9, 0.75), (0.3, 0.0, 0.25)),
            A_LABEL: _judge_entry("anthropic", "claude-opus-5", "anthropic", (0.9, 1.0, 0.75), (0.5, 0.25, 0.25)),
            N_LABEL: _judge_entry(
                "nv_build",
                "nvidia/nemotron-3-super-120b-a12b",
                "nvidia",
                (0.6, 0.5, 0.75),
                (0.9, 1.0, 1.0),
            ),
        },
        "agreement": {
            "accuracy": {
                "fleiss_kappa": 0.15,
                "kappa_items": 10,
                "krippendorff_alpha": 0.35,
                "alpha_units": 2,
                "observed_agreement": 0.7,
            },
            "goal_accuracy": {
                "fleiss_kappa": None,
                "kappa_items": 2,
                "krippendorff_alpha": 0.9,
                "alpha_units": 2,
                "observed_agreement": 1.0,
            },
            "behavior_check": {
                "fleiss_kappa": 0.4,
                "kappa_items": 4,
                "krippendorff_alpha": 0.2,
                "alpha_units": 2,
                "observed_agreement": 0.75,
            },
        },
        "lift_sign_consistent": False,
        "lift_signs": {O_LABEL: 1, A_LABEL: 1, N_LABEL: -1},
        "same_family_judges": [N_LABEL],
        "lift_excluding_same_family": {
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
        },
        "same_family_comparison": {
            "same_family_mean_llm_lift": -0.35,
            "other_judges_mean_llm_lift": 0.575,
            "gap": -0.925,
        },
        "disagreement_cases": [
            {
                "entry_id": "case-1",
                "trial_id": "case-1__attempt",
                "condition": "with_skill",
                "metric": "goal_accuracy",
                "spread": 1.0,
                "scores": {O_LABEL: 1.0, A_LABEL: 1.0, N_LABEL: 0.0},
                "reasons": {O_LABEL: "the goal was met", A_LABEL: "goal fully met", N_LABEL: SCRIPT_REASON},
            },
            {
                "entry_id": "case-2",
                "trial_id": "case-2__attempt",
                "step": "finish",
                "condition": "without_skill",
                "metric": "accuracy",
                "spread": 0.8,
                "scores": {O_LABEL: 0.2, A_LABEL: 0.4, N_LABEL: None},
                "reasons": {O_LABEL: "wrong total", A_LABEL: "partly right", N_LABEL: "judge timed out"},
            },
        ],
        "disagreement_cases_truncated": 3,
    }
    panel.update(overrides)
    return panel


def _write_run(run_dir: Path, *, judge_panel: dict[str, Any] | None) -> None:
    for variant, score in (("with-skill", 0.9), ("without-skill", 0.5)):
        summary = run_dir / "opencode" / variant / "summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {
                    "agent": "opencode",
                    "model": AGENT_MODEL,
                    "scores": dict.fromkeys(DEFAULT_METRICS, score),
                    "metrics": list(DEFAULT_METRICS),
                    "num_trials": 0,
                    "execution_status": "succeeded",
                    "execution_errors": [],
                    "expected_attempts": 0,
                    "scored_attempts": 0,
                }
            ),
            encoding="utf-8",
        )
    if judge_panel is not None:
        (run_dir / "opencode" / "judge_panel.json").write_text(json.dumps(judge_panel), encoding="utf-8")


def _render(tmp_path: Path, run_dir: Path) -> str:
    skill = tmp_path / "demo"
    skill.mkdir(exist_ok=True)
    return render_agent_eval_html_report(skill, run_dir, use_llm_judge=False).read_text(encoding="utf-8")


def _render_payload(payload: dict[str, Any]) -> str:
    result = ValidationResult(validator_name="AGENT_EVAL", validator_description="Live evaluation")
    result.metadata["agent_eval"] = payload
    return HTMLReporter(include_timestamp=False).render_all([result])


def _embedded_payload(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="tier3-full"(?P<attrs>[^>]*)>(?P<body>.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    body = match.group("body").strip()
    if 'data-encoding="base64"' in match.group("attrs"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _judge_panel_section(html: str) -> str:
    assert 'id="tier3-judge-panel"' in html
    return html.split('id="tier3-judge-panel"', 1)[1].split('id="tier3-page-trials"', 1)[0]


def _warning_text(section: str, name: str) -> str:
    """Return the visible text of one judge-panel warning box with whitespace collapsed."""
    body = section.split(f'data-judge-warning="{name}">', 1)[1].split("</div>", 1)[0]
    return " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", body)).split())


def _agreement_cell(section: str, metric: str, stat: str) -> tuple[set[str], str]:
    match = re.search(
        rf'<td class="(?P<classes>[^"]*)" data-agreement-metric="{metric}" data-agreement-stat="{stat}">'
        r"\s*(?P<text>[^<]*?)\s*</td>",
        section,
    )
    assert match is not None, (metric, stat)
    return set(match.group("classes").split()), match.group("text")


def _agents(run_dir: Path) -> dict[str, dict[str, Any]]:
    return report_data.load_agent_data(run_dir)


# ---------------------------------------------------------------------------
# Loader and payload
# ---------------------------------------------------------------------------


def test_report_loader_reads_the_judge_panel_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=_panel_artifact())

    agents = _agents(run_dir)

    assert agents["opencode"]["judge_panel"] == _panel_artifact()


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]", "x" * (2 * 1024 * 1024 + 1)])
def test_report_loader_ignores_missing_or_unusable_judge_panel_artifacts(tmp_path: Path, content: str | None) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    if content is not None:
        (run_dir / "opencode" / "judge_panel.json").write_text(content, encoding="utf-8")

    agents = _agents(run_dir)

    assert "judge_panel" not in agents["opencode"]
    if content is not None and len(content) > 2 * 1024 * 1024:
        assert {"code": "json_bytes", "artifact": "judge_panel", "limit": 2 * 1024 * 1024} in agents["opencode"][
            "_report_truncation"
        ]["reasons"]


@pytest.mark.skipif(not Path("/dev/fd").is_dir(), reason="needs /dev/fd to count open descriptors")
def test_report_loader_closes_the_descriptor_of_a_directory_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    artifact = run_dir / "opencode" / "judge_panel.json"
    artifact.mkdir()

    before = len(list(Path("/dev/fd").iterdir()))
    for _ in range(25):
        diagnostics: list[dict[str, Any]] = []
        assert (
            report_data._load_bounded_json(artifact, diagnostics, artifact="judge_panel") is report_data._INVALID_JSON
        )
        assert diagnostics == [{"code": "json_file_type", "artifact": "judge_panel", "limit": 0}]
    agents = _agents(run_dir)
    after = len(list(Path("/dev/fd").iterdir()))

    # os.open() succeeds on a directory and os.fdopen() then fails without closing it.
    assert after <= before
    assert "judge_panel" not in agents["opencode"]
    assert {"code": "json_file_type", "artifact": "judge_panel", "limit": 0} in agents["opencode"][
        "_report_truncation"
    ]["reasons"]


def test_payload_exposes_judge_panel_only_when_an_agent_has_one(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    plain_agents = _agents(run_dir)
    panel_agents = copy.deepcopy(plain_agents)
    panel_agents["opencode"]["judge_panel"] = _panel_artifact()

    plain = build_agent_eval_payload("demo", plain_agents, use_llm_judge=False)
    with_panel = build_agent_eval_payload("demo", panel_agents, use_llm_judge=False)

    assert plain is not None
    assert with_panel is not None
    assert "judge_panel" not in plain
    assert with_panel["judge_panel"] == {"opencode": _panel_artifact()}
    without_panel_key = {key: value for key, value in with_panel.items() if key != "judge_panel"}
    assert json.dumps(without_panel_key) == json.dumps(plain)


def test_payload_normalizes_malformed_judge_panel_data_for_display(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    agents = _agents(run_dir)
    malformed = _panel_artifact()
    malformed["judges"].append("not-a-judge")
    malformed["per_judge"][O_LABEL]["with_skill"]["accuracy"] = "0.7"
    malformed["per_judge"][A_LABEL] = "broken"
    malformed["agreement"]["accuracy"]["fleiss_kappa"] = float("nan")
    malformed["agreement"]["goal_accuracy"] = ["not", "a", "mapping"]
    malformed["lift_sign_consistent"] = "false"
    malformed["disagreement_cases"].append("not-a-case")
    malformed["disagreement_cases"][0]["reasons"][O_LABEL] = {"nested": "value"}
    malformed["disagreement_cases_truncated"] = -4
    agents["opencode"]["judge_panel"] = malformed
    agents["other"] = {**copy.deepcopy(agents["opencode"]), "judge_panel": {"judges": "nobody"}}

    payload = build_agent_eval_payload("demo", agents, use_llm_judge=False)

    assert payload is not None
    assert set(payload["judge_panel"]) == {"opencode"}
    panel = payload["judge_panel"]["opencode"]
    assert [judge["judge"] for judge in panel["judges"]] == [O_LABEL, A_LABEL, N_LABEL]
    assert panel["per_judge"][O_LABEL]["with_skill"]["accuracy"] is None
    assert panel["per_judge"][A_LABEL]["with_skill"] == dict.fromkeys(
        ("accuracy", "goal_accuracy", "behavior_check", "llm_overall")
    )
    assert panel["agreement"]["accuracy"]["fleiss_kappa"] is None
    assert panel["agreement"]["goal_accuracy"]["krippendorff_alpha"] is None
    assert panel["lift_sign_consistent"] is None
    assert len(panel["disagreement_cases"]) == 2
    assert O_LABEL not in panel["disagreement_cases"][0]["reasons"]
    assert panel["disagreement_cases_truncated"] == 0
    html = _render_payload(payload)
    assert "Judge Panel" in html


def test_payload_budget_prunes_disagreement_cases_first(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    agents = _agents(run_dir)
    agents["opencode"]["judge_panel"] = _panel_artifact()

    kept = build_agent_eval_payload("demo", copy.deepcopy(agents), use_llm_judge=False)
    pruned = build_agent_eval_payload(
        "demo",
        agents,
        comparison={"unrendered_blob": "x" * (3 * 1024 * 1024)},
        use_llm_judge=False,
    )

    assert kept is not None
    assert len(kept["judge_panel"]["opencode"]["disagreement_cases"]) == 2
    assert "report_truncation" not in kept
    assert pruned is not None
    panel = pruned["judge_panel"]["opencode"]
    assert panel["disagreement_cases"] == []
    assert panel["disagreement_cases_truncated"] == 5
    assert panel["per_judge"] == _panel_artifact()["per_judge"]
    assert pruned["report_truncation"]["omitted"]["judge_panel_disagreement_cases"] == 2


_FIVE_JUDGES = (
    ("openai", "gpt-5.6-sol", "openai"),
    ("anthropic", "claude-opus-5", "anthropic"),
    ("nv_build", "nvidia/nemotron-3-super-120b-a12b", "nvidia"),
    ("openai-compatible", "google/gemini-3-pro", "google"),
    ("bedrock", "us.meta.llama4-maverick-17b-instruct-v1:0", "meta"),
)


def _five_judge_panel(*, cases: int, reason_chars: int) -> dict[str, Any]:
    """Return the largest panel the collector writes: five judges and capped cases with full-length reasons."""
    labels = [f"{provider}:{model}" for provider, model, _family in _FIVE_JUDGES]
    reason = ("The agent reported the regional totals but skipped the per-quarter breakdown. " * 10)[:reason_chars]
    return _panel_artifact(
        judges=[
            {"judge": label, "provider": provider, "model": model, "family": family, "same_family": family == "nvidia"}
            for label, (provider, model, family) in zip(labels, _FIVE_JUDGES, strict=True)
        ],
        per_judge={
            label: _judge_entry(provider, model, family, (0.8, 0.9, 0.7), (0.4, 0.3, 0.5))
            for label, (provider, model, family) in zip(labels, _FIVE_JUDGES, strict=True)
        },
        lift_sign_consistent=True,
        lift_signs=dict.fromkeys(labels, 1),
        disagreement_cases=[
            {
                "entry_id": f"case-{index:03d}",
                "trial_id": f"case-{index:03d}__attempt",
                "condition": "with_skill",
                "metric": "accuracy",
                "spread": 0.8,
                "scores": dict.fromkeys(labels, 0.6),
                "reasons": dict.fromkeys(labels, reason),
            }
            for index in range(cases)
        ],
        disagreement_cases_truncated=0,
    )


def test_large_judge_panel_does_not_starve_the_html_preview_overview(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    agents = _agents(run_dir)
    agents["opencode"]["judge_panel"] = _five_judge_panel(cases=50, reason_chars=512)
    dataset = [
        {"id": f"case-{index:03d}", "prompt": ("Sum the regional totals and explain each step. " * 70)[:3000]}
        for index in range(40)
    ]

    payload = build_agent_eval_payload("demo", agents, dataset=dataset, use_llm_judge=False)

    assert payload is not None
    # Over the preview trigger, the HTML renders a 128K-character projection spent in key order.
    assert len(html_module._compact_json(payload).encode("utf-8")) > html_module._TIER3_HTML_PREVIEW_TRIGGER_BYTES
    keys = list(payload)
    assert keys.index("judge_panel") == keys.index("dataset") - 1
    overview = ("agents", "dimensions", "evaluator_cards", "pass_at_k", "insights", "conclusions", "recommendations")
    assert keys.index("judge_panel") > max(keys.index(key) for key in overview)
    html = _render_payload(payload)
    visible = re.sub(r"<script\b.*?</script>", "", html, flags=re.DOTALL)
    assert "<strong>Correctness</strong>" in visible
    assert "opencode leads with overall score" in visible
    assert "t3-pill ok" in visible
    assert "{&#39;" not in visible
    section = _judge_panel_section(html)
    assert re.findall(r'<tr data-judge="([^"]*)"', section) == [
        f"{provider}:{model}" for provider, model, _family in _FIVE_JUDGES
    ]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def test_judge_panel_section_renders_tables_warnings_and_cases(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=_panel_artifact())

    html = _render(tmp_path, run_dir)

    section = _judge_panel_section(html)
    assert "Judge Panel" in html
    for label in (O_LABEL, A_LABEL, N_LABEL):
        assert label in section
    assert section.count('<tr data-judge="') == 3
    assert "same family" in section
    assert "+0.60" in section  # gpt-5.6-sol LLM overall lift
    assert "-0.35" in section  # nemotron LLM overall lift
    sign_warning = _warning_text(section, "lift-sign")
    assert sign_warning.startswith("Judges disagree on the direction of the skill lift.")
    assert f"{O_LABEL} up, {A_LABEL} up, {N_LABEL} down." in sign_warning
    assert 'data-judge-note="lift-sign-unavailable"' not in section
    same_family_warning = _warning_text(section, "same-family")
    assert N_LABEL in same_family_warning
    # The per-judge comparison comes first, then the re-voted lift next to its reference.
    comparison = (
        f"Their mean LLM overall lift is {-0.35:+.2f}, against {0.575:+.2f} for the other judges (gap {-0.925:+.2f})."
    )
    excluded = f"Re-voted without them, the LLM overall lift is {0.575:+.2f} and the overall lift is {0.5042:+.2f};"
    reference = (
        f"dropping as many other-family judges instead gives {0.1417:+.2f} and {0.2875:+.2f} "
        f"(LLM overall gap {-0.4333:+.2f}, averaged over 2 subsets)."
    )
    assert comparison in same_family_warning
    assert excluded in same_family_warning
    assert reference in same_family_warning
    assert same_family_warning.index(comparison) < same_family_warning.index(excluded)
    assert "not with the headline lift" in same_family_warning

    assert _agreement_cell(section, "accuracy", "kappa") == ({"t3-agree-value", "t3-agree-poor"}, "0.15")
    assert _agreement_cell(section, "accuracy", "alpha") == ({"t3-agree-value", "t3-agree-weak"}, "0.35")
    assert _agreement_cell(section, "goal_accuracy", "kappa") == ({"t3-agree-value"}, "n/a")
    assert _agreement_cell(section, "goal_accuracy", "alpha") == ({"t3-agree-value"}, "0.90")
    assert _agreement_cell(section, "behavior_check", "kappa") == ({"t3-agree-value"}, "0.40")
    assert _agreement_cell(section, "behavior_check", "alpha") == ({"t3-agree-value", "t3-agree-weak"}, "0.20")

    cases = re.findall(r'<details class="t3-judge-case".*?</details>', section, re.DOTALL)
    assert len(cases) == 2
    assert "case-1" in cases[0]
    assert "Goal Accuracy" in cases[0]
    assert "the goal was met" in cases[0]
    assert "goal fully met" in cases[0]
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in cases[0]
    assert "judge timed out" in cases[1]
    assert "finish" in cases[1]
    assert "3 more" in section
    assert SCRIPT_REASON not in html


def test_warnings_appear_only_when_warranted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        judge_panel=_panel_artifact(
            lift_sign_consistent=True,
            lift_signs={O_LABEL: 1, A_LABEL: 1, N_LABEL: 1},
            same_family_judges=[],
            lift_excluding_same_family=None,
            disagreement_cases=[],
            disagreement_cases_truncated=0,
        ),
    )

    section = _judge_panel_section(_render(tmp_path, run_dir))

    assert 'data-judge-warning="lift-sign"' not in section
    assert 'data-judge-warning="same-family"' not in section
    assert '<details class="t3-judge-case"' not in section
    assert "No judge disagreements" in section


def test_undefined_sign_check_and_all_same_family_panel_render_without_warnings_about_lift(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        judge_panel=_panel_artifact(
            lift_sign_consistent=None,
            lift_signs={},
            same_family_judges=[O_LABEL, A_LABEL, N_LABEL],
            lift_excluding_same_family={"excluded": [O_LABEL, A_LABEL, N_LABEL], "judges": [], "available": False},
            # No other judges to compare with, so the collector writes no comparison.
            same_family_comparison=None,
        ),
    )

    section = _judge_panel_section(_render(tmp_path, run_dir))

    assert 'data-judge-warning="lift-sign"' not in section
    # Three judges but no sign check: say so instead of staying silent.
    assert 'data-judge-note="lift-sign-unavailable"' in section
    assert "Sign check unavailable" in section
    same_family_warning = _warning_text(section, "same-family")
    assert "Every judge shares the agent" in same_family_warning
    assert "Their mean LLM overall lift" not in same_family_warning
    assert "Re-voted without them" not in same_family_warning


@pytest.mark.parametrize("comparison", [None, {}, "not-a-mapping"])
def test_same_family_box_renders_without_a_comparison(tmp_path: Path, comparison: object) -> None:
    # The other judges have no LLM overall lift (or the artifact lacks a usable comparison):
    # the box still reports the re-voted lift and its reference.
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=_panel_artifact(same_family_comparison=comparison))

    section = _judge_panel_section(_render(tmp_path, run_dir))

    same_family_warning = _warning_text(section, "same-family")
    assert "Their mean LLM overall lift" not in same_family_warning
    assert "Re-voted without them, the LLM overall lift is" in same_family_warning


def test_sign_check_unavailable_note_needs_two_judges(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = _panel_artifact(lift_sign_consistent=None, lift_signs={}, same_family_judges=[])
    artifact["judges"] = artifact["judges"][:1]
    artifact["lift_excluding_same_family"] = None
    artifact["same_family_comparison"] = None
    _write_run(run_dir, judge_panel=artifact)

    section = _judge_panel_section(_render(tmp_path, run_dir))

    assert 'data-judge-warning="lift-sign"' not in section
    assert 'data-judge-note="lift-sign-unavailable"' not in section
    assert 'data-judge-warning="same-family"' not in section


def test_flat_judge_next_to_a_rising_one_raises_no_sign_warning(tmp_path: Path) -> None:
    # gpt-5.6-sol scores the same in both arms (a ceiling effect) while claude-opus-5 sees the skill help.
    report = build_judge_panel_report(_signed_lift_rows({OPENAI: 0, ANTHROPIC: 1}), agent_model=AGENT_MODEL)
    assert report is not None
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=report)

    section = _judge_panel_section(_render(tmp_path, run_dir))

    assert report["lift_signs"] == {O_LABEL: 0, A_LABEL: 1}
    assert 'data-judge-warning="lift-sign"' not in section
    assert 'data-judge-note="lift-sign-unavailable"' not in section


def test_disagreement_case_summaries_name_the_trial(tmp_path: Path) -> None:
    def case(entry_id: str, trial_id: str) -> dict[str, Any]:
        return {
            "entry_id": entry_id,
            "trial_id": trial_id,
            "condition": "with_skill",
            "metric": "accuracy",
            "spread": 0.8,
            "scores": {O_LABEL: 0.2, A_LABEL: 1.0},
            "reasons": {O_LABEL: "wrong total", A_LABEL: "right total"},
        }

    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        judge_panel=_panel_artifact(
            disagreement_cases=[
                case("case-1", "case-1__attempt-1"),
                case("case-1", "case-1__attempt-2"),
                case("case-2", "case-2"),
            ],
            disagreement_cases_truncated=0,
        ),
    )

    section = _judge_panel_section(_render(tmp_path, run_dir))

    summaries = [
        " ".join(re.sub(r"<[^>]+>", " ", summary).split())
        for summary in re.findall(r'<details class="t3-judge-case">\s*<summary>(.*?)</summary>', section, re.DOTALL)
    ]
    # Repeated attempts of one case, metric, and arm stay distinguishable.
    assert summaries[0].startswith("case-1 · trial case-1__attempt-1 · Accuracy")
    assert summaries[1].startswith("case-1 · trial case-1__attempt-2 · Accuracy")
    # A trial id equal to the case id is not repeated.
    assert summaries[2].startswith("case-2 · Accuracy")
    assert summaries[2].count("case-2") == 1


def test_collected_panel_results_render_through_the_report_path(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    _write_jobs(jobs_dir, with_rewards=_with_skill_rewards(), without_rewards=_without_skill_rewards())
    _collect(jobs_dir, results_dir)

    html = _render(tmp_path, results_dir)

    section = _judge_panel_section(html)
    payload = _embedded_payload(html)
    artifact = json.loads((results_dir / "opencode" / "judge_panel.json").read_text(encoding="utf-8"))
    assert payload["judge_panel"] == {"opencode": artifact}
    assert 'data-judge-warning="lift-sign"' in section
    assert 'data-judge-warning="same-family"' in section
    assert len(re.findall(r'<details class="t3-judge-case"', section)) == 10


def test_report_without_panel_data_has_no_section_and_unchanged_payload(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    results_dir = tmp_path / "results"
    _write_jobs(jobs_dir, with_rewards=_with_skill_rewards(), without_rewards=_without_skill_rewards())
    _collect(jobs_dir, results_dir)
    (results_dir / "opencode" / "judge_panel.json").unlink()

    html = _render(tmp_path, results_dir)

    assert 'id="tier3-judge-panel"' not in html
    assert "Judge Panel" not in html
    assert "t3-agree" not in html
    assert "t3-judge-" not in html
    assert "judge_panel" not in _embedded_payload(html)


def test_report_without_panel_data_is_byte_identical_to_the_pre_feature_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir, judge_panel=None)
    payload = build_agent_eval_payload("demo", _agents(run_dir), use_llm_judge=False)
    assert payload is not None
    with_panel = copy.deepcopy(payload)
    with_panel["judge_panel"] = {"opencode": _panel_artifact()}
    current = _render_payload(payload)
    assert 'id="tier3-judge-panel"' in _render_payload(with_panel)

    original_get_source = html_module.PackageLoader.get_source

    def pre_feature_source(self, environment, template):
        source, name, uptodate = original_get_source(self, environment, template)
        stripped, count = _TEMPLATE_PANEL_BLOCK.subn("", source)
        assert count == 2
        return stripped, name, uptodate

    monkeypatch.setattr(html_module.PackageLoader, "get_source", pre_feature_source)
    baseline = _render_payload(payload)

    assert current == baseline
    # The stripped template really is the pre-feature one: it has no panel section.
    assert 'id="tier3-judge-panel"' not in _render_payload(with_panel)


# ---------------------------------------------------------------------------
# Verifier to report
# ---------------------------------------------------------------------------


def _judge_reply(metric: str, member: dict[str, Any]) -> str:
    """Return the JSON a judge would send for one hand-built panel member; a failed member replies with prose."""
    if member["status"] != "ok":
        return "I cannot grade this response."
    if metric == "accuracy":
        payload = {"criteria": member["criteria"], "score": member["score"], "reason": member["reason"]}
    elif metric == "goal_accuracy":
        payload = {
            "user_goal": "finish the task",
            "end_state": "the task state",
            "achieved": member["achieved"],
            "score": member["score"],
            "reason": member["reason"],
        }
    else:
        payload = {"results": member["results"], "score": member["score"], "summary": member["reason"]}
    return json.dumps(payload)


def _run_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    fixture: dict[str, Any],
    has_skill: bool,
    calls: list[dict[str, Any]],
    expected_behavior: tuple[str, ...] = ("Read the numbers", "Report the sum"),
) -> Path:
    """Run the real verifier ``main()`` for one trial; each judge answers from the fixture's panel members."""
    module_name = f"judge_panel_e2e_{name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, verifier)
    spec.loader.exec_module(verifier)

    logs_dir = tmp_path / "verifier-runs" / name / "logs"
    tests_dir = tmp_path / "verifier-runs" / name / "tests"
    for directory in (logs_dir / "agent", logs_dir / "verifier", tests_dir):
        directory.mkdir(parents=True)
    verifier.LOGS_DIR = logs_dir
    verifier.AGENT_LOGS_DIR = logs_dir / "agent"
    verifier.VERIFIER_DIR = logs_dir / "verifier"
    verifier.TESTS_DIR = tests_dir
    verifier.ATIF_PATH = logs_dir / "agent" / "trajectory.json"
    verifier.ENTRY_PATH = tests_dir / "entry.json"
    verifier.REWARD_JSON = logs_dir / "verifier" / "reward.json"
    verifier.REWARD_TXT = logs_dir / "verifier" / "reward.txt"
    verifier.SKILL_EVALUATOR_REWARD_JSON = logs_dir / "verifier" / "skill_evaluator_reward.json"
    verifier.ATIF_PATH.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": f"Sum the numbers for {fixture['entry_id']}."},
                    {"source": "agent", "message": "The sum is 42."},
                ]
            }
        ),
        encoding="utf-8",
    )
    verifier.ENTRY_PATH.write_text(
        json.dumps(
            {
                "id": fixture["entry_id"],
                "question": f"Sum the numbers for {fixture['entry_id']}.",
                "ground_truth": "The sum is 42.",
                "expected_behavior": list(expected_behavior),
                "should_trigger": False,
                "evaluated_skill": "demo",
                "has_skill": has_skill,
            }
        ),
        encoding="utf-8",
    )
    # A metric the fixture skipped has no panel block and the verifier never asks a judge about it.
    members = {
        metric: {member["provider"]: member for member in fixture["details"][metric]["panel"]["members"]}
        for metric in LLM_METRICS
        if "panel" in fixture["details"][metric]
    }

    def fake_urlopen(request, *_args, **_kwargs):
        provider = _JUDGE_ENDPOINTS[request.full_url]
        headers = {name.lower(): value for name, value in request.header_items()}
        body = json.loads(request.data)
        prompt = "\n".join(str(message.get("content")) for message in body.get("messages", []))
        if "SKILL_IDENTIFIED" in prompt:
            metric = "accuracy"
        elif "achieve the expected goal" in prompt:
            metric = "goal_accuracy"
        else:
            metric = "behavior_check"
        calls.append(
            {
                "provider": provider,
                "credential": headers.get("x-api-key") or headers.get("authorization", ""),
                "model": body.get("model"),
                "metric": metric,
            }
        )
        reply = _judge_reply(metric, members[metric][provider])
        if provider == "anthropic":
            return MockUrllibResponse({"content": [{"type": "text", "text": reply}]})
        return MockUrllibResponse({"choices": [{"message": {"content": reply}}]})

    monkeypatch.setattr(verifier.urllib.request, "urlopen", fake_urlopen)
    verifier.main()
    return logs_dir / "verifier"


def test_verifier_panel_artifacts_flow_through_collector_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _VERIFIER_ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", f"{O_LABEL},{A_LABEL},{N_LABEL}")
    for key_env, key in _JUDGE_KEYS.values():
        monkeypatch.setenv(key_env, key)

    calls: list[dict[str, Any]] = []
    jobs_dir = tmp_path / "jobs"
    for variant, fixtures in (("with", _with_skill_rewards()), ("without", _without_skill_rewards())):
        job_dir = jobs_dir / f"demo-opencode-{variant}"
        trial_names = []
        for fixture in fixtures:
            trial_name = f"{fixture['entry_id']}__attempt"
            verifier_dir = _run_verifier(
                tmp_path,
                monkeypatch,
                name=f"{variant}-{fixture['entry_id']}",
                fixture=fixture,
                has_skill=variant == "with",
                calls=calls,
            )
            # The verifier adds no panel numbers to Harbor's numeric reward.
            assert set(json.loads((verifier_dir / "reward.json").read_text(encoding="utf-8"))) == {
                *DEFAULT_METRICS,
                "overall",
            }
            shutil.copytree(verifier_dir, job_dir / trial_name / "verifier")
            trial_names.append(trial_name)
        _write_complete_job_result(job_dir, trial_names)

    # Every member judged every metric, and each key reached only its own provider.
    models = {"openai": "gpt-5.6-sol", "anthropic": "claude-opus-5", "nv_build": "nvidia/nemotron-3-super-120b-a12b"}
    assert {(call["provider"], call["metric"]) for call in calls} == {
        (provider, metric) for provider in models for metric in LLM_METRICS
    }
    for call in calls:
        assert call["model"] == models[call["provider"]]
        for provider, (_key_env, key) in _JUDGE_KEYS.items():
            assert (key in call["credential"]) is (provider == call["provider"])

    results_dir = tmp_path / "results"
    result = _collect(jobs_dir, results_dir)

    agent = result["agents"]["opencode"]
    assert result["execution_status"] == "succeeded"
    assert agent["custom_with_skill"] == {}
    assert not (results_dir / "opencode" / "custom_lift.json").exists()
    artifact = json.loads((results_dir / "opencode" / "judge_panel.json").read_text(encoding="utf-8"))
    expected = _panel_report()
    for key in (
        "judges",
        "per_judge",
        "agreement",
        "lift_signs",
        "lift_sign_consistent",
        "same_family_judges",
        "same_family_comparison",
    ):
        assert artifact[key] == expected[key], key
    # The real verifier computes its own deterministic metrics, so only the overall lifts differ.
    for key in ("excluded", "judges", "available", "metrics", "llm_overall", "same_family_gap"):
        assert artifact["lift_excluding_same_family"][key] == expected["lift_excluding_same_family"][key], key
    for key in ("subsets", "metrics", "llm_overall"):
        assert (
            artifact["lift_excluding_same_family"]["reference"][key]
            == expected["lift_excluding_same_family"]["reference"][key]
        ), key
    assert [
        (case["condition"], case["entry_id"], case["metric"], case["spread"]) for case in artifact["disagreement_cases"]
    ] == [
        (case["condition"], case["entry_id"], case["metric"], case["spread"]) for case in expected["disagreement_cases"]
    ]
    assert agent["judge_panel"]["disagreement_case_count"] == 10

    html = _render(tmp_path, results_dir)
    section = _judge_panel_section(html)
    assert _embedded_payload(html)["judge_panel"] == {"opencode": artifact}
    assert 'data-judge-warning="lift-sign"' in section
    assert 'data-judge-warning="same-family"' in section
    assert len(re.findall(r'<details class="t3-judge-case"', section)) == 10


def test_sign_flip_warning_renders_for_dataset_without_assertions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _VERIFIER_ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", f"{O_LABEL},{A_LABEL},{N_LABEL}")
    for key_env, key in _JUDGE_KEYS.values():
        monkeypatch.setenv(key_env, key)

    # No case lists expected behaviors, so the verifier skips behavior_check;
    # gpt-5.6-sol and claude-opus-5 see the skill help on accuracy and goal, nemotron sees it hurt.
    calls: list[dict[str, Any]] = []
    jobs_dir = tmp_path / "jobs"
    for variant, fixtures in zip(("with", "without"), _split_verdict_rewards(behaviors=False), strict=True):
        job_dir = jobs_dir / f"demo-opencode-{variant}"
        trial_names = []
        for fixture in fixtures:
            trial_name = f"{fixture['entry_id']}__attempt"
            verifier_dir = _run_verifier(
                tmp_path,
                monkeypatch,
                name=f"no-assertions-{variant}-{fixture['entry_id']}",
                fixture=fixture,
                has_skill=variant == "with",
                calls=calls,
                expected_behavior=(),
            )
            shutil.copytree(verifier_dir, job_dir / trial_name / "verifier")
            trial_names.append(trial_name)
        _write_complete_job_result(job_dir, trial_names)
    assert {call["metric"] for call in calls} == {"accuracy", "goal_accuracy"}

    results_dir = tmp_path / "results"
    result = _collect(jobs_dir, results_dir)

    artifact = json.loads((results_dir / "opencode" / "judge_panel.json").read_text(encoding="utf-8"))
    assert artifact["lift_signs"] == {O_LABEL: 1, A_LABEL: 1, N_LABEL: -1}
    assert artifact["lift_sign_consistent"] is False
    assert result["agents"]["opencode"]["judge_panel"]["lift_sign_consistent"] is False
    section = _judge_panel_section(_render(tmp_path, results_dir))
    assert 'data-judge-warning="lift-sign"' in section
    llm_overall_lifts = [
        re.findall(r'class="t3-mad-lift"[^>]*>([^<]*)<', row)[-1].strip()
        for row in re.findall(r'<tr data-judge="[^"]+">(.*?)</tr>', section, re.DOTALL)
    ]
    assert llm_overall_lifts == ["+0.67 lift", "+0.67 lift", "-0.67 lift"]


def _tied_accuracy_reward(*, panel: bool) -> dict[str, Any]:
    # Two judges split on ACTION_CORRECT, so the vote is a tie worth 0.5: (1 + 0.5 + 0 + 1 + 1) / 5.
    criteria = {
        "SKILL_IDENTIFIED": True,
        "ACTION_CORRECT": None,
        "FACTUALLY_ACCURATE": False,
        "TASK_ADDRESSED": True,
        "ACTIONABLE": True,
    }
    detail: dict[str, Any] = {"score": 0.7, "reason": "panel vote (2/2 judges)", "criteria": criteria}
    if panel:
        detail["panel"] = {"aggregation": "vote", "quorum": 2, "members": [], "failed_members": 0}
    return {"entry_id": "case-1", "accuracy": 0.7, "details": {"accuracy": detail}}


def test_findings_report_a_tied_panel_criterion_as_tied_not_failed() -> None:
    findings = report._extract_findings([_tied_accuracy_reward(panel=True)])

    accuracy = next(finding for finding in findings if finding["metric"] == "accuracy")
    assert accuracy["reasons"][:2] == ["ACTION_CORRECT tied between judges", "FACTUALLY_ACCURATE failed"]
    assert "ACTION_CORRECT failed" not in accuracy["reasons"]


def test_findings_keep_failed_wording_for_a_null_criterion_without_a_panel() -> None:
    findings = report._extract_findings([_tied_accuracy_reward(panel=False)])

    accuracy = next(finding for finding in findings if finding["metric"] == "accuracy")
    assert accuracy["reasons"][:2] == ["ACTION_CORRECT failed", "FACTUALLY_ACCURATE failed"]


# ---------------------------------------------------------------------------
# Judge rationale in evidence cards and findings
# ---------------------------------------------------------------------------

_INPUT_RANGE = "never validated the input range before computing"
_RANGE_AFTER = "the range check came after the computation"


def _panel_behavior_reward(members: list[tuple[str, str, list[tuple[bool, str]]]]) -> dict[str, Any]:
    """Return a reward whose behavior_check the real ``aggregate_panel`` voted from each member's verdicts."""
    rows = [
        (
            provider,
            model,
            {
                "score": sum(passed for passed, _why in verdicts) / len(verdicts),
                "reason": "summary",
                "results": [
                    {"step": index + 1, "passed": passed, "reason": why} for index, (passed, why) in enumerate(verdicts)
                ],
            },
        )
        for provider, model, verdicts in members
    ]
    detail = aggregate_panel("behavior_check", rows, aggregation="vote", expected_count=len(members[0][2]))
    return {"entry_id": "case-1", "behavior_check": detail["score"], "details": {"behavior_check": detail}}


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (
            [
                ("openai", "gpt-5.6-sol", [(True, "validated the inputs first"), (False, _INPUT_RANGE)]),
                ("anthropic", "claude-opus-5", [(True, "checked the inputs"), (False, "skipped the range check")]),
                ("nv_build", "nvidia/nemotron-3-super-120b-a12b", [(True, "inputs checked"), (True, "range checked")]),
            ],
            [f"1/3 judges observed this behavior: {_INPUT_RANGE}"],
        ),
        (
            [
                ("openai", "gpt-5.6-sol", [(True, "validated the inputs first")]),
                ("anthropic", "claude-opus-5", [(False, _RANGE_AFTER)]),
            ],
            [f"1/2 judges observed this behavior (tied): {_RANGE_AFTER}"],
        ),
    ],
    ids=("majority-failed", "tied"),
)
def test_panel_behavior_evidence_keeps_a_judges_rationale(
    members: list[tuple[str, str, list[tuple[bool, str]]]],
    expected: list[str],
) -> None:
    reward = _panel_behavior_reward(members)

    [card] = tier3_report._metric_evidence("behavior_check", [reward], tier3_report._ReportBudget())
    behavior = next(finding for finding in report._extract_findings([reward]) if finding["metric"] == "behavior_check")

    # The aggregated reason is only the vote tally; the card and the findings add
    # the reasoning of the first judge on the losing side of that behavior.
    assert card["failures"] == expected
    assert behavior["reasons"] == expected


def test_panel_behavior_rationale_is_bounded() -> None:
    reward = _panel_behavior_reward(
        [
            ("openai", "gpt-5.6-sol", [(False, "the agent skipped validation " * 40)]),
            ("anthropic", "claude-opus-5", [(False, "no validation")]),
            ("nv_build", "nvidia/nemotron-3-super-120b-a12b", [(True, "validated")]),
        ]
    )

    [card] = tier3_report._metric_evidence("behavior_check", [reward], tier3_report._ReportBudget())

    [failure] = card["failures"]
    assert failure.startswith("1/3 judges observed this behavior: the agent skipped validation")
    assert len(failure) <= 512


def test_single_judge_behavior_evidence_is_unchanged() -> None:
    detail = {
        "score": 0.5,
        "reason": "summary",
        "results": [
            {"step": 1, "passed": True, "reason": "read the numbers"},
            {"step": 2, "passed": False, "reason": _INPUT_RANGE},
        ],
    }
    reward = {"entry_id": "case-1", "behavior_check": 0.5, "details": {"behavior_check": detail}}

    [card] = tier3_report._metric_evidence("behavior_check", [reward], tier3_report._ReportBudget())
    behavior = next(finding for finding in report._extract_findings([reward]) if finding["metric"] == "behavior_check")

    assert card["notes"] == ["summary"]
    assert card["failures"] == [_INPUT_RANGE]
    assert behavior["reasons"] == [_INPUT_RANGE]
