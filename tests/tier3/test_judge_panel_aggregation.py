# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-model judge panel aggregation in the Harbor verifier.

The first half checks that the verifier's verbatim copy of the panel helpers
behaves exactly like ``eval_core.llm_judge`` on the same inputs. The second
half runs the verifier's ``main()`` with a three-judge panel whose members
answer from per-member scripts, covering the artifact contract, quorum
boundaries, fail-closed configuration errors, per-member time budgets, RAGAS
exclusion, and skipped metrics.
"""

from __future__ import annotations

import copy
import email.message
import importlib.util
import io
import json
import logging
import math
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tests.conftest import MockUrllibResponse

from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.harbor import collector, report
from skillevaluator.tier3.harbor.metrics import extract_custom_metrics, overall_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_TEMPLATE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PROVIDER_URLS = {"openai": OPENAI_URL, "anthropic": ANTHROPIC_URL, "nv_build": NVIDIA_URL}
URL_PROVIDERS = {url: provider for provider, url in PROVIDER_URLS.items()}
MEMBERS = (("openai", "gpt-5.6-sol"), ("anthropic", "claude-opus-5"), ("nv_build", "nvidia/nemotron-3-super-120b-a12b"))
FAMILIES = {"openai": "openai", "anthropic": "anthropic", "nv_build": "nvidia"}
PANEL = ",".join(f"{provider}:{model}" for provider, model in MEMBERS)
KEYS = {
    "OPENAI_API_KEY": "sk-AggregationOpenaiKey0001",
    "ANTHROPIC_API_KEY": "sk-ant-AggregationAnthropicKey0002",
    "NVIDIA_API_KEY": "nvapi-AggregationNvidiaKey0003",
}
METRICS = ("accuracy", "goal_accuracy", "behavior_check")
CRITERIA = ("SKILL_IDENTIFIED", "ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE")
EXPECTED_BEHAVIOR = ["Report that the task is complete", "Stay on the requested task"]
PANEL_BLOCK_KEYS = {
    "aggregation",
    "quorum",
    "members",
    "spread",
    "agreement",
    "disagreement",
    "disagreement_threshold",
    "failed_members",
}
REWARD_KEYS = [
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
    "overall",
]
RICH_KEYS = [
    "security",
    "skill_execution",
    "skill_efficiency",
    "accuracy",
    "goal_accuracy",
    "behavior_check",
    "metric_set",
    "entry_id",
    "has_skill",
    "trajectory_source",
    "details",
]

_ENVIRONMENT = (
    "SKILL_EVAL_LLM_PROVIDER",
    "SKILL_EVAL_LLM_MODEL",
    "SKILL_EVAL_LLM_BASE_URL",
    "SKILL_EVAL_LLM_API_KEY",
    "SKILL_EVAL_JUDGE_MODEL",
    "LLM_JUDGE_MODEL",
    "LLM_JUDGE_FALLBACK_MODELS",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "NVIDIA_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_REGION",
    "SKILL_EVAL_JUDGE_PANEL",
    "SKILL_EVAL_JUDGE_PANEL_AGGREGATION",
    "SKILL_EVAL_JUDGE_PANEL_QUORUM",
    "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT",
    "SKILL_EVAL_LLM_JUDGE_BUDGET_SEC",
    "SKILL_EVAL_LLM_MAX_RETRIES",
    "SKILL_EVAL_LLM_RETRY_BASE_DELAY",
    "SKILL_EVAL_LLM_RETRY_MAX_DELAY",
)


@pytest.fixture(autouse=True)
def _hermetic_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep panel runs independent of credentials and judge settings in the invoking shell."""
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _exec_template(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def template() -> ModuleType:
    """Return one standalone verifier module for pure helper comparisons."""
    return _exec_template("harbor_eval_panel_aggregation_helpers")


# ---------------------------------------------------------------------------
# Behavioral parity: the verifier copy matches eval_core.llm_judge
# ---------------------------------------------------------------------------


def _outcome(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    try:
        return ("returned", function(*copy.deepcopy(args), **copy.deepcopy(kwargs)))
    except Exception as exc:
        return ("raised", type(exc).__name__, str(exc))


def _same_in_both(template: ModuleType, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Call one helper in both modules on identical inputs and require identical outcomes."""
    shared = _outcome(getattr(llm_judge, name), *args, **kwargs)
    copied = _outcome(getattr(template, name), *args, **kwargs)
    assert copied == shared
    return shared


def _accuracy(*failed: str, score: float | None = None, reason: str | None = None) -> dict[str, Any]:
    criteria = {key: key not in failed for key in CRITERIA}
    return {
        "criteria": criteria,
        "score": sum(criteria.values()) / 5 if score is None else score,
        "reason": reason or f"{5 - len(failed)}/5 criteria",
    }


def _goal(achieved: bool, score: float) -> dict[str, Any]:
    return {
        "user_goal": "complete the task",
        "end_state": "done" if achieved else "not done",
        "achieved": achieved,
        "score": score,
        "reason": "goal met" if achieved else "goal missed",
    }


def _behavior(*passed: bool, summary: str | None = None) -> dict[str, Any]:
    return {
        "results": [
            {"step": index + 1, "passed": value, "reason": "seen" if value else "missing"}
            for index, value in enumerate(passed)
        ],
        "score": sum(passed) / len(passed),
        "summary": summary or f"{sum(passed)}/{len(passed)} behaviors",
    }


def _failed(reason: str = "LLM judge error: HTTP 401: Unauthorized") -> dict[str, Any]:
    return {"score": None, "status": "error", "reason": reason}


def _rows(*results: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(provider, model, result) for (provider, model), result in zip(MEMBERS, results, strict=False)]


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({}, id="unset"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "   "}, id="blank"),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": f" OpenAI : gpt-5.6-sol ,{MEMBERS[1][0]}:{MEMBERS[1][1]}"}, id="normalized"
        ),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "bedrock:us.anthropic.claude-opus-5-v1:0"}, id="first-colon-split"),
        pytest.param(
            {
                "SKILL_EVAL_JUDGE_PANEL": PANEL,
                "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "MEDIAN",
                "SKILL_EVAL_JUDGE_PANEL_QUORUM": "1",
                "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.25",
            },
            id="knobs",
        ),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "openai"}, id="missing-colon"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "openai:gpt-5.6-sol,"}, id="trailing-comma"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "mystery:model-x"}, id="unknown-provider"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "openai:gpt-5.6-sol,OPENAI:gpt-5.6-sol"}, id="duplicate"),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": "openai-compatible:model-a,openai-compatible:model-b"}, id="two-gateways"
        ),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": ",".join(f"openai:gpt-{index}" for index in range(6))}, id="six"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_QUORUM": "0"}, id="quorum-zero"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_QUORUM": "4"}, id="quorum-above-n"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_QUORUM": "two"}, id="quorum-word"),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "majority"}, id="aggregation"
        ),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "nan"}, id="nan"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL_QUORUM": "2"}, id="knob-without-panel"),
    ],
)
def test_template_parses_panel_settings_exactly_like_the_shared_module(
    template: ModuleType, environ: dict[str, str]
) -> None:
    outcome = _same_in_both(template, "parse_judge_panel_env", environ)

    if outcome[0] == "returned" and outcome[1] is not None:
        members, aggregation, quorum, threshold = outcome[1]
        assert all(provider == provider.strip().lower() and model == model.strip() for provider, model in members)
        assert aggregation in {"vote", "median", "mean"}
        assert 1 <= quorum <= len(members)
        assert 0.0 <= threshold <= 1.0


def test_template_panel_settings_have_the_contract_values(template: ModuleType) -> None:
    outcome = _same_in_both(template, "parse_judge_panel_env", {"SKILL_EVAL_JUDGE_PANEL": PANEL})

    assert outcome == ("returned", (MEMBERS, "vote", 2, 0.4))


@pytest.mark.parametrize(
    ("provider", "model", "family"),
    [
        ("openai", "gpt-5.6-sol", "openai"),
        ("openai", "o3", "openai"),
        ("openai", "my-finetune", "openai"),
        ("anthropic", "claude-opus-5", "anthropic"),
        ("nv_build", "nvidia/nemotron-3-super-120b-a12b", "nvidia"),
        ("nv_build", "nvidia/llama-3.1-nemotron-70b-instruct", "meta"),
        ("nv_build", "qwen/qwen3-coder-480b-a35b-instruct", "qwen"),
        ("nv_build", "deepseek-ai/deepseek-v3.1", "deepseek"),
        ("nv_build", "mistralai/mixtral-8x22b-instruct", "mistral"),
        ("nv_build", "some-unlisted-model", "unknown"),
        ("bedrock", "us.anthropic.claude-opus-5-v1:0", "anthropic"),
        # Unlisted Bedrock vendors fall back to the model name after the vendor.
        ("bedrock", "zai.glm-4.6", "zhipu"),
        ("bedrock", "us.moonshot.kimi-k2-thinking", "moonshot"),
        ("bedrock", "minimax.minimax-m2", "unknown"),
        ("openai-compatible", "aws/anthropic/bedrock-claude-opus-5", "anthropic"),
        (None, "microsoft/phi-4", "microsoft"),
        (None, "philosopher-7b", "unknown"),
        (None, "", "unknown"),
    ],
)
def test_template_infers_model_families_exactly_like_the_shared_module(
    template: ModuleType, provider: str | None, model: str, family: str
) -> None:
    assert _same_in_both(template, "_model_family", provider, model) == ("returned", family)


def test_vote_takes_the_majority_per_accuracy_criterion(template: ModuleType) -> None:
    outcome = _same_in_both(
        template,
        "aggregate_panel",
        "accuracy",
        _rows(_accuracy(), _accuracy("ACTIONABLE"), _accuracy("SKILL_IDENTIFIED", "ACTIONABLE")),
    )

    result = outcome[1]
    assert result["score"] == 0.8
    assert result["reason"] == "panel vote (3/3 judges)"
    assert result["criteria"] == {key: key != "ACTIONABLE" for key in CRITERIA}
    assert {key: result["panel"][key] for key in ("spread", "agreement", "disagreement", "failed_members")} == {
        "spread": 0.4,
        "agreement": 0.8667,
        "disagreement": True,
        "failed_members": 0,
    }


def test_vote_counts_a_tied_criterion_as_half(template: ModuleType) -> None:
    outcome = _same_in_both(
        template, "aggregate_panel", "accuracy", _rows(_accuracy(), _accuracy("TASK_ADDRESSED", "ACTIONABLE"))
    )

    result = outcome[1]
    assert result["criteria"]["TASK_ADDRESSED"] is None
    assert result["criteria"]["ACTIONABLE"] is None
    assert result["score"] == 0.8


@pytest.mark.parametrize(("aggregation", "score"), [("median", 0.8), ("mean", 0.6667)])
def test_median_and_mean_combine_member_scores(template: ModuleType, aggregation: str, score: float) -> None:
    outcome = _same_in_both(
        template,
        "aggregate_panel",
        "accuracy",
        _rows(_accuracy(), _accuracy("ACTIONABLE"), _accuracy(*CRITERIA[:4])),
        aggregation=aggregation,
    )

    assert outcome[1]["score"] == score
    assert outcome[1]["reason"] == f"panel {aggregation} (3/3 judges)"


def test_behavior_vote_rejects_a_member_with_the_wrong_behavior_count(template: ModuleType) -> None:
    outcome = _same_in_both(
        template,
        "aggregate_panel",
        "behavior_check",
        _rows(_behavior(True, True), _behavior(True, False), _behavior(False, False, True)),
        expected_count=2,
    )

    result = outcome[1]
    assert result["score"] == 0.75
    assert result["results"] == [
        {"step": 1, "passed": True, "reason": "2/2 judges observed this behavior"},
        {"step": 2, "passed": None, "reason": "1/2 judges observed this behavior"},
    ]
    assert result["panel"]["members"][2] == {
        "provider": "nv_build",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "family": "nvidia",
        "status": "error",
        "reason": "behavior result count 3 does not match expected 2",
    }
    assert result["panel"]["failed_members"] == 1


def test_goal_vote_scores_the_majority_side_median(template: ModuleType) -> None:
    outcome = _same_in_both(
        template,
        "aggregate_panel",
        "goal_accuracy",
        _rows({**_goal(True, 1.0), "method": "custom"}, _goal(True, 0.8), _goal(False, 0.0)),
    )

    assert {key: outcome[1][key] for key in ("score", "achieved", "method")} == {
        "score": 0.9,
        "achieved": True,
        "method": "custom",
    }


def test_goal_vote_tie_is_undecided_at_half(template: ModuleType) -> None:
    outcome = _same_in_both(template, "aggregate_panel", "goal_accuracy", _rows(_goal(True, 1.0), _goal(False, 0.0)))

    assert {key: outcome[1][key] for key in ("score", "achieved")} == {"score": 0.5, "achieved": None}


@pytest.mark.parametrize(
    ("results", "quorum", "scored", "reason"),
    [
        pytest.param((_accuracy(), _accuracy(), _failed()), None, True, "panel vote (2/3 judges; 1 failed)", id="met"),
        pytest.param(
            (_accuracy(), _failed(), _failed()),
            None,
            False,
            "Judge panel quorum not met for accuracy: 1/3 judges succeeded (quorum 2)",
            id="one-short",
        ),
        pytest.param(
            (_failed(), _failed(), _failed()),
            None,
            False,
            "Judge panel quorum not met for accuracy: 0/3 judges succeeded (quorum 2)",
            id="all-failed",
        ),
        pytest.param(
            (_accuracy(), _accuracy(), _failed()),
            3,
            False,
            "Judge panel quorum not met for accuracy: 2/3 judges succeeded (quorum 3)",
            id="quorum-n",
        ),
        pytest.param(
            (_accuracy(score=1.5), {"score": math.nan}, {"score": True}),
            1,
            False,
            "Judge panel quorum not met for accuracy: 0/3 judges succeeded (quorum 1)",
            id="invalid-scores",
        ),
    ],
)
def test_quorum_boundaries_match_the_shared_module(
    template: ModuleType,
    results: tuple[dict[str, Any], ...],
    quorum: int | None,
    scored: bool,
    reason: str,
) -> None:
    outcome = _same_in_both(template, "aggregate_panel", "accuracy", _rows(*results), quorum=quorum)

    result = outcome[1]
    assert result["reason"] == reason
    if scored:
        assert result["score"] == 1.0
        assert "status" not in result
    else:
        assert result["score"] is None
        assert result["status"] == "error"
    assert set(result["panel"]) == PANEL_BLOCK_KEYS


# ---------------------------------------------------------------------------
# main() with a three-judge panel
# ---------------------------------------------------------------------------

# Per-member scripts: openai is generous, anthropic is in the middle, nv_build is strict.
SCRIPT: dict[str, dict[str, Any]] = {
    "openai": {"accuracy": _accuracy(), "goal_accuracy": _goal(True, 1.0), "behavior_check": _behavior(True, True)},
    "anthropic": {
        "accuracy": _accuracy("ACTIONABLE"),
        "goal_accuracy": _goal(True, 0.8),
        "behavior_check": _behavior(True, False),
    },
    "nv_build": {
        "accuracy": _accuracy("SKILL_IDENTIFIED", "ACTIONABLE"),
        "goal_accuracy": _goal(False, 0.0),
        "behavior_check": _behavior(False, False),
    },
}


def _load_verifier(tmp_path: Path, **entry_overrides: Any) -> ModuleType:
    """Load the standalone verifier against a temporary Harbor workspace."""
    module = _exec_template(f"harbor_eval_panel_aggregation_{tmp_path.name}")
    agent_dir = tmp_path / "logs" / "agent"
    verifier_dir = tmp_path / "logs" / "verifier"
    tests_dir = tmp_path / "tests"
    for directory in (agent_dir, verifier_dir, tests_dir):
        directory.mkdir(parents=True)
    module.LOGS_DIR = tmp_path / "logs"
    module.AGENT_LOGS_DIR = agent_dir
    module.VERIFIER_DIR = verifier_dir
    module.TESTS_DIR = tests_dir
    module.ATIF_PATH = agent_dir / "trajectory.json"
    module.ENTRY_PATH = tests_dir / "entry.json"
    module.REWARD_JSON = verifier_dir / "reward.json"
    module.REWARD_TXT = verifier_dir / "reward.txt"
    module.SKILL_EVALUATOR_REWARD_JSON = verifier_dir / "skill_evaluator_reward.json"
    module.ATIF_PATH.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "Complete the task."},
                    {"source": "agent", "message": "The task is complete."},
                ]
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "id": "panel-aggregation-case",
        "question": "Complete the task.",
        "ground_truth": "The task is complete.",
        "expected_behavior": EXPECTED_BEHAVIOR,
        "should_trigger": False,
        "evaluated_skill": "demo",
        "has_skill": True,
        **entry_overrides,
    }
    module.ENTRY_PATH.write_text(json.dumps(entry), encoding="utf-8")
    return module


def _judge_kind(request: urllib.request.Request) -> str:
    prompt = json.loads(request.data)["messages"][0]["content"]
    if "SKILL_IDENTIFIED" in prompt:
        return "accuracy"
    if "EXPECTED BEHAVIORS" in prompt:
        return "behavior_check"
    assert "Did the agent achieve the expected goal?" in prompt
    return "goal_accuracy"


def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "Error", hdrs=email.message.Message(), fp=io.BytesIO(body.encode()))


class _ScriptedPanel:
    """Answer each member's request from its script; an exception entry is raised instead."""

    def __init__(self, script: dict[str, dict[str, Any]]) -> None:
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        provider = URL_PROVIDERS[request.full_url]
        kind = _judge_kind(request)
        self.calls.append((provider, kind))
        answer = self.script[provider][kind]
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(request, timeout)
        text = json.dumps(answer)
        if provider == "anthropic":
            return MockUrllibResponse({"content": [{"type": "text", "text": text}]})
        return MockUrllibResponse({"choices": [{"message": {"content": text}}]})


def _with(**changes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy SCRIPT with some (provider -> {metric: answer}) entries replaced."""
    script = copy.deepcopy(SCRIPT)
    for provider, answers in changes.items():
        script[provider].update(answers)
    return script


def _unauthorized(provider: str) -> urllib.error.HTTPError:
    return _http_error(PROVIDER_URLS[provider], 401, '{"error": "invalid api key"}')


def _configure_panel(monkeypatch: pytest.MonkeyPatch, verifier: ModuleType, script: dict[str, Any]) -> _ScriptedPanel:
    for variable, key in KEYS.items():
        monkeypatch.setenv(variable, key)
    # A canonical OpenAI primary would enable RAGAS for a single judge; the panel must not use it.
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    monkeypatch.setenv("SKILL_EVAL_LLM_MAX_RETRIES", "0")
    transport = _ScriptedPanel(script)
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)
    return transport


def _artifacts(verifier: ModuleType) -> tuple[dict[str, Any], dict[str, Any], str]:
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    return rich, numeric, verifier.REWARD_TXT.read_text(encoding="utf-8")


def _keys_named(value: Any, name: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == name:
                found.append(nested)
            found.extend(_keys_named(nested, name))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_keys_named(nested, name))
    return found


def test_panel_aggregates_member_verdicts_into_details_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Score each LLM metric by panel vote and keep every panel field under details."""
    verifier = _load_verifier(tmp_path)
    transport = _configure_panel(monkeypatch, verifier, SCRIPT)

    verifier.main()

    rich, numeric, reward_txt = _artifacts(verifier)
    assert transport.calls == [(provider, metric) for metric in METRICS for provider, _model in MEMBERS]
    assert list(numeric) == REWARD_KEYS
    assert numeric == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 0.8,
        "goal_accuracy": 0.9,
        "behavior_check": 0.5,
        "overall": 0.8667,
    }
    assert reward_txt == "0.8667"
    assert list(rich) == RICH_KEYS
    assert "evaluation_status" not in rich

    merged = dict(numeric)
    collector._merge_reward_sidecars(merged, verifier.VERIFIER_DIR)
    assert extract_custom_metrics(numeric) == {}
    assert extract_custom_metrics(merged) == {}
    assert overall_score(merged) == pytest.approx(0.8667, abs=1e-4)
    findings = report._extract_findings([merged])
    assert all(isinstance(reason, str) for finding in findings for reason in finding["reasons"])

    details = rich["details"]
    for metric in METRICS:
        assert set(details[metric]["panel"]) == PANEL_BLOCK_KEYS
        assert details[metric]["reason"] == "panel vote (3/3 judges)"
        assert details[metric]["panel"]["aggregation"] == "vote"
        assert details[metric]["panel"]["quorum"] == 2
        assert details[metric]["panel"]["disagreement_threshold"] == 0.4
        assert details[metric]["panel"]["failed_members"] == 0
        assert "omitted" in details[metric]
        members = details[metric]["panel"]["members"]
        assert [(member["provider"], member["model"], member["family"]) for member in members] == [
            (provider, model, FAMILIES[provider]) for provider, model in MEMBERS
        ]
        assert all(member["status"] == "ok" for member in members)

    assert details["accuracy"]["criteria"] == {key: key != "ACTIONABLE" for key in CRITERIA}
    assert details["accuracy"]["panel"]["members"][1] == {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "family": "anthropic",
        "status": "ok",
        "score": 0.8,
        "reason": "4/5 criteria",
        "criteria": {key: key != "ACTIONABLE" for key in CRITERIA},
    }
    assert {key: details["accuracy"]["panel"][key] for key in ("spread", "agreement", "disagreement")} == {
        "spread": 0.4,
        "agreement": 0.8667,
        "disagreement": True,
    }
    assert {key: details["goal_accuracy"][key] for key in ("score", "achieved", "method")} == {
        "score": 0.9,
        "achieved": True,
        "method": "custom",
    }
    assert [(member["achieved"], member["method"]) for member in details["goal_accuracy"]["panel"]["members"]] == [
        (True, "custom"),
        (True, "custom"),
        (False, "custom"),
    ]
    assert details["behavior_check"]["results"] == [
        {"step": 1, "passed": True, "reason": "2/3 judges observed this behavior"},
        {"step": 2, "passed": False, "reason": "1/3 judges observed this behavior"},
    ]
    assert details["behavior_check"]["panel"]["members"][2]["results"] == [
        {"step": 1, "passed": False, "reason": "missing"},
        {"step": 2, "passed": False, "reason": "missing"},
    ]
    assert verifier._ACTIVE_JUDGE_TARGET.get() is None


def test_panel_honors_the_configured_aggregation_and_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply SKILL_EVAL_JUDGE_PANEL_AGGREGATION, _QUORUM, and _DISAGREEMENT from the verifier environment."""
    verifier = _load_verifier(tmp_path)
    _configure_panel(monkeypatch, verifier, SCRIPT)
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_AGGREGATION", "mean")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_QUORUM", "3")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT", "0.5")

    verifier.main()

    rich, numeric, _reward_txt = _artifacts(verifier)
    assert {metric: numeric[metric] for metric in METRICS} == {
        "accuracy": 0.8,
        "goal_accuracy": 0.6,
        "behavior_check": 0.5,
    }
    assert {
        metric: (rich["details"][metric]["reason"], rich["details"][metric]["panel"]["disagreement"])
        for metric in METRICS
    } == {
        "accuracy": ("panel mean (3/3 judges)", False),
        "goal_accuracy": ("panel mean (3/3 judges)", True),
        "behavior_check": ("panel mean (3/3 judges)", True),
    }
    assert {rich["details"][metric]["panel"]["quorum"] for metric in METRICS} == {3}


def test_panel_scores_when_the_quorum_is_exactly_met(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Score every metric from two of three judges and record the failed member."""
    verifier = _load_verifier(tmp_path)
    _configure_panel(monkeypatch, verifier, _with(nv_build={metric: _unauthorized("nv_build") for metric in METRICS}))

    verifier.main()

    rich, numeric, _reward_txt = _artifacts(verifier)
    assert {metric: numeric[metric] for metric in METRICS} == {
        "accuracy": 0.9,
        "goal_accuracy": 0.9,
        "behavior_check": 0.75,
    }
    for metric in METRICS:
        panel = rich["details"][metric]["panel"]
        assert rich["details"][metric]["reason"] == "panel vote (2/3 judges; 1 failed)"
        assert panel["failed_members"] == 1
        assert panel["members"][2]["status"] == "error"
        assert panel["members"][2]["reason"].startswith("LLM judge error: HTTP 401")
        assert set(panel["members"][2]) == {"provider", "model", "family", "status", "reason"}
    assert rich["details"]["accuracy"]["criteria"]["ACTIONABLE"] is None


def test_panel_one_judge_short_of_quorum_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail only the metric below quorum and keep reward.json deliberately incomplete."""
    verifier = _load_verifier(tmp_path)
    _configure_panel(
        monkeypatch,
        verifier,
        _with(
            anthropic={"accuracy": _unauthorized("anthropic")},
            nv_build={"accuracy": _unauthorized("nv_build")},
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    rich, numeric, reward_txt = _artifacts(verifier)
    reason = "Judge panel quorum not met for accuracy: 1/3 judges succeeded (quorum 2)"
    assert rich["accuracy"] is None
    assert rich["evaluation_status"] == "failed"
    assert rich["evaluation_errors"] == {"accuracy": reason}
    assert rich["details"]["accuracy"]["score"] is None
    assert rich["details"]["accuracy"]["status"] == "error"
    assert rich["details"]["accuracy"]["reason"] == reason
    assert rich["details"]["accuracy"]["panel"]["failed_members"] == 2
    assert numeric == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "goal_accuracy": 0.9,
        "behavior_check": 0.5,
        "overall": 0.0,
    }
    assert reward_txt == "0.0"
    assert overall_score(numeric) is None


@pytest.mark.parametrize("quorum", [None, "3"], ids=["all-members-failing", "quorum-equals-members"])
def test_panel_fails_every_metric_closed_when_no_metric_reaches_quorum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quorum: str | None
) -> None:
    """Fail every LLM metric closed when all judges fail, or when quorum N meets one failure."""
    verifier = _load_verifier(tmp_path)
    failing = ("openai", "anthropic", "nv_build") if quorum is None else ("nv_build",)
    _configure_panel(
        monkeypatch,
        verifier,
        _with(**{provider: {metric: _unauthorized(provider) for metric in METRICS} for provider in failing}),
    )
    if quorum is not None:
        monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_QUORUM", quorum)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    rich, numeric, reward_txt = _artifacts(verifier)
    succeeded = 3 - len(failing)
    assert rich["evaluation_errors"] == {
        metric: (
            f"Judge panel quorum not met for {metric}: {succeeded}/3 judges succeeded (quorum {3 if quorum else 2})"
        )
        for metric in METRICS
    }
    assert all(rich[metric] is None for metric in METRICS)
    assert numeric == {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "overall": 0.0}
    assert reward_txt == "0.0"
    assert rich["details"]["behavior_check"]["results"] == []
    assert rich["details"]["goal_accuracy"]["method"] == "custom"


def test_panel_never_uses_ragas_even_for_a_canonical_openai_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the custom goal prompt for every member although RAGAS would serve a single OpenAI judge."""
    verifier = _load_verifier(tmp_path)
    for variable, key in KEYS.items():
        monkeypatch.setenv(variable, key)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    assert verifier._ragas_goal_accuracy_enabled() is True  # control: the single judge would use RAGAS here

    _configure_panel(monkeypatch, verifier, SCRIPT)
    ragas_calls: list[tuple[Any, ...]] = []

    def ragas_spy(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        ragas_calls.append(args)
        return {"score": 1.0, "reason": "RAGAS", "method": "ragas", "provider": "openai", "model": "gpt-5.6-sol"}

    monkeypatch.setattr(verifier, "_judge_goal_accuracy_ragas", ragas_spy)

    assert verifier._ragas_goal_accuracy_enabled() is False
    token = verifier._ACTIVE_JUDGE_TARGET.set(verifier.JudgeTarget("openai", "gpt-5.6-sol"))
    try:
        monkeypatch.delenv("SKILL_EVAL_JUDGE_PANEL")
        assert verifier._ragas_goal_accuracy_enabled() is False
    finally:
        verifier._ACTIVE_JUDGE_TARGET.reset(token)
        monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)

    verifier.main()

    rich, _numeric, _reward_txt = _artifacts(verifier)
    assert ragas_calls == []
    assert [member["method"] for member in rich["details"]["goal_accuracy"]["panel"]["members"]] == ["custom"] * 3
    assert "RAGAS" not in json.dumps(rich)


def test_each_member_gets_its_own_judge_time_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spend one member's whole budget without leaving later members an expired deadline."""
    verifier = _load_verifier(tmp_path)
    elapsed = [0.0]

    class _FakeTime:
        def monotonic(self) -> float:
            return elapsed[0]

        def sleep(self, seconds: float) -> None:
            elapsed[0] += seconds

        def __getattr__(self, name: str) -> Any:
            return getattr(time, name)

    def exhaust_budget(_request: urllib.request.Request, timeout: float) -> None:
        elapsed[0] += timeout
        raise urllib.error.URLError("timed out")

    _configure_panel(monkeypatch, verifier, _with(anthropic=dict.fromkeys(METRICS, exhaust_budget)))
    monkeypatch.setenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", "30")
    monkeypatch.setattr(verifier, "time", _FakeTime())

    verifier.main()

    rich, numeric, _reward_txt = _artifacts(verifier)
    assert elapsed[0] == pytest.approx(90.0)
    for metric in METRICS:
        members = rich["details"][metric]["panel"]["members"]
        assert [member["status"] for member in members] == ["ok", "error", "ok"]
        assert "timed out" in members[1]["reason"]
    assert {metric: numeric[metric] for metric in METRICS} == {
        "accuracy": 0.8,
        "goal_accuracy": 0.5,
        "behavior_check": 0.5,
    }


def test_a_hanging_member_is_interrupted_without_failing_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt a member whose response body stalls, on its own SIGALRM budget, then run the next member."""
    if (
        not hasattr(signal, "setitimer")
        or signal.getitimer(signal.ITIMER_REAL)[0] > 0
        or threading.current_thread() is not threading.main_thread()
    ):
        pytest.skip("free POSIX interval timer on the main thread required")
    verifier = _load_verifier(tmp_path)

    class _StalledBody:
        def __enter__(self) -> _StalledBody:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def read(self) -> bytes:
            time.sleep(10)
            return b"{}"

    _configure_panel(monkeypatch, verifier, _with(anthropic={"accuracy": lambda _request, _timeout: _StalledBody()}))
    monkeypatch.setenv("SKILL_EVAL_LLM_JUDGE_BUDGET_SEC", "0.3")
    previous_handler = signal.getsignal(signal.SIGALRM)

    started = time.monotonic()
    verifier.main()

    assert time.monotonic() - started < 5
    rich, numeric, _reward_txt = _artifacts(verifier)
    members = rich["details"]["accuracy"]["panel"]["members"]
    assert [member["status"] for member in members] == ["ok", "error", "ok"]
    assert "time budget exhausted" in members[1]["reason"]
    # openai and nv_build tie on SKILL_IDENTIFIED and ACTIONABLE: (3 + 0.5 + 0.5) / 5.
    assert numeric["accuracy"] == 0.8
    assert numeric["goal_accuracy"] == 0.9
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "openai"}, "must have the form provider:model", id="malformed"),
        pytest.param({"SKILL_EVAL_JUDGE_PANEL": "mystery:model-x"}, "unsupported provider", id="unknown-provider"),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": "openai:gpt-5.6-sol,openai:gpt-5.6-sol"}, "more than once", id="duplicate"
        ),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": PANEL, "SKILL_EVAL_JUDGE_PANEL_QUORUM": "4"},
            "must be an integer from 1 to 3",
            id="quorum-out-of-range",
        ),
        pytest.param(
            {"SKILL_EVAL_JUDGE_PANEL": "", "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "median"},
            "set without SKILL_EVAL_JUDGE_PANEL",
            id="knob-without-panel",
        ),
    ],
)
def test_invalid_panel_configuration_fails_every_llm_metric_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    message: str,
) -> None:
    """Fail closed without any LLM call, never falling back to the single judge."""
    verifier = _load_verifier(tmp_path)
    transport = _configure_panel(monkeypatch, verifier, SCRIPT)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    assert transport.calls == []
    rich, numeric, reward_txt = _artifacts(verifier)
    assert set(rich["evaluation_errors"]) == set(METRICS)
    for metric in METRICS:
        reason = rich["evaluation_errors"][metric]
        assert reason.startswith("Invalid judge panel configuration: ")
        assert message in reason
        assert rich["details"][metric]["status"] == "error"
        assert rich["details"][metric]["score"] is None
        assert "panel" not in rich["details"][metric]
    assert numeric == {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "overall": 0.0}
    assert reward_txt == "0.0"


@pytest.mark.parametrize("panel", [PANEL, "not a panel"], ids=["valid-panel", "invalid-panel"])
def test_skipped_metrics_make_no_llm_calls_and_carry_no_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: str
) -> None:
    """Return the documented skip results without any judge call when there is nothing to judge."""
    verifier = _load_verifier(tmp_path, ground_truth="", expected_behavior=[])
    transport = _configure_panel(monkeypatch, verifier, SCRIPT)
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", panel)

    verifier.main()

    assert transport.calls == []
    rich, numeric, _reward_txt = _artifacts(verifier)
    assert _keys_named(rich, "panel") == []
    assert "evaluation_status" not in rich
    assert {metric: numeric[metric] for metric in METRICS} == dict.fromkeys(METRICS, 1.0)
    assert rich["details"]["accuracy"]["reason"] == "No ground_truth -- skipped"
    assert rich["details"]["goal_accuracy"]["reason"] == "No ground_truth -- skipped"
    assert rich["details"]["behavior_check"]["reason"] == "No expected_behavior defined"


def test_only_judged_metrics_carry_a_panel_when_some_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip accuracy and goal without ground truth while the panel still judges behaviors."""
    verifier = _load_verifier(tmp_path, ground_truth="")
    transport = _configure_panel(monkeypatch, verifier, SCRIPT)

    verifier.main()

    assert transport.calls == [(provider, "behavior_check") for provider, _model in MEMBERS]
    rich, numeric, _reward_txt = _artifacts(verifier)
    assert "panel" not in rich["details"]["accuracy"]
    assert "panel" not in rich["details"]["goal_accuracy"]
    assert rich["details"]["behavior_check"]["panel"]["failed_members"] == 0
    assert numeric["behavior_check"] == 0.5


def test_panel_logs_and_artifacts_never_hold_member_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Log one summary per metric and keep echoed keys out of artifacts, including truncated prefixes."""
    verifier = _load_verifier(tmp_path)
    leaked = KEYS["NVIDIA_API_KEY"]
    echo = "x" * 495 + leaked
    _configure_panel(
        monkeypatch,
        verifier,
        _with(
            nv_build={
                "accuracy": _accuracy("SKILL_IDENTIFIED", "ACTIONABLE", reason=echo),
                "behavior_check": _behavior(False, False, summary=echo),
            }
        ),
    )
    caplog.set_level(logging.INFO)

    verifier.main()

    summaries = [record.getMessage() for record in caplog.records if record.getMessage().startswith("Judge panel ")]
    assert len(summaries) == 3
    assert all(f"{provider}:{model}=ok" in summary for summary in summaries for provider, model in MEMBERS)
    artifacts = "".join(
        path.read_text(encoding="utf-8")
        for path in (verifier.SKILL_EVALUATOR_REWARD_JSON, verifier.REWARD_JSON, verifier.REWARD_TXT)
    )
    for key in KEYS.values():
        assert key not in caplog.text
        assert key not in artifacts
        # Redaction precedes the panel's 512-character bound, so no credential prefix survives truncation.
        assert key[:12] not in artifacts
    members = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))["details"]
    assert members["behavior_check"]["panel"]["members"][2]["reason"].startswith("x" * 495 + "[REDACTED")


def test_a_raising_aggregation_still_writes_fail_closed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn an unexpected aggregation error into a judged failure instead of a verifier crash."""
    verifier = _load_verifier(tmp_path)
    _configure_panel(monkeypatch, verifier, SCRIPT)

    def broken_aggregation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("aggregation exploded")

    monkeypatch.setattr(verifier, "aggregate_panel", broken_aggregation)

    with pytest.raises(SystemExit) as exc_info:
        verifier.main()

    assert exc_info.value.code == 1
    rich, numeric, reward_txt = _artifacts(verifier)
    assert rich["evaluation_errors"] == {
        metric: f"Judge panel aggregation for {metric} raised RuntimeError: aggregation exploded" for metric in METRICS
    }
    assert numeric == {"security": 1.0, "skill_execution": 1.0, "skill_efficiency": 1.0, "overall": 0.0}
    assert reward_txt == "0.0"
    assert verifier._ACTIVE_JUDGE_TARGET.get() is None
