# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-model judge panel wiring through ``run_harbor_eval`` with only Harbor's edges mocked.

Task staging, the Harbor command builder, and the NVIDIA Build stdin handoff run for real;
``subprocess.run`` captures each Harbor invocation instead of launching containers.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from harbor.utils.env import resolve_env_vars

from skillevaluator.provider_config import (
    JUDGE_PANEL_ENV,
    ProviderConfig,
    resolve_judge_panel_config,
    resolve_llm_provider,
)
from skillevaluator.tier3.harbor import runner, runtime_preflight
from skillevaluator.tier3.harbor.adapter import _runtime_env_toml_block
from skillevaluator.tier3.harbor.secure_docker_environment import SECURE_DOCKER_ENV_IMPORT_PATH
from skillevaluator.tier3.harbor.sensitive_stdin import NVIDIA_BUILD_KEY_STDIN_ENV, NVIDIA_BUILD_STDIN_SENTINEL

NVIDIA_KEY = "nvapi-primary-secret-0001"
OPENAI_KEY = "sk-openai-member-secret-0002"
ANTHROPIC_KEY = "sk-ant-member-secret-0003"
GATEWAY_KEY = "gateway-primary-secret-0004"
GATEWAY_URL = "https://gateway.example/v1"
AWS_ACCESS_KEY = "AKIAMEMBERACCESS0005"
AWS_SECRET_KEY = "aws-member-secret-access-0006"
AGENT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NV_JUDGE = "nv_build:nvidia/llama-3.3-nemotron-super-49b-v1"
PANEL = f"openai:gpt-5.6-sol,anthropic:claude-opus-5,{NV_JUDGE}"
ALIAS = "SKILLEVALUATOR_JUDGE_PANEL__"
PANEL_SETTINGS_JOB_ENV = {
    "SKILL_EVAL_JUDGE_PANEL": "${SKILL_EVAL_JUDGE_PANEL}",
    "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "${SKILL_EVAL_JUDGE_PANEL_AGGREGATION}",
    "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "${SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT}",
    "SKILL_EVAL_JUDGE_PANEL_QUORUM": "${SKILL_EVAL_JUDGE_PANEL_QUORUM}",
}
MEMBER_SECRETS = (OPENAI_KEY, ANTHROPIC_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY)
_SCRUBBED_PREFIXES = (
    "SKILL_EVAL_",
    "OPENAI_",
    "ANTHROPIC_",
    "NVIDIA_",
    "AWS_",
    "LLM_JUDGE",
    "SKILLEVALUATOR_JUDGE_PANEL",
    "SKILLEVALUATOR_NVIDIA",
)
_CATALOG_DEGRADED = "model catalog access does not verify runtime credentials for this endpoint"


@dataclass(frozen=True)
class _HarborCall:
    command: list[str]
    env: dict[str, str]
    stdin: str | None

    @property
    def verifier_env(self) -> dict[str, str]:
        """Return Harbor's job-level ``--verifier-env`` assignments."""
        assignments = [self.command[index + 1] for index, part in enumerate(self.command) if part == "--verifier-env"]
        return dict(assignment.split("=", 1) for assignment in assignments)


class _RecordingReporter:
    def __init__(self) -> None:
        self.plans: list[Any] = []
        self.events: list[Any] = []
        self.secret_calls: list[set[str]] = []
        self.closed = False

    @property
    def is_active(self) -> bool:
        return bool(self.plans) and not self.closed

    def start(self, plan: Any) -> None:
        self.plans.append(plan)

    def set_secret_values(self, values: Any) -> None:
        self.secret_calls.append(set(values))

    def emit(self, event: Any) -> None:
        self.events.append(event)

    def heartbeat(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    def stage_events(self, stage: str) -> list[Any]:
        return [event for event in self.events if event.stage == stage]


ProbeOverride = Callable[[ProviderConfig], runtime_preflight.ModelProbeResult]


class _Harness:
    """Run the real Tier 3 engine with Harbor, probes, collection, and HTML rendering mocked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        self.harbor_calls: list[_HarborCall] = []
        self.emitted: list[dict[str, Any]] = []
        self.probed: list[ProviderConfig] = []
        self.probe_overrides: dict[str, ProbeOverride] = {}
        self.reporter = _RecordingReporter()
        self.skill = tmp_path / "demo-skill"
        (self.skill / "evals").mkdir(parents=True)
        (self.skill / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n",
            encoding="utf-8",
        )
        (self.skill / "evals" / "evals.json").write_text(
            json.dumps([{"id": "case-001", "question": "Q?", "expected_answer": "ok", "files": []}]),
            encoding="utf-8",
        )
        for name in list(os.environ):
            if name.startswith(_SCRUBBED_PREFIXES):
                monkeypatch.delenv(name)
        self._install_edges()

    def _install_edges(self) -> None:
        real_run = subprocess.run

        def harbor_run(command: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(command, list) and command[1:2] == ["run"] and "--job-name" in command:
                self.harbor_calls.append(_HarborCall(list(command), dict(kwargs["env"]), kwargs.get("input")))
                return subprocess.CompletedProcess(command, 0, "", "")
            return real_run(command, *args, **kwargs)

        real_generate = runner.generate_harbor_tasks
        real_stage_native = runner.stage_native_harbor_tasks

        def generate(skill_path: Path, output_dir: Path, **kwargs: Any) -> list[Path]:
            self.emitted.append({"output_dir": output_dir, **kwargs})
            return real_generate(skill_path, output_dir, **kwargs)

        def stage_native(skill_path: Path, output_dir: Path, **kwargs: Any) -> list[Path]:
            self.emitted.append({"output_dir": output_dir, **kwargs})
            return real_stage_native(skill_path, output_dir, **kwargs)

        def probe(selected_provider: ProviderConfig) -> runtime_preflight.ModelProbeResult:
            self.probed.append(selected_provider)
            override = self.probe_overrides.get(f"{selected_provider.provider}:{selected_provider.model}")
            if override is not None:
                return override(selected_provider)
            return runtime_preflight.ModelProbeResult(
                True,
                selected_provider.provider,
                selected_provider.model,
                f"model {selected_provider.model} is available",
            )

        def render_report(_skill_path: Path, run_dir: Path, **_kwargs: Any) -> Path:
            report = run_dir / "report.html"
            report.write_text("<html></html>\n", encoding="utf-8")
            return report

        self.monkeypatch.setattr(runner.subprocess, "run", harbor_run)
        self.monkeypatch.setattr(runner, "generate_harbor_tasks", generate)
        self.monkeypatch.setattr(runner, "stage_native_harbor_tasks", stage_native)
        self.monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
        self.monkeypatch.setattr(runner, "_validate_harbor_job_result", lambda *_args, **_kwargs: (True, ""))
        self.monkeypatch.setattr(
            runner,
            "collect_harbor_results",
            lambda **_kwargs: {"execution_status": "complete", "execution_errors": [], "metrics": [], "agents": {}},
        )
        self.monkeypatch.setattr(runner, "render_agent_eval_html_report", render_report)
        self.monkeypatch.setattr(runtime_preflight, "probe_model", probe)

    def run(
        self,
        environment: Mapping[str, str],
        *,
        agents: tuple[str, ...] = ("opencode",),
        **kwargs: Any,
    ) -> dict[str, Any]:
        for name, value in environment.items():
            self.monkeypatch.setenv(name, value)
        kwargs.setdefault("skip_baseline", True)
        return runner.run_harbor_eval(
            self.skill,
            list(agents),
            output_dir=self.tmp_path / "results",
            env_mode="docker",
            keep_harbor_jobs=True,
            agent_runtime_preflight=False,
            progress_reporter=self.reporter,
            **kwargs,
        )

    def staged_task_tomls(self, result: Mapping[str, Any]) -> list[str]:
        tasks_dir = Path(result["run_dir"]) / "_harbor-tasks"
        return [path.read_text(encoding="utf-8") for path in sorted(tasks_dir.rglob("case-001/task.toml"))]

    def single_call(self) -> _HarborCall:
        (call,) = self.harbor_calls
        return call


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    return _Harness(monkeypatch, tmp_path)


def _nv_build_primary(**extra: str) -> dict[str, str]:
    return {"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": NVIDIA_KEY, **extra}


def _panel_environment(panel: str = PANEL, **extra: str) -> dict[str, str]:
    return _nv_build_primary(
        OPENAI_API_KEY=OPENAI_KEY,
        ANTHROPIC_API_KEY=ANTHROPIC_KEY,
        **{JUDGE_PANEL_ENV: panel},
        **extra,
    )


def _resolve_verifier_env(
    monkeypatch: pytest.MonkeyPatch,
    task_toml: str,
    call: _HarborCall,
) -> dict[str, str]:
    """Resolve task-level and job-level verifier env exactly as Harbor's verifier does."""
    task_env = tomllib.loads(task_toml)["verifier"]["env"]
    with monkeypatch.context() as harbor_process:
        harbor_process.setattr(os, "environ", dict(call.env))
        return resolve_env_vars({**task_env, **call.verifier_env})


def _names_holding(environment: Mapping[str, str], value: str) -> set[str]:
    return {name for name, candidate in environment.items() if candidate == value}


def _base_harbor_env() -> dict[str, str]:
    names = runner._HARBOR_BASE_ENV_VARS | runner._HARBOR_ENV_MODE_VARS["docker"]
    return {name: os.environ[name] for name in names if os.environ.get(name)}


# --- Compatibility boundary --------------------------------------------------------------------------

_GOLDEN_NO_PANEL_TASK_HEAD = (
    'schema_version = "1.3"\n\n[task]\nname = "nvidia/skillevaluator-case-001"\n'
    'description = "Skill evaluation task for none"\n\n[metadata]\nskill = "none"\nentry_id = "case-001"\n'
    "has_skill = true\n\n[agent]\ntimeout_sec = 300.0\n\n[verifier]\ntimeout_sec = 600.0\n\n[verifier.env]\n"
    'NVIDIA_API_KEY = "${NVIDIA_API_KEY}"\nSKILL_EVAL_LLM_MODEL = "${SKILL_EVAL_LLM_MODEL}"\n'
    'SKILL_EVAL_LLM_PROVIDER = "${SKILL_EVAL_LLM_PROVIDER}"\n\n'
    '[environment]\ncpus = 2\nmemory_mb = 4096\nstorage_mb = 2048\nnetwork_mode = "public"\n'
    'skills_dir = "/workspace/skills"\n'
)


def test_no_panel_run_is_identical_to_the_single_judge_baseline(harness: _Harness) -> None:
    result = harness.run(_nv_build_primary())

    assert "error" not in result
    run_config = result["run_config"]
    assert run_config["judge"] == {
        "enabled": True,
        "provider": "nv_build",
        "model": AGENT_MODEL,
        "source": "provider default",
        "override_applied": False,
        "catalog_verification": "degraded",
    }
    assert run_config["credential_validation"] == {
        "status": "degraded",
        "targets": [
            {
                "labels": ["opencode", "standard grader"],
                "provider": "nv_build",
                "model": AGENT_MODEL,
                "status": "degraded",
                "detail": _CATALOG_DEGRADED,
            }
        ],
    }
    run_dir = Path(result["run_dir"])
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) == run_config

    call = harness.single_call()
    assert call.command == [
        runner._harbor_bin(),
        "run",
        "--job-name",
        "demo-skill-opencode-with",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "4",
        "-p",
        str(run_dir / "_harbor-tasks" / "opencode" / "with"),
        "-a",
        "opencode",
        "--environment-import-path",
        SECURE_DOCKER_ENV_IMPORT_PATH,
        "--jobs-dir",
        str(run_dir / "_harbor-jobs"),
        "--model",
        f"nvidia/{AGENT_MODEL}",
        "--yes",
    ]
    assert call.env == {
        **_base_harbor_env(),
        "NVIDIA_API_KEY": NVIDIA_BUILD_STDIN_SENTINEL,
        NVIDIA_BUILD_KEY_STDIN_ENV: "1",
        "SKILL_EVAL_LLM_MODEL": AGENT_MODEL,
        "SKILL_EVAL_LLM_PROVIDER": "nv_build",
    }
    assert call.stdin == NVIDIA_KEY

    (task_toml,) = harness.staged_task_tomls(result)
    assert task_toml == _GOLDEN_NO_PANEL_TASK_HEAD + _runtime_env_toml_block({"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"})
    assert all(emitted.get("verifier_timeout_sec") is None for emitted in harness.emitted)

    surfaces = json.dumps([run_config, call.command, call.env, task_toml, result])
    assert "JUDGE_PANEL" not in surfaces
    assert harness.reporter.stage_events("judge-panel") == []


# --- Configuration, probing, and run_config ----------------------------------------------------------


def test_panel_probes_every_member_and_records_a_redacted_run_config(harness: _Harness) -> None:
    result = harness.run(_panel_environment())

    assert "error" not in result
    run_config = result["run_config"]
    assert run_config["judge"] == {
        "enabled": True,
        "mode": "panel",
        "panel": [
            {
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "label": "openai:gpt-5.6-sol",
                "catalog_verification": "verified",
            },
            {
                "provider": "anthropic",
                "model": "claude-opus-5",
                "label": "anthropic:claude-opus-5",
                "catalog_verification": "verified",
            },
            {
                "provider": "nv_build",
                "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
                "label": NV_JUDGE,
                "catalog_verification": "degraded",
            },
        ],
        "aggregation": "vote",
        "quorum": 2,
        "disagreement_threshold": 0.4,
        "source": "SKILL_EVAL_JUDGE_PANEL",
        "override_applied": True,
        "warnings": [],
        "catalog_verification": "degraded",
    }
    assert run_config["credential_validation"] == {
        "status": "degraded",
        "targets": [
            {
                "labels": ["opencode"],
                "provider": "nv_build",
                "model": AGENT_MODEL,
                "status": "degraded",
                "detail": _CATALOG_DEGRADED,
            },
            {
                "labels": ["standard grader: openai:gpt-5.6-sol"],
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "status": "verified",
                "detail": "model gpt-5.6-sol is available",
            },
            {
                "labels": ["standard grader: anthropic:claude-opus-5"],
                "provider": "anthropic",
                "model": "claude-opus-5",
                "status": "verified",
                "detail": "model claude-opus-5 is available",
            },
            {
                "labels": [f"standard grader: {NV_JUDGE}"],
                "provider": "nv_build",
                "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
                "status": "degraded",
                "detail": _CATALOG_DEGRADED,
            },
        ],
    }
    probed = {f"{config.provider}:{config.model}": config for config in harness.probed}
    assert probed["openai:gpt-5.6-sol"].api_key == OPENAI_KEY
    assert probed["openai:gpt-5.6-sol"].base_url == "https://api.openai.com/v1"
    assert probed["anthropic:claude-opus-5"].api_key == ANTHROPIC_KEY
    assert probed["anthropic:claude-opus-5"].base_url is None
    assert probed[NV_JUDGE].api_key == NVIDIA_KEY

    run_dir = Path(result["run_dir"])
    persisted = [
        json.dumps(run_config),
        (run_dir / "run_config.json").read_text(encoding="utf-8"),
        (run_dir / "result.json").read_text(encoding="utf-8"),
    ]
    for surface in persisted:
        for secret in (NVIDIA_KEY, *MEMBER_SECRETS):
            assert secret not in surface
        assert ALIAS not in surface

    assert [emitted["verifier_timeout_sec"] for emitted in harness.emitted] == [1800.0]
    (task_toml,) = harness.staged_task_tomls(result)
    assert tomllib.loads(task_toml)["verifier"]["timeout_sec"] == 1800.0


def test_panel_progress_event_summarizes_judges_without_secrets(harness: _Harness) -> None:
    result = harness.run(_panel_environment())

    assert "error" not in result
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "ready"
    assert PANEL in event.detail
    assert "vote" in event.detail
    for secret in (NVIDIA_KEY, *MEMBER_SECRETS):
        assert all(secret not in str(item.detail) for item in harness.reporter.events)
    engine_secret_calls = harness.reporter.secret_calls[1:]
    assert engine_secret_calls
    assert all({NVIDIA_KEY, OPENAI_KEY, ANTHROPIC_KEY} <= values for values in engine_secret_calls)


def test_fatal_member_probe_blocks_the_run_before_staging(harness: _Harness) -> None:
    harness.probe_overrides["anthropic:claude-opus-5"] = lambda selected: runtime_preflight.ModelProbeResult(
        False,
        selected.provider,
        selected.model,
        f"model catalog returned HTTP 401 for key {ANTHROPIC_KEY}",
        failure_kind="authentication",
        http_status=401,
    )

    result = harness.run(_panel_environment())

    assert set(result) == {"error"}
    (error,) = result["error"]
    assert error.startswith("standard grader: anthropic:claude-opus-5 provider verification failed")
    assert ANTHROPIC_KEY not in error
    assert harness.emitted == []
    assert harness.harbor_calls == []
    assert not (harness.tmp_path / "results").exists()
    (failure,) = [event for event in harness.reporter.stage_events("credential-validation") if event.state == "failed"]
    assert ANTHROPIC_KEY not in failure.detail


def test_degraded_member_probe_continues_and_is_recorded(harness: _Harness) -> None:
    def unavailable(_selected: ProviderConfig) -> runtime_preflight.ModelProbeResult:
        raise RuntimeError(f"catalog unavailable for {OPENAI_KEY}")

    harness.probe_overrides["openai:gpt-5.6-sol"] = unavailable

    result = harness.run(_panel_environment())

    assert "error" not in result
    judge = result["run_config"]["judge"]
    assert [member["catalog_verification"] for member in judge["panel"]] == ["degraded", "verified", "degraded"]
    assert judge["catalog_verification"] == "degraded"
    openai_target = next(
        target
        for target in result["run_config"]["credential_validation"]["targets"]
        if target["labels"] == ["standard grader: openai:gpt-5.6-sol"]
    )
    assert openai_target == {
        "labels": ["standard grader: openai:gpt-5.6-sol"],
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "status": "degraded",
        "detail": "model catalog probe failed: RuntimeError",
    }
    assert len(harness.harbor_calls) == 1


def test_all_verified_members_report_a_verified_panel(harness: _Harness) -> None:
    result = harness.run(
        _nv_build_primary(
            OPENAI_API_KEY=OPENAI_KEY,
            ANTHROPIC_API_KEY=ANTHROPIC_KEY,
            **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol,anthropic:claude-opus-5,openai:gpt-5.4-mini"},
        )
    )

    assert "error" not in result
    judge = result["run_config"]["judge"]
    assert {member["catalog_verification"] for member in judge["panel"]} == {"verified"}
    assert judge["catalog_verification"] == "verified"


def test_cli_judge_panel_overrides_the_environment_panel(harness: _Harness) -> None:
    result = harness.run(_panel_environment(), judge_panel="anthropic:claude-opus-5")

    assert "error" not in result
    judge = result["run_config"]["judge"]
    assert [member["label"] for member in judge["panel"]] == ["anthropic:claude-opus-5"]
    assert judge["source"] == "--judge-panel"
    assert judge["quorum"] == 1
    assert {f"{config.provider}:{config.model}" for config in harness.probed} == {
        f"nv_build:{AGENT_MODEL}",
        "anthropic:claude-opus-5",
    }
    call = harness.single_call()
    assert call.env["SKILL_EVAL_JUDGE_PANEL"] == "anthropic:claude-opus-5"
    assert OPENAI_KEY not in call.env.values()
    assert f"{ALIAS}OPENAI_API_KEY" not in call.env


def test_invalid_cli_judge_panel_fails_configuration_even_with_a_valid_environment_panel(harness: _Harness) -> None:
    result = harness.run(_panel_environment(), judge_panel="gemini:gemini-pro")

    assert set(result) == {"error"}
    assert "gemini" in result["error"][0]
    assert harness.probed == []
    assert harness.harbor_calls == []


def test_custom_only_grading_ignores_the_panel(harness: _Harness) -> None:
    (harness.skill / "evals" / "grader.py").write_text("def grade(*args, **kwargs):\n    return 1\n", encoding="utf-8")

    result = harness.run(_panel_environment(), grading_mode="custom_only")

    assert "error" not in result
    assert result["run_config"]["judge"] == {
        "enabled": False,
        "panel_ignored": "custom_only grading runs no standard judges",
    }
    assert [target["labels"] for target in result["run_config"]["credential_validation"]["targets"]] == [["opencode"]]
    call = harness.single_call()
    assert "--verifier-env" not in call.command
    assert "JUDGE_PANEL" not in json.dumps(call.env)
    for secret in MEMBER_SECRETS:
        assert secret not in call.env.values()
    assert all(emitted.get("verifier_timeout_sec") is None for emitted in harness.emitted)
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "skipped"


def test_even_vote_panel_warns_and_still_runs(harness: _Harness) -> None:
    result = harness.run(_panel_environment("openai:gpt-5.6-sol,anthropic:claude-opus-5"))

    assert "error" not in result
    (warning,) = result["run_config"]["judge"]["warnings"]
    assert "odd number" in warning
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "degraded"
    assert warning in event.detail
    assert len(harness.harbor_calls) == 1


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            _nv_build_primary(OPENAI_API_KEY=OPENAI_KEY, **{JUDGE_PANEL_ENV: PANEL}),
            ("anthropic:claude-opus-5", "ANTHROPIC_API_KEY"),
        ),
        (
            _panel_environment(LLM_JUDGE_MODEL="single-judge"),
            ("LLM_JUDGE_MODEL", "the panel names each judge's model explicitly"),
        ),
        (_nv_build_primary(SKILL_EVAL_JUDGE_PANEL_QUORUM="2"), ("SKILL_EVAL_JUDGE_PANEL_QUORUM",)),
    ],
)
def test_invalid_panel_configuration_fails_before_any_probe(
    harness: _Harness,
    environment: dict[str, str],
    expected: tuple[str, ...],
) -> None:
    result = harness.run(environment)

    assert set(result) == {"error"}
    (error,) = result["error"]
    for fragment in expected:
        assert fragment in error
    assert harness.probed == []
    assert harness.harbor_calls == []
    assert [event.state for event in harness.reporter.stage_events("configuration")][-1] == "failed"


# --- Credential delivery (contract section 3.1) ------------------------------------------------------


def test_judge_only_member_credentials_reach_the_verifier_only_through_private_aliases(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = harness.run(_panel_environment())

    assert "error" not in result
    call = harness.single_call()
    for standard_name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        assert standard_name not in call.env
    assert _names_holding(call.env, ANTHROPIC_KEY) == {f"{ALIAS}ANTHROPIC_API_KEY"}
    assert _names_holding(call.env, OPENAI_KEY) == {f"{ALIAS}OPENAI_API_KEY"}
    assert {name: value for name, value in call.env.items() if "JUDGE_PANEL" in name} == {
        f"{ALIAS}ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        f"{ALIAS}ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        f"{ALIAS}OPENAI_API_KEY": OPENAI_KEY,
        f"{ALIAS}OPENAI_BASE_URL": "https://api.openai.com/v1",
        "SKILL_EVAL_JUDGE_PANEL": PANEL,
        "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "vote",
        "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.4",
        "SKILL_EVAL_JUDGE_PANEL_QUORUM": "2",
    }
    # The NVIDIA member shares the primary's key, so the stdin handoff still owns it.
    assert call.env["NVIDIA_API_KEY"] == NVIDIA_BUILD_STDIN_SENTINEL
    assert call.stdin == NVIDIA_KEY
    assert call.verifier_env == {
        "ANTHROPIC_API_KEY": f"${{{ALIAS}ANTHROPIC_API_KEY}}",
        "ANTHROPIC_BASE_URL": f"${{{ALIAS}ANTHROPIC_BASE_URL}}",
        "OPENAI_API_KEY": f"${{{ALIAS}OPENAI_API_KEY}}",
        "OPENAI_BASE_URL": f"${{{ALIAS}OPENAI_BASE_URL}}",
        **PANEL_SETTINGS_JOB_ENV,
    }
    for secret in (NVIDIA_KEY, *MEMBER_SECRETS):
        assert all(secret not in part for part in call.command)

    (task_toml,) = harness.staged_task_tomls(result)
    resolved = _resolve_verifier_env(monkeypatch, task_toml, call)
    assert resolved == {
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "NVIDIA_API_KEY": NVIDIA_BUILD_STDIN_SENTINEL,
        "OPENAI_API_KEY": OPENAI_KEY,
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "SKILL_EVAL_JUDGE_PANEL": PANEL,
        "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "vote",
        "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.4",
        "SKILL_EVAL_JUDGE_PANEL_QUORUM": "2",
        "SKILL_EVAL_LLM_MODEL": AGENT_MODEL,
        "SKILL_EVAL_LLM_PROVIDER": "nv_build",
    }


def test_primary_equivalent_nvidia_member_is_not_aliased_and_keeps_the_stdin_handoff(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = harness.run(_nv_build_primary(**{JUDGE_PANEL_ENV: NV_JUDGE}))

    assert "error" not in result
    call = harness.single_call()
    assert f"{ALIAS}NVIDIA_API_KEY" not in call.env
    assert NVIDIA_KEY not in call.env.values()
    assert call.env["NVIDIA_API_KEY"] == NVIDIA_BUILD_STDIN_SENTINEL
    assert call.env[NVIDIA_BUILD_KEY_STDIN_ENV] == "1"
    assert call.stdin == NVIDIA_KEY
    assert call.verifier_env == PANEL_SETTINGS_JOB_ENV
    (task_toml,) = harness.staged_task_tomls(result)
    assert _resolve_verifier_env(monkeypatch, task_toml, call)["NVIDIA_API_KEY"] == NVIDIA_BUILD_STDIN_SENTINEL


def test_gateway_primary_with_native_openai_member_routes_each_credential_to_its_own_endpoint(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
        "SKILL_EVAL_LLM_API_KEY": GATEWAY_KEY,
        "SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL,
        "OPENAI_API_KEY": OPENAI_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        JUDGE_PANEL_ENV: "openai-compatible:gateway-judge,openai:gpt-5.6-sol,anthropic:claude-opus-5",
    }

    result = harness.run(environment)

    assert "error" not in result
    call = harness.single_call()
    # The gateway agent keeps the primary's OPENAI_* pair; the native key exists only under its alias.
    assert call.env["OPENAI_API_KEY"] == GATEWAY_KEY
    assert call.env["OPENAI_BASE_URL"] == GATEWAY_URL
    assert _names_holding(call.env, OPENAI_KEY) == {f"{ALIAS}OPENAI_API_KEY"}
    assert call.env[f"{ALIAS}OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert call.env[f"{ALIAS}SKILL_EVAL_LLM_API_KEY"] == GATEWAY_KEY
    assert call.env[f"{ALIAS}SKILL_EVAL_LLM_BASE_URL"] == GATEWAY_URL
    assert "SKILL_EVAL_LLM_API_KEY" not in call.env
    assert call.verifier_env == {
        "ANTHROPIC_API_KEY": f"${{{ALIAS}ANTHROPIC_API_KEY}}",
        "ANTHROPIC_BASE_URL": f"${{{ALIAS}ANTHROPIC_BASE_URL}}",
        "OPENAI_API_KEY": f"${{{ALIAS}OPENAI_API_KEY}}",
        "OPENAI_BASE_URL": f"${{{ALIAS}OPENAI_BASE_URL}}",
        "SKILL_EVAL_LLM_API_KEY": f"${{{ALIAS}SKILL_EVAL_LLM_API_KEY}}",
        "SKILL_EVAL_LLM_BASE_URL": f"${{{ALIAS}SKILL_EVAL_LLM_BASE_URL}}",
        **PANEL_SETTINGS_JOB_ENV,
    }

    (task_toml,) = harness.staged_task_tomls(result)
    resolved = _resolve_verifier_env(monkeypatch, task_toml, call)
    assert resolved["OPENAI_API_KEY"] == OPENAI_KEY
    assert resolved["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert resolved["SKILL_EVAL_LLM_API_KEY"] == GATEWAY_KEY
    assert resolved["SKILL_EVAL_LLM_BASE_URL"] == GATEWAY_URL
    assert resolved["ANTHROPIC_API_KEY"] == ANTHROPIC_KEY
    assert resolved["SKILL_EVAL_LLM_PROVIDER"] == "openai-compatible"


def test_bedrock_member_aws_credentials_use_aliases_when_bedrock_is_not_the_primary(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = harness.run(
        _panel_environment(
            "bedrock:us.anthropic.claude-opus-5,openai:gpt-5.6-sol,anthropic:claude-opus-5",
            AWS_REGION="us-east-1",
            AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY,
            AWS_SECRET_ACCESS_KEY=AWS_SECRET_KEY,
        )
    )

    assert "error" not in result
    call = harness.single_call()
    assert not [name for name in call.env if name.startswith("AWS_")]
    assert call.env[f"{ALIAS}AWS_REGION"] == "us-east-1"
    assert call.env[f"{ALIAS}AWS_ACCESS_KEY_ID"] == AWS_ACCESS_KEY
    assert call.env[f"{ALIAS}AWS_SECRET_ACCESS_KEY"] == AWS_SECRET_KEY
    assert call.verifier_env["AWS_ACCESS_KEY_ID"] == f"${{{ALIAS}AWS_ACCESS_KEY_ID}}"
    assert call.verifier_env["AWS_SECRET_ACCESS_KEY"] == f"${{{ALIAS}AWS_SECRET_ACCESS_KEY}}"
    assert call.verifier_env["AWS_REGION"] == f"${{{ALIAS}AWS_REGION}}"
    assert (
        call.verifier_env["AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"] == f"${{{ALIAS}AWS_IGNORE_CONFIGURED_ENDPOINT_URLS}}"
    )
    (task_toml,) = harness.staged_task_tomls(result)
    resolved = _resolve_verifier_env(monkeypatch, task_toml, call)
    assert resolved["AWS_SECRET_ACCESS_KEY"] == AWS_SECRET_KEY
    assert resolved["AWS_REGION"] == "us-east-1"
    assert resolved["AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"] == "true"
    engine_secret_calls = harness.reporter.secret_calls[1:]
    assert all({AWS_ACCESS_KEY, AWS_SECRET_KEY} <= values for values in engine_secret_calls)


def test_staged_agent_environment_never_contains_member_credentials(harness: _Harness) -> None:
    result = harness.run(_panel_environment(), skip_baseline=False)

    assert "error" not in result
    assert len(harness.harbor_calls) == 2
    assert {call.verifier_env["OPENAI_API_KEY"] for call in harness.harbor_calls} == {f"${{{ALIAS}OPENAI_API_KEY}}"}
    for emitted in harness.emitted:
        assert emitted["runtime_env"] == {"NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
        assert emitted["verifier_timeout_sec"] == 1800.0
        assert "JUDGE_PANEL" not in json.dumps(emitted["verifier_env"])
    task_tomls = harness.staged_task_tomls(result)
    assert len(task_tomls) == 2
    for task_toml in task_tomls:
        staged = tomllib.loads(task_toml)
        assert staged["verifier"]["timeout_sec"] == 1800.0
        assert set(staged["verifier"]["env"]) == {"NVIDIA_API_KEY", "SKILL_EVAL_LLM_MODEL", "SKILL_EVAL_LLM_PROVIDER"}
        assert "JUDGE_PANEL" not in task_toml
        for secret in (NVIDIA_KEY, *MEMBER_SECRETS):
            assert secret not in task_toml
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
            assert name not in staged["environment"]["env"]


def test_stop_on_pass_attempts_receive_the_panel_environment(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_job_passed", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_merge_attempt_jobs", lambda *_args, **_kwargs: None)

    result = harness.run(_panel_environment(), n_attempts=2, stop_on_pass=True)

    assert "error" not in result
    call = harness.single_call()
    assert "demo-skill-opencode-with-case-001-attempt001" in call.command
    assert call.verifier_env["ANTHROPIC_API_KEY"] == f"${{{ALIAS}ANTHROPIC_API_KEY}}"
    assert call.env[f"{ALIAS}ANTHROPIC_API_KEY"] == ANTHROPIC_KEY
    assert call.env["SKILL_EVAL_JUDGE_PANEL"] == PANEL


def _write_native_task(skill: Path, verifier_table: str) -> None:
    task = skill / "evals" / "harbor" / "case-001"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
        f"{verifier_table}[environment]\n",
        encoding="utf-8",
    )
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: native_harbor\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("verifier_table", "expected_timeout", "budget_warning"),
    [
        (
            '[verifier]\ntimeout_sec = 180.0\n\n[verifier.env]\nLLM_JUDGE_MODEL = "authored-native-judge"\n\n',
            180.0,
            True,
        ),
        ("", 1800.0, False),
    ],
)
def test_native_tasks_use_explicit_member_probes_and_keep_author_timeouts(
    harness: _Harness,
    verifier_table: str,
    expected_timeout: float,
    budget_warning: bool,
) -> None:
    _write_native_task(harness.skill, verifier_table)

    result = harness.run(_panel_environment())

    assert "error" not in result
    assert result["run_config"]["task_source"] == "native_harbor"
    judge = result["run_config"]["judge"]
    assert judge["mode"] == "panel"
    assert "effective_model_source" not in judge
    # The authored timeout is kept, and a budget below the panel's needs is reported instead.
    assert [warning for warning in judge["warnings"] if "case-001 (180s)" in warning] == (
        judge["warnings"] if budget_warning else []
    )
    assert len(judge["warnings"]) == int(budget_warning)
    labels = [target["labels"] for target in result["run_config"]["credential_validation"]["targets"]]
    assert ["standard grader"] not in labels
    assert ["standard grader: anthropic:claude-opus-5"] in labels
    assert [emitted["verifier_timeout_sec"] for emitted in harness.emitted] == [1800.0]
    (task_toml,) = harness.staged_task_tomls(result)
    assert tomllib.loads(task_toml)["verifier"]["timeout_sec"] == expected_timeout
    assert harness.single_call().verifier_env["ANTHROPIC_API_KEY"] == f"${{{ALIAS}ANTHROPIC_API_KEY}}"


# --- Helper units --------------------------------------------------------------------------------------


def _resolved_panel(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]) -> Any:
    monkeypatch.setattr(runner.os, "environ", environment)
    panel = resolve_judge_panel_config(environment)
    assert panel is not None
    return panel


def test_panel_harbor_environment_skips_credentials_the_primary_already_delivers_but_pins_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": OPENAI_KEY,
        "SKILL_EVAL_LLM_BASE_URL": "https://openai-proxy.example/v1",
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1",
        JUDGE_PANEL_ENV: "openai:gpt-5.4-mini,anthropic:claude-opus-5,openai:gpt-5.6-sol",
    }
    panel = _resolved_panel(monkeypatch, environment)
    provider_env = runner._provider_environment(resolve_llm_provider(environment))

    subprocess_env, job_env = runner._judge_panel_harbor_environment(panel, provider_env)

    # The primary already delivers OPENAI_API_KEY; its endpoint is still pinned at the job layer.
    assert subprocess_env == {
        f"{ALIAS}ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        f"{ALIAS}ANTHROPIC_BASE_URL": "https://anthropic-proxy.example",
        f"{ALIAS}OPENAI_BASE_URL": "https://openai-proxy.example/v1",
        **panel.verifier_settings(),
    }
    assert job_env == {
        "ANTHROPIC_API_KEY": f"${{{ALIAS}ANTHROPIC_API_KEY}}",
        "ANTHROPIC_BASE_URL": f"${{{ALIAS}ANTHROPIC_BASE_URL}}",
        "OPENAI_BASE_URL": f"${{{ALIAS}OPENAI_BASE_URL}}",
        **PANEL_SETTINGS_JOB_ENV,
    }


def test_panel_harbor_environment_reuses_the_primary_bedrock_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "bedrock",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_KEY,
        JUDGE_PANEL_ENV: "bedrock:us.anthropic.claude-opus-5",
    }
    panel = _resolved_panel(monkeypatch, environment)
    provider_env = runner._provider_environment(resolve_llm_provider(environment))

    subprocess_env, job_env = runner._judge_panel_harbor_environment(panel, provider_env)

    # AWS credentials stay on the primary's task-level path; region and endpoint handling are pinned.
    assert subprocess_env == {
        f"{ALIAS}AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true",
        f"{ALIAS}AWS_REGION": "us-east-1",
        **panel.verifier_settings(),
    }
    assert job_env == {
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": f"${{{ALIAS}AWS_IGNORE_CONFIGURED_ENDPOINT_URLS}}",
        "AWS_REGION": f"${{{ALIAS}AWS_REGION}}",
        **PANEL_SETTINGS_JOB_ENV,
    }


def test_panel_harbor_environment_pins_the_official_anthropic_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        JUDGE_PANEL_ENV: "anthropic:claude-opus-5",
    }
    panel = _resolved_panel(monkeypatch, environment)
    provider_env = runner._provider_environment(resolve_llm_provider(environment))

    subprocess_env, job_env = runner._judge_panel_harbor_environment(panel, provider_env)

    # The key is already delivered by the primary; only the endpoint pin is added so an
    # authored ANTHROPIC_BASE_URL can never redirect the judge credential.
    assert f"{ALIAS}ANTHROPIC_API_KEY" not in subprocess_env
    assert subprocess_env[f"{ALIAS}ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert job_env["ANTHROPIC_BASE_URL"] == f"${{{ALIAS}ANTHROPIC_BASE_URL}}"
    assert "ANTHROPIC_API_KEY" not in job_env


def test_judge_model_config_describes_the_panel_without_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = _panel_environment()
    panel = _resolved_panel(monkeypatch, environment)
    provider = resolve_llm_provider(environment)
    provider_env = runner._provider_environment(provider)

    config = runner._judge_model_config(provider, provider_env, "default", panel, panel_source="--judge-panel")

    assert config == {
        "enabled": True,
        "mode": "panel",
        "panel": [
            {"provider": "openai", "model": "gpt-5.6-sol", "label": "openai:gpt-5.6-sol"},
            {"provider": "anthropic", "model": "claude-opus-5", "label": "anthropic:claude-opus-5"},
            {"provider": "nv_build", "model": "nvidia/llama-3.3-nemotron-super-49b-v1", "label": NV_JUDGE},
        ],
        "aggregation": "vote",
        "quorum": 2,
        "disagreement_threshold": 0.4,
        "source": "--judge-panel",
        "override_applied": True,
        "warnings": [],
    }
    assert runner._judge_model_config(provider, provider_env, "custom_only", panel) == {
        "enabled": False,
        "panel_ignored": "custom_only grading runs no standard judges",
    }
    assert runner._judge_model_config(provider, provider_env, "custom_only") == {"enabled": False}
