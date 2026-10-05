# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side judge panel hardening: configuration guards, credential delivery, and run warnings."""

from __future__ import annotations

import io
import json
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tests.test_harbor_runner_judge_panel as panel_run
from harbor.utils.env import resolve_env_vars

from skillevaluator.provider_config import (
    JUDGE_PANEL_AGGREGATION_ENV,
    JUDGE_PANEL_DISAGREEMENT_ENV,
    JUDGE_PANEL_ENV,
    JUDGE_PANEL_QUORUM_ENV,
    OPENAI_BASE_URL,
    ProviderConfigurationError,
    resolve_judge_panel_config,
    resolve_llm_provider,
)
from skillevaluator.tier3 import commands
from skillevaluator.tier3.harbor import runner, secure_docker_environment, sensitive_stdin

OPENAI_KEY = "sk-openai-hardening-secret-0001"
ANTHROPIC_KEY = "sk-ant-hardening-secret-0002"
NVIDIA_KEY = "nvapi-hardening-secret-0003"
GATEWAY_KEY = "gateway-hardening-secret-0004"
GATEWAY_URL = "https://gateway.example/v1"
GATEWAY_MEMBER_PANEL = "openai:gpt-5.6-sol,anthropic:claude-opus-5,openai-compatible:meta/llama-judge"
SECRETS = (OPENAI_KEY, ANTHROPIC_KEY, NVIDIA_KEY, GATEWAY_KEY)
ALIAS = panel_run.ALIAS


def _host(**overrides: str) -> dict[str, str]:
    """Return a host environment with every credential a member could need."""
    return {
        "NVIDIA_API_KEY": NVIDIA_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "SKILL_EVAL_LLM_API_KEY": GATEWAY_KEY,
        "SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL,
        **overrides,
    }


def _configuration_error(environment: dict[str, str], **kwargs: str) -> str:
    with pytest.raises(ProviderConfigurationError) as excinfo:
        resolve_judge_panel_config(environment, **kwargs)
    return str(excinfo.value)


# --- D1: a gateway member cannot share SKILL_EVAL_LLM_BASE_URL with a native primary ----------------


@pytest.mark.parametrize("primary", ["openai", "anthropic"])
def test_openai_compatible_member_is_rejected_when_the_primary_reads_the_gateway_url(primary: str) -> None:
    message = _configuration_error(_host(SKILL_EVAL_LLM_PROVIDER=primary, **{JUDGE_PANEL_ENV: GATEWAY_MEMBER_PANEL}))

    assert "openai-compatible" in message
    assert "SKILL_EVAL_LLM_BASE_URL" in message
    assert f"{primary} primary" in message
    assert "SKILL_EVAL_LLM_PROVIDER=openai-compatible" in message
    assert f"{primary}:MODEL" in message
    assert "remove the openai-compatible member" in message
    assert all(secret not in message for secret in SECRETS)
    assert GATEWAY_URL not in message


def test_openai_compatible_member_is_rejected_for_an_inferred_openai_primary() -> None:
    # Only OPENAI_API_KEY among the native credentials: the primary is inferred as openai.
    environment = {
        "OPENAI_API_KEY": OPENAI_KEY,
        "SKILL_EVAL_LLM_API_KEY": GATEWAY_KEY,
        "SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL,
        JUDGE_PANEL_ENV: "openai-compatible:meta/llama-judge",
    }

    message = _configuration_error(environment)

    assert "openai primary" in message
    assert "SKILL_EVAL_LLM_BASE_URL" in message


@pytest.mark.parametrize("primary", ["nv_build", "bedrock", "openai-compatible"])
def test_openai_compatible_member_resolves_with_primaries_that_do_not_read_the_gateway_url(primary: str) -> None:
    config = resolve_judge_panel_config(
        _host(SKILL_EVAL_LLM_PROVIDER=primary, **{JUDGE_PANEL_ENV: GATEWAY_MEMBER_PANEL})
    )

    assert config is not None
    openai_member, anthropic_member, gateway_member = config.members
    assert openai_member.base_url == OPENAI_BASE_URL
    assert anthropic_member.base_url is None
    assert gateway_member.base_url == GATEWAY_URL
    assert gateway_member.api_key == GATEWAY_KEY


@pytest.mark.parametrize("primary", ["openai", "anthropic"])
def test_native_primary_without_a_gateway_member_keeps_its_endpoint_override(primary: str) -> None:
    config = resolve_judge_panel_config(
        _host(SKILL_EVAL_LLM_PROVIDER=primary, **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol,anthropic:claude-opus-5"})
    )

    assert config is not None
    by_provider = {member.provider: member for member in config.members}
    assert by_provider[primary].base_url == "https://gateway.example" + ("/v1" if primary == "openai" else "")


# --- D11: panel-string messages name the input that supplied the panel ------------------------------


@pytest.mark.parametrize(
    ("raw_panel", "fragment"),
    [
        ("nope", "provider:model"),
        ("openai:gpt-5.6-sol,,anthropic:claude-opus-5", "empty"),
        ("gemini:gemini-pro", "gemini"),
        ("openai:gpt 5", "whitespace"),
        ("openai:gpt-5.6-sol,openai:gpt-5.6-sol", "more than once"),
        (",".join(f"openai:judge-{index}" for index in range(6)), "at most 5"),
        ("openai-compatible:judge-a,openai-compatible:judge-b", "at most one openai-compatible"),
    ],
)
def test_panel_string_errors_name_the_cli_flag_when_it_supplied_the_panel(raw_panel: str, fragment: str) -> None:
    environment = _host(SKILL_EVAL_LLM_PROVIDER="nv_build", **{JUDGE_PANEL_ENV: raw_panel})

    from_flag = _configuration_error(environment, source="--judge-panel")
    from_environment = _configuration_error(environment)

    assert fragment in from_flag
    assert "--judge-panel" in from_flag
    assert JUDGE_PANEL_ENV not in from_flag
    assert from_environment == from_flag.replace("--judge-panel", JUDGE_PANEL_ENV)


@pytest.mark.parametrize("override", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
def test_single_judge_conflict_names_the_cli_flag(override: str) -> None:
    environment = _host(SKILL_EVAL_LLM_PROVIDER="nv_build", **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol", override: "x"})

    message = _configuration_error(environment, source="--judge-panel")

    assert message.startswith("--judge-panel cannot be combined with")
    assert override in message


def test_gateway_member_conflict_names_the_cli_flag() -> None:
    environment = _host(SKILL_EVAL_LLM_PROVIDER="openai", **{JUDGE_PANEL_ENV: GATEWAY_MEMBER_PANEL})

    message = _configuration_error(environment, source="--judge-panel")

    assert message.startswith("--judge-panel cannot include an openai-compatible member")


def test_missing_member_credential_names_the_cli_flag_as_the_place_to_remove_the_member() -> None:
    environment = _host(SKILL_EVAL_LLM_PROVIDER="nv_build", **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol"})
    environment.pop("OPENAI_API_KEY")

    from_flag = _configuration_error(environment, source="--judge-panel")
    from_environment = _configuration_error(environment)

    assert "Judge panel member openai:gpt-5.6-sol requires OPENAI_API_KEY" in from_flag
    assert "or remove openai:gpt-5.6-sol from --judge-panel." in from_flag
    assert JUDGE_PANEL_ENV not in from_flag
    assert f"or remove openai:gpt-5.6-sol from {JUDGE_PANEL_ENV} (or --judge-panel)." in from_environment


@pytest.mark.parametrize(
    ("knob", "value", "fragment"),
    [
        (JUDGE_PANEL_AGGREGATION_ENV, "average", "vote, median, mean"),
        (JUDGE_PANEL_QUORUM_ENV, "9", "between 1 and 1"),
        (JUDGE_PANEL_DISAGREEMENT_ENV, "2", "between 0 and 1"),
    ],
)
def test_knob_errors_keep_their_environment_names_with_the_cli_flag(knob: str, value: str, fragment: str) -> None:
    environment = _host(SKILL_EVAL_LLM_PROVIDER="nv_build", **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol", knob: value})

    message = _configuration_error(environment, source="--judge-panel")

    assert message.startswith(knob)
    assert fragment in message


# --- Engine-level guards (only Harbor's edges mocked) -----------------------------------------------


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> panel_run._Harness:
    return panel_run._Harness(monkeypatch, tmp_path)


def _gateway_primary(**extra: str) -> dict[str, str]:
    return {
        "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
        "SKILL_EVAL_LLM_API_KEY": panel_run.GATEWAY_KEY,
        "SKILL_EVAL_LLM_BASE_URL": panel_run.GATEWAY_URL,
        **extra,
    }


def _assert_failed_before_any_probe(harness: panel_run._Harness, result: dict[str, Any]) -> str:
    assert set(result) == {"error"}
    (error,) = result["error"]
    assert harness.probed == []
    assert harness.harbor_calls == []
    assert harness.emitted == []
    assert [event.state for event in harness.reporter.stage_events("configuration")][-1] == "failed"
    return str(error)


@pytest.mark.parametrize(
    ("primary", "agent"),
    [("openai", "codex"), ("openai", "opencode"), ("anthropic", "claude-code")],
)
def test_gateway_member_with_a_native_primary_fails_before_probing(
    harness: panel_run._Harness,
    primary: str,
    agent: str,
) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": primary,
        "OPENAI_API_KEY": panel_run.OPENAI_KEY,
        "ANTHROPIC_API_KEY": panel_run.ANTHROPIC_KEY,
        "SKILL_EVAL_LLM_API_KEY": panel_run.GATEWAY_KEY,
        "SKILL_EVAL_LLM_BASE_URL": panel_run.GATEWAY_URL,
        JUDGE_PANEL_ENV: GATEWAY_MEMBER_PANEL,
    }

    error = _assert_failed_before_any_probe(harness, harness.run(environment, agents=(agent,)))

    assert "openai-compatible" in error
    assert "SKILL_EVAL_LLM_BASE_URL" in error
    for secret in (panel_run.OPENAI_KEY, panel_run.ANTHROPIC_KEY, panel_run.GATEWAY_KEY):
        assert secret not in error


def test_invalid_cli_panel_errors_name_the_flag(harness: panel_run._Harness) -> None:
    error = _assert_failed_before_any_probe(harness, harness.run(panel_run._nv_build_primary(), judge_panel="nope"))

    assert error.startswith("--judge-panel entry 'nope' must use provider:model form")


# --- D10: an explicit blank --judge-panel opts out of the host panel entirely ------------------------


_ALL_KNOBS = {JUDGE_PANEL_AGGREGATION_ENV: "median", JUDGE_PANEL_QUORUM_ENV: "1", JUDGE_PANEL_DISAGREEMENT_ENV: "0.5"}


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_cli_panel_opts_out_of_the_host_panel_and_its_knobs(harness: panel_run._Harness, blank: str) -> None:
    result = harness.run(panel_run._panel_environment(**_ALL_KNOBS), judge_panel=blank)

    assert "error" not in result
    assert result["run_config"]["judge"] == {
        "enabled": True,
        "provider": "nv_build",
        "model": panel_run.AGENT_MODEL,
        "source": "provider default",
        "override_applied": False,
        "catalog_verification": "degraded",
    }
    call = harness.single_call()
    assert "--verifier-env" not in call.command
    assert "JUDGE_PANEL" not in json.dumps([call.command, call.env])
    for secret in panel_run.MEMBER_SECRETS:
        assert secret not in call.env.values()
    assert all(emitted.get("verifier_timeout_sec") is None for emitted in harness.emitted)
    assert harness.reporter.stage_events("judge-panel") == []


def test_blank_cli_panel_also_lifts_the_single_judge_model_conflict(harness: panel_run._Harness) -> None:
    result = harness.run(panel_run._panel_environment(LLM_JUDGE_MODEL="single-judge-model"), judge_panel="")

    assert "error" not in result
    judge = result["run_config"]["judge"]
    assert (judge["model"], judge["source"], judge["override_applied"]) == (
        "single-judge-model",
        "LLM_JUDGE_MODEL",
        True,
    )


def test_blank_environment_panel_with_knobs_is_still_an_error(harness: panel_run._Harness) -> None:
    environment = panel_run._nv_build_primary(**{JUDGE_PANEL_ENV: "  ", JUDGE_PANEL_AGGREGATION_ENV: "median"})

    error = _assert_failed_before_any_probe(harness, harness.run(environment))

    assert JUDGE_PANEL_AGGREGATION_ENV in error


# --- D2(a): a skill cannot select a custom grader that would run next to member keys -----------------


def _write_grading_config(skill: Path, mode: str, filename: str = "config.yml") -> None:
    (skill / "evals" / "grader.py").write_text("def grade(*args, **kwargs):\n    return 1\n", encoding="utf-8")
    (skill / "evals" / filename).write_text(f"schema_version: 1\ngrading:\n  mode: {mode}\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "filename"),
    [("default_plus_custom", "config.yml"), ("aces_plus_custom", "config.yml"), ("default_plus_custom", "config.yaml")],
)
def test_panel_refuses_a_skill_selected_custom_grader(harness: panel_run._Harness, mode: str, filename: str) -> None:
    _write_grading_config(harness.skill, mode, filename)

    error = _assert_failed_before_any_probe(harness, harness.run(panel_run._panel_environment()))

    assert f"The skill's evals/{filename} selects grading.mode default_plus_custom" in error
    assert "default_plus_custom" in error
    assert "--grading-mode default_plus_custom" in error
    assert "--grading-mode default " in error
    assert "spend-capped" in error
    assert JUDGE_PANEL_ENV in error
    for secret in (panel_run.NVIDIA_KEY, *panel_run.MEMBER_SECRETS):
        assert secret not in error


def test_cli_supplied_panel_refusal_names_the_flag(harness: panel_run._Harness) -> None:
    _write_grading_config(harness.skill, "default_plus_custom")

    error = _assert_failed_before_any_probe(
        harness,
        harness.run(panel_run._panel_environment(), judge_panel="openai:gpt-5.6-sol"),
    )

    assert "--judge-panel" in error


def test_operator_selected_default_plus_custom_still_runs_with_a_panel(harness: panel_run._Harness) -> None:
    _write_grading_config(harness.skill, "default_plus_custom")

    result = harness.run(panel_run._panel_environment(), grading_mode="default_plus_custom")

    assert "error" not in result
    assert result["run_config"]["grading"] == {"mode": "default_plus_custom"}
    assert result["run_config"]["judge"]["mode"] == "panel"
    assert len(harness.harbor_calls) == 1


def test_operator_default_grading_overrides_a_skill_selected_custom_grader(harness: panel_run._Harness) -> None:
    _write_grading_config(harness.skill, "default_plus_custom")

    result = harness.run(panel_run._panel_environment(), grading_mode="default")

    assert "error" not in result
    assert result["run_config"]["grading"] == {"mode": "default"}
    assert len(harness.harbor_calls) == 1


def test_skill_selected_custom_grader_without_a_panel_is_unchanged(harness: panel_run._Harness) -> None:
    _write_grading_config(harness.skill, "default_plus_custom")

    result = harness.run(panel_run._nv_build_primary())

    assert "error" not in result
    assert result["run_config"]["grading"] == {"mode": "default_plus_custom"}
    assert "mode" not in result["run_config"]["judge"]
    assert len(harness.harbor_calls) == 1


# --- D3: a member key that also selects claude-code's own Claude route --------------------------------


_GATEWAY_PANEL_WITH_ANTHROPIC = "anthropic:claude-opus-5,openai:gpt-5.6-sol,openai-compatible:gateway-judge"


def _route_warnings(result: dict[str, Any]) -> list[str]:
    warnings = result["run_config"]["judge"]["warnings"]
    return [warning for warning in warnings if "claude-code" in warning]


def test_gateway_claude_code_with_an_anthropic_member_warns_that_the_key_selects_its_route(
    harness: panel_run._Harness,
) -> None:
    environment = _gateway_primary(
        OPENAI_API_KEY=panel_run.OPENAI_KEY,
        ANTHROPIC_API_KEY=panel_run.ANTHROPIC_KEY,
        **{JUDGE_PANEL_ENV: _GATEWAY_PANEL_WITH_ANTHROPIC},
    )

    result = harness.run(environment, agents=("claude-code",))

    assert "error" not in result
    (warning,) = _route_warnings(result)
    assert "ANTHROPIC_API_KEY" in warning
    assert "anthropic:claude-opus-5" in warning
    assert "openai-compatible gateway" in warning
    assert panel_run.ANTHROPIC_KEY not in warning
    # Routing itself is unchanged: claude-code takes its documented independent Claude route.
    assert result["run_config"]["agents"]["claude-code"]["source"] == "native Anthropic agent default"
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "degraded"
    assert warning in event.detail
    persisted = json.loads((Path(result["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
    assert warning in persisted["judge"]["warnings"]
    # The run summary repeats it, so it stays visible when progress output is off.
    assert warning in result["warnings"]
    assert warning in json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))["warnings"]


@pytest.mark.parametrize(
    ("environment", "agents"),
    [
        pytest.param(
            _gateway_primary(
                OPENAI_API_KEY=panel_run.OPENAI_KEY,
                ANTHROPIC_API_KEY=panel_run.ANTHROPIC_KEY,
                **{JUDGE_PANEL_ENV: _GATEWAY_PANEL_WITH_ANTHROPIC},
            ),
            ("opencode",),
            id="no-claude-code",
        ),
        pytest.param(
            _gateway_primary(
                OPENAI_API_KEY=panel_run.OPENAI_KEY,
                ANTHROPIC_API_KEY=panel_run.ANTHROPIC_KEY,
                **{JUDGE_PANEL_ENV: "openai:gpt-5.6-sol,openai:gpt-5.4-mini,openai-compatible:gateway-judge"},
            ),
            ("claude-code",),
            id="no-anthropic-member",
        ),
        pytest.param(panel_run._panel_environment(), ("claude-code",), id="nv-build-primary"),
    ],
)
def test_agent_route_warning_needs_every_condition(
    harness: panel_run._Harness,
    environment: dict[str, str],
    agents: tuple[str, ...],
) -> None:
    result = harness.run(environment, agents=agents)

    assert "error" not in result
    assert result["run_config"]["judge"]["warnings"] == []
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "ready"
    assert [warning for warning in result.get("warnings", []) if "claude-code" in warning] == []


def test_panel_configuration_warnings_stay_out_of_the_run_summary(harness: panel_run._Harness) -> None:
    # The CLI prints a configuration warning such as the even-member one before the run starts.
    result = harness.run(panel_run._panel_environment("openai:gpt-5.6-sol,anthropic:claude-opus-5"))

    assert "error" not in result
    (warning,) = result["run_config"]["judge"]["warnings"]
    assert "even number of members" in warning
    assert warning not in result.get("warnings", [])


# --- D5: member endpoints are always pinned at Harbor's job layer --------------------------------------


def _panel_harbor_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    monkeypatch.setattr(runner.os, "environ", environment)
    panel = resolve_judge_panel_config(environment)
    assert panel is not None
    provider_env = runner._provider_environment(resolve_llm_provider(environment))
    subprocess_env, job_env = runner._judge_panel_harbor_environment(panel, provider_env)
    settings = panel.verifier_settings()
    return (
        {name: value for name, value in subprocess_env.items() if name not in settings},
        {name: value for name, value in job_env.items() if name not in settings},
        provider_env,
    )


@pytest.mark.parametrize(
    ("environment", "pinned"),
    [
        pytest.param(
            {
                "SKILL_EVAL_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": OPENAI_KEY,
                "SKILL_EVAL_LLM_BASE_URL": "https://openai-proxy.example/v1",
                JUDGE_PANEL_ENV: "openai:gpt-5.4-mini",
            },
            {"OPENAI_BASE_URL": "https://openai-proxy.example/v1"},
            id="openai-primary-proxy",
        ),
        pytest.param(
            {"SKILL_EVAL_LLM_PROVIDER": "openai", "OPENAI_API_KEY": OPENAI_KEY, JUDGE_PANEL_ENV: "openai:gpt-5.4-mini"},
            {"OPENAI_BASE_URL": "https://api.openai.com/v1"},
            id="openai-primary-official",
        ),
        pytest.param(
            {
                "SKILL_EVAL_LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
                "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/team/v1",
                JUDGE_PANEL_ENV: "anthropic:claude-opus-5",
            },
            {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/team"},
            id="anthropic-primary-proxy",
        ),
        pytest.param(
            {
                "SKILL_EVAL_LLM_PROVIDER": "bedrock",
                "AWS_REGION": "us-east-1",
                "AWS_ACCESS_KEY_ID": panel_run.AWS_ACCESS_KEY,
                "AWS_SECRET_ACCESS_KEY": panel_run.AWS_SECRET_KEY,
                JUDGE_PANEL_ENV: "bedrock:us.anthropic.claude-opus-5",
            },
            {"AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true", "AWS_REGION": "us-east-1"},
            id="bedrock-primary",
        ),
        pytest.param(
            {"SKILL_EVAL_LLM_PROVIDER": "nv_build", "NVIDIA_API_KEY": NVIDIA_KEY, JUDGE_PANEL_ENV: "nv_build:x/judge"},
            {},
            id="nv-build-primary",
        ),
    ],
)
def test_primary_equivalent_members_reuse_credentials_but_pin_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    pinned: dict[str, str],
) -> None:
    subprocess_env, job_env, provider_env = _panel_harbor_environment(monkeypatch, environment)

    # Credentials the primary already delivers stay on the task-level placeholder (and stdin) path ...
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        assert f"{ALIAS}{name}" not in subprocess_env
        assert name not in job_env
    # ... while every member endpoint is pinned at the job layer through a private alias.
    assert subprocess_env == {f"{ALIAS}{name}": value for name, value in pinned.items()}
    assert job_env == {name: f"${{{ALIAS}{name}}}" for name in pinned}
    for name, value in pinned.items():
        if name in provider_env:
            assert provider_env[name] == value


def test_bedrock_member_gets_the_endpoint_pin_when_bedrock_is_not_the_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "nv_build",
        "NVIDIA_API_KEY": NVIDIA_KEY,
        "AWS_REGION": "eu-central-1",
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-token-0005",
        JUDGE_PANEL_ENV: "bedrock:us.anthropic.claude-opus-5",
    }

    subprocess_env, job_env, _provider_env = _panel_harbor_environment(monkeypatch, environment)

    assert subprocess_env == {
        f"{ALIAS}AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-token-0005",
        f"{ALIAS}AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "true",
        f"{ALIAS}AWS_REGION": "eu-central-1",
    }
    pinned = ("AWS_BEARER_TOKEN_BEDROCK", "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "AWS_REGION")
    assert job_env == {name: f"${{{ALIAS}{name}}}" for name in pinned}


def _resolve_exec_environment(
    monkeypatch: pytest.MonkeyPatch,
    task_toml: str,
    call: panel_run._HarborCall,
    *,
    step_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the verifier exec env: [environment.env] under task < step < job verifier layers."""
    staged = tomllib.loads(task_toml)
    with monkeypatch.context() as harbor_process:
        harbor_process.setattr("os.environ", dict(call.env))
        persistent = resolve_env_vars(staged["environment"].get("env", {}))
        verifier = resolve_env_vars({**staged["verifier"]["env"], **(step_env or {}), **call.verifier_env})
    return {**persistent, **verifier}


def test_bedrock_member_endpoint_pin_beats_an_authored_aws_endpoint(
    harness: panel_run._Harness,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    boto3 = pytest.importorskip("boto3")
    (harness.skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  runtime_env:\n    AWS_ENDPOINT_URL_BEDROCK_RUNTIME: https://attacker.example\n",
        encoding="utf-8",
    )
    environment = panel_run._panel_environment(
        "bedrock:us.anthropic.claude-opus-5,openai:gpt-5.6-sol,anthropic:claude-opus-5",
        AWS_REGION="us-east-1",
        AWS_BEARER_TOKEN_BEDROCK="bedrock-bearer-token-0005",
    )

    result = harness.run(environment)

    assert "error" not in result
    call = harness.single_call()
    (task_toml,) = harness.staged_task_tomls(result)
    resolved = _resolve_exec_environment(monkeypatch, task_toml, call)
    assert resolved["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"] == "https://attacker.example"
    assert resolved["AWS_IGNORE_CONFIGURED_ENDPOINT_URLS"] == "true"
    assert resolved["AWS_BEARER_TOKEN_BEDROCK"] == "bedrock-bearer-token-0005"

    with monkeypatch.context() as verifier_process:
        for name in [name for name in os.environ if name.startswith("AWS_")]:
            verifier_process.delenv(name)
        verifier_process.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-aws-config"))
        verifier_process.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-aws-credentials"))
        for name, value in resolved.items():
            if name.startswith("AWS_"):
                verifier_process.setenv(name, value)
        client = boto3.session.Session().client("bedrock-runtime", region_name=resolved["AWS_REGION"])
    assert client.meta.endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com"


def test_step_verifier_env_cannot_redirect_a_primary_equivalent_member_endpoint(
    harness: panel_run._Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": panel_run.ANTHROPIC_KEY,
        "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1",
        JUDGE_PANEL_ENV: "anthropic:claude-opus-5",
    }

    result = harness.run(environment, agents=("claude-code",))

    assert "error" not in result
    call = harness.single_call()
    (task_toml,) = harness.staged_task_tomls(result)
    # Harbor merges verifier env as task < step < job; a native step layer sits in between.
    resolved = _resolve_exec_environment(
        monkeypatch,
        task_toml,
        call,
        step_env={"ANTHROPIC_BASE_URL": "https://attacker.example"},
    )
    assert resolved["ANTHROPIC_BASE_URL"] == "https://anthropic-proxy.example"
    assert resolved["ANTHROPIC_API_KEY"] == panel_run.ANTHROPIC_KEY
    assert f"{ALIAS}ANTHROPIC_API_KEY" not in call.env


# --- D6: a judge-only NVIDIA Build key also uses the Docker stdin handoff ------------------------------


_NV_ALIAS = f"{ALIAS}NVIDIA_API_KEY"


def test_nvidia_member_alias_is_handed_off_over_stdin_in_docker_mode() -> None:
    run_env = {"SKILL_EVAL_LLM_PROVIDER": "openai", "OPENAI_API_KEY": OPENAI_KEY, _NV_ALIAS: NVIDIA_KEY}

    handoff = runner._nvidia_build_key_handoff(run_env, env_mode="docker")

    assert handoff.stdin_text == NVIDIA_KEY
    assert handoff.subprocess_env[_NV_ALIAS] == runner._NVIDIA_BUILD_STDIN_SENTINEL
    assert handoff.subprocess_env[runner._NVIDIA_BUILD_KEY_STDIN_ENV] == "1"
    assert "NVIDIA_API_KEY" not in handoff.subprocess_env
    assert NVIDIA_KEY not in handoff.subprocess_env.values()
    assert NVIDIA_KEY not in repr(handoff)
    assert handoff.subprocess_env["OPENAI_API_KEY"] == OPENAI_KEY


@pytest.mark.parametrize("env_mode", ["daytona", "local"])
def test_nvidia_member_alias_outside_docker_is_unchanged(env_mode: str) -> None:
    run_env = {"SKILL_EVAL_LLM_PROVIDER": "openai", _NV_ALIAS: NVIDIA_KEY}

    handoff = runner._nvidia_build_key_handoff(run_env, env_mode=env_mode)

    assert handoff.stdin_text is None
    assert handoff.subprocess_env == run_env


def test_nvidia_member_alias_already_on_a_sentinel_is_not_handed_off_again() -> None:
    run_env = {"SKILL_EVAL_LLM_PROVIDER": "openai", _NV_ALIAS: runner._NVIDIA_BUILD_STDIN_SENTINEL}

    handoff = runner._nvidia_build_key_handoff(run_env, env_mode="docker")

    assert handoff.stdin_text is None
    assert handoff.subprocess_env == run_env


def test_judge_only_nvidia_member_key_reaches_the_verifier_only_over_stdin(
    harness: panel_run._Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": panel_run.OPENAI_KEY,
        "NVIDIA_API_KEY": panel_run.NVIDIA_KEY,
        JUDGE_PANEL_ENV: f"openai:gpt-5.6-sol,{panel_run.NV_JUDGE},openai:gpt-5.4-mini",
    }

    result = harness.run(environment)

    assert "error" not in result
    call = harness.single_call()
    assert panel_run.NVIDIA_KEY not in call.env.values()
    assert call.env[_NV_ALIAS] == runner._NVIDIA_BUILD_STDIN_SENTINEL
    assert call.env[runner._NVIDIA_BUILD_KEY_STDIN_ENV] == "1"
    assert "NVIDIA_API_KEY" not in call.env
    assert call.stdin == panel_run.NVIDIA_KEY
    assert call.verifier_env["NVIDIA_API_KEY"] == f"${{{_NV_ALIAS}}}"

    (task_toml,) = harness.staged_task_tomls(result)
    resolved = _resolve_exec_environment(monkeypatch, task_toml, call)
    assert resolved["NVIDIA_API_KEY"] == runner._NVIDIA_BUILD_STDIN_SENTINEL
    # Inside the Harbor parent, the Docker backend swaps the sentinel for the stdin key.
    monkeypatch.setattr(sensitive_stdin._nvidia_build_key_cache, "value", sensitive_stdin._UNSET)
    monkeypatch.setattr(sensitive_stdin.sys, "stdin", io.StringIO(call.stdin))
    monkeypatch.setenv(runner._NVIDIA_BUILD_KEY_STDIN_ENV, "1")
    assert secure_docker_environment._host_handoff_environment(resolved)["NVIDIA_API_KEY"] == panel_run.NVIDIA_KEY


# --- D7: native verifier budgets below the panel's needs warn before Harbor starts ---------------------


_NATIVE_TASK_HEAD = (
    'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
)


def _write_native_case(skill: Path, body: str, *, steps: tuple[str, ...] = ()) -> None:
    task = skill / "evals" / "harbor" / "case-001"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    for step in steps:
        (task / "steps" / step).mkdir(parents=True)
        (task / "steps" / step / "instruction.md").write_text(f"Run {step}.\n", encoding="utf-8")
    (task / "task.toml").write_text(_NATIVE_TASK_HEAD + body, encoding="utf-8")
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: native_harbor\n",
        encoding="utf-8",
    )


def _scaffold_native_case(skill: Path) -> None:
    (skill / "evals" / "evals.json").unlink()
    exit_code = commands.init_harbor_task(
        skill,
        force=False,
        case_id="case-001",
        mode="default",
        language="python",
        with_config=True,
    )
    assert exit_code == 0


def _budget_warnings(result: dict[str, Any]) -> list[str]:
    warnings = result["run_config"]["judge"]["warnings"]
    return [warning for warning in warnings if "verifier timeout" in warning]


_STEPS_BODY = '[[steps]]\nname = "step-one"\n\n[[steps]]\nname = "step-two"\n\n[environment]\n'


@pytest.mark.parametrize(
    ("setup", "kwargs", "listed", "multiplier_hint"),
    [
        pytest.param(_scaffold_native_case, {}, ["case-001 (600s)"], "--timeout-multiplier 3 ", id="scaffold"),
        pytest.param(
            lambda skill: _write_native_case(skill, "[verifier]\ntimeout_sec = 180.0\n\n[environment]\n"),
            {},
            ["case-001 (180s)"],
            "--timeout-multiplier 10 ",
            id="authored-180",
        ),
        pytest.param(
            lambda skill: _write_native_case(skill, _STEPS_BODY, steps=("step-one", "step-two")),
            {},
            ["case-001 step 'step-one' (600s)", "case-001 step 'step-two' (600s)"],
            "--timeout-multiplier 3 ",
            id="steps-ignore-the-task-budget",
        ),
        pytest.param(
            lambda skill: _write_native_case(skill, "[verifier]\ntimeout_sec = 900.0\n\n[environment]\n"),
            {"timeout_multiplier": 1.5},
            ["case-001 (1350s)"],
            "--timeout-multiplier 2 ",
            id="multiplier-too-small",
        ),
    ],
)
def test_native_verifier_budget_below_the_panel_needs_warns(
    harness: panel_run._Harness,
    setup: Callable[[Path], None],
    kwargs: dict[str, Any],
    listed: list[str],
    multiplier_hint: str,
) -> None:
    setup(harness.skill)

    result = harness.run(panel_run._panel_environment(), **kwargs)

    assert "error" not in result
    assert result["run_config"]["task_source"] == "native_harbor"
    (warning,) = _budget_warnings(result)
    assert "1800s" in warning
    assert f": {', '.join(listed)}." in warning
    assert "[verifier] or [steps.verifier] timeout_sec" in warning
    assert multiplier_hint in warning
    events = harness.reporter.stage_events("judge-panel")
    assert events[-1].state == "degraded"
    assert warning in events[-1].detail
    persisted = json.loads((Path(result["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
    assert warning in persisted["judge"]["warnings"]
    # The run summary repeats it, so it stays visible when progress output is off.
    assert warning in result["warnings"]
    assert warning in json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))["warnings"]
    assert len(harness.harbor_calls) == 1


@pytest.mark.parametrize(
    ("setup", "kwargs"),
    [
        pytest.param(_scaffold_native_case, {"timeout_multiplier": 3.0}, id="scaffold-with-multiplier"),
        pytest.param(
            lambda skill: _write_native_case(skill, "[verifier]\ntimeout_sec = 1800.0\n\n[environment]\n"),
            {},
            id="authored-1800",
        ),
        pytest.param(lambda skill: _write_native_case(skill, "[environment]\n"), {}, id="panel-budget-added"),
        pytest.param(
            lambda skill: _write_native_case(
                skill,
                '[verifier]\ntimeout_sec = 180.0\n\n[[steps]]\nname = "step-one"\n\n[steps.verifier]\n'
                "timeout_sec = 1800.0\n\n[environment]\n",
                steps=("step-one",),
            ),
            {},
            id="steps-with-their-own-budget",
        ),
    ],
)
def test_native_verifier_budget_within_the_panel_needs_does_not_warn(
    harness: panel_run._Harness,
    setup: Callable[[Path], None],
    kwargs: dict[str, Any],
) -> None:
    setup(harness.skill)

    result = harness.run(panel_run._panel_environment(), **kwargs)

    assert "error" not in result
    assert result["run_config"]["judge"]["warnings"] == []
    (event,) = harness.reporter.stage_events("judge-panel")
    assert event.state == "ready"
    assert [warning for warning in result.get("warnings", []) if "verifier timeout" in warning] == []


def test_native_verifier_budget_is_not_checked_without_a_panel(harness: panel_run._Harness) -> None:
    _write_native_case(harness.skill, "[verifier]\ntimeout_sec = 180.0\n\n[environment]\n")

    result = harness.run(panel_run._nv_build_primary())

    assert "error" not in result
    assert "warnings" not in result["run_config"]["judge"]
    assert harness.reporter.stage_events("judge-panel") == []


def test_generated_tasks_never_warn_about_the_verifier_budget(harness: panel_run._Harness) -> None:
    result = harness.run(panel_run._panel_environment())

    assert "error" not in result
    assert _budget_warnings(result) == []


# --- D8: a panel run whose scored trials yield no judge statistics says so -----------------------------


def _collected(*, judge_panel: bool, status: str = "succeeded", scored: int = 1) -> dict[str, Any]:
    condition = {"execution_status": status, "execution_errors": [], "expected_attempts": 1, "scored_attempts": scored}
    skipped = {"execution_status": "skipped", "execution_errors": [], "expected_attempts": 0, "scored_attempts": 0}
    agent: dict[str, Any] = {
        "conditions": {"with_skill": condition, "without_skill": skipped},
        "execution_status": status,
        "scored_attempts": scored,
    }
    if judge_panel:
        agent["judge_panel"] = {"schema_version": "1.0", "judges": []}
    return {"execution_status": status, "execution_errors": [], "metrics": [], "agents": {"opencode": agent}}


def _statistics_warnings(result: dict[str, Any]) -> list[str]:
    return [warning for warning in result.get("warnings", []) if "Judge panel statistics" in warning]


def test_panel_run_without_judge_statistics_warns(harness: panel_run._Harness) -> None:
    harness.monkeypatch.setattr(runner, "collect_harbor_results", lambda **_kwargs: _collected(judge_panel=False))

    result = harness.run(panel_run._panel_environment())

    (warning,) = _statistics_warnings(result)
    assert "judge_panel.json" in warning
    assert "multi-step" in warning
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert warning in persisted["warnings"]
    (finished,) = harness.reporter.stage_events("run-finished")
    assert finished.state == "degraded"


@pytest.mark.parametrize(
    ("environment", "collected"),
    [
        pytest.param(panel_run._panel_environment(), _collected(judge_panel=True), id="statistics-present"),
        pytest.param(panel_run._nv_build_primary(), _collected(judge_panel=False), id="no-panel"),
        pytest.param(panel_run._panel_environment(), _collected(judge_panel=False, scored=0), id="nothing-scored"),
        pytest.param(
            panel_run._panel_environment(),
            _collected(judge_panel=False, status="failed"),
            id="condition-failed",
        ),
    ],
)
def test_judge_statistics_warning_needs_a_panel_and_scored_trials(
    harness: panel_run._Harness,
    environment: dict[str, str],
    collected: dict[str, Any],
) -> None:
    harness.monkeypatch.setattr(runner, "collect_harbor_results", lambda **_kwargs: collected)

    result = harness.run(environment)

    assert _statistics_warnings(result) == []


def test_custom_only_panel_run_never_warns_about_judge_statistics(harness: panel_run._Harness) -> None:
    (harness.skill / "evals" / "grader.py").write_text("def grade(*args, **kwargs):\n    return 1\n", encoding="utf-8")
    harness.monkeypatch.setattr(runner, "collect_harbor_results", lambda **_kwargs: _collected(judge_panel=False))

    result = harness.run(panel_run._panel_environment(), grading_mode="custom_only")

    assert _statistics_warnings(result) == []


def _staged_case(root: Path, case: str, body: str) -> None:
    (root / case).mkdir(parents=True)
    (root / case / "task.toml").write_text(_NATIVE_TASK_HEAD + body, encoding="utf-8")


def test_verifier_budget_hint_is_not_inflated_by_float_noise(tmp_path: Path) -> None:
    _staged_case(tmp_path, "case-001", "[verifier]\ntimeout_sec = 600.0\n")

    (warning,) = runner._native_verifier_budget_warnings([tmp_path], member_count=3, timeout_multiplier=0.7)

    # 0.7 * 600 is 419.99999999999994 in binary floating point; the needed multiplier is exactly 3.
    assert "case-001 (420s)" in warning
    assert "--timeout-multiplier 3 " in warning
    assert "to at least 2571.43s" in warning


def test_verifier_budget_warning_lists_a_bounded_number_of_locations(tmp_path: Path) -> None:
    for index in range(7):
        _staged_case(tmp_path, f"case-{index:03d}", "[verifier]\ntimeout_sec = 700.0\n")

    (warning,) = runner._native_verifier_budget_warnings([tmp_path, tmp_path], member_count=2, timeout_multiplier=1.0)

    assert warning.count("(700s)") == 5
    assert "case-004 (700s), 2 more." in warning
    assert "--timeout-multiplier 1.72 " in warning


@pytest.mark.parametrize(
    "body",
    [
        "[verifier]\ntimeout_sec = true\n",
        '[verifier]\ntimeout_sec = "600"\n',
        "[verifier]\ntimeout_sec = nan\n",
        "not = [valid toml\n",
        '[[steps]]\nname = "only"\n\n[steps.verifier]\ntimeout_sec = "slow"\n',
    ],
)
def test_verifier_budget_skips_values_harbor_would_reject(tmp_path: Path, body: str) -> None:
    _staged_case(tmp_path, "case-001", body)

    assert runner._native_verifier_budget_warnings([tmp_path], member_count=3, timeout_multiplier=1.0) == []


def test_verifier_budget_names_unnamed_steps_by_position(tmp_path: Path) -> None:
    _staged_case(tmp_path, "case-001", '[[steps]]\nname = ""\n\n[[steps]]\nname = "grade"\n')

    (warning,) = runner._native_verifier_budget_warnings([tmp_path], member_count=2, timeout_multiplier=1.0)

    assert "case-001 step '#1' (600s), case-001 step 'grade' (600s)" in warning


def test_verifier_budget_ignores_oversized_and_non_regular_task_files(tmp_path: Path) -> None:
    oversized = tmp_path / "case-001"
    oversized.mkdir()
    (oversized / "task.toml").write_bytes(b"#" * (runner._MAX_STAGED_TASK_TOML_BYTES + 1))
    linked = tmp_path / "case-002"
    linked.mkdir()
    (tmp_path / "real.toml").write_text("[verifier]\ntimeout_sec = 1.0\n", encoding="utf-8")
    (linked / "task.toml").symlink_to(tmp_path / "real.toml")

    assert runner._native_verifier_budget_warnings([tmp_path], member_count=3, timeout_multiplier=1.0) == []
