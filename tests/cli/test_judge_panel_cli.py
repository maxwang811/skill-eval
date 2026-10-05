# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--judge-panel`` reaches the Tier 3 engine from every CLI surface, and ``doctor`` checks the panel."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from skillevaluator import cli as cli_module
from skillevaluator.cli import cli
from skillevaluator.cli_help import GroupedOption, render_help
from skillevaluator.evaluation import EvaluationService
from skillevaluator.models.result import ValidationResult
from skillevaluator.provider_config import resolve_judge_panel_config
from skillevaluator.tier3 import commands as tier3_commands

PANEL = "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b"
# Two members with vote aggregation: valid, but resolved with an advisory warning.
EVEN_PANEL = "openai:gpt-5.6-sol,anthropic:claude-opus-5"
ENTRY_POINTS = ("tier3 evaluate", "evaluate", "tier3 PATH", "validate")
FAIL_FAST_ENTRY_POINTS = ("tier3 evaluate", "evaluate", "tier3 PATH")
_PANEL_KNOBS = {
    "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "median",
    "SKILL_EVAL_JUDGE_PANEL_QUORUM": "2",
    "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.5",
}
_CREDENTIALS = {
    "NVIDIA_API_KEY": "nvapi-test-key-never-sent",
    "OPENAI_API_KEY": "sk-test-openai-key-never-sent",
    "ANTHROPIC_API_KEY": "sk-ant-test-key-never-sent",
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("SKILL_EVAL_", "SKILLEVALUATOR_", "SKILLSPECTOR_", "AWS_")) or name in {
            "NVIDIA_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "LLM_JUDGE_MODEL",
            "LLM_JUDGE_FALLBACK_MODELS",
        }:
            monkeypatch.delenv(name)


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def skill(tmp_path: Path) -> Path:
    return _skill(tmp_path / "demo-skill", with_dataset=True)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch, credentials: None) -> list[dict[str, object]]:
    """Record every engine call; the CLI must hand the panel to ``run_harbor_eval``."""
    from skillevaluator.tier3.harbor import runner

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(
        tier3_commands,
        "run_harbor_eval",
        lambda **kwargs: (
            calls.append(kwargs) or {"execution_status": "succeeded", "execution_errors": [], "agents": {}}
        ),
    )
    # Tier 1 is not under test here; keep validate's gating tier green and fast.
    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [_passing("SCHEMA")])
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("an evaluation source exists, so no dataset may be generated"),
    )
    return calls


def _skill(path: Path, *, with_dataset: bool) -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Explain how to validate a configuration.\n---\n"
        "# Demo\nValidate the configuration, then show the result.\n",
        encoding="utf-8",
    )
    if with_dataset:
        (path / "evals").mkdir()
        (path / "evals" / "evals.json").write_text(
            json.dumps(
                {
                    "skill_name": path.name,
                    "evals": [
                        {"id": "demo-001", "prompt": "Validate this config.", "expected_output": "Validation result."}
                    ],
                }
            ),
            encoding="utf-8",
        )
    return path


def _passing(name: str) -> ValidationResult:
    result = ValidationResult(validator_name=name)
    result.add_success("checked", "Completed this check")
    return result


def _argv(entry: str, skill: Path, tmp_path: Path, *options: str, autopilot: bool = False) -> list[str]:
    results = ["--results-dir", str(tmp_path / "results")]
    if entry == "validate":
        return [
            "validate",
            str(skill),
            "--no-tier2",
            "--autopilot" if autopilot else "--no-autopilot",
            "--tier3",
            "--agents",
            "opencode",
            *results,
            "-r",
            "json",
            "-o",
            str(tmp_path / "reports"),
            *options,
        ]
    prefix = {"tier3 evaluate": ["tier3", "evaluate"], "evaluate": ["evaluate"], "tier3 PATH": ["tier3"]}[entry]
    # The direct workflow always prepares a missing dataset; it has no flag for it.
    autopilot_flag = ["--autopilot"] if autopilot and entry != "tier3 PATH" else []
    return [*prefix, str(skill), "--agents", "opencode", "--progress", "off", *autopilot_flag, *results, *options]


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _plain(text: str) -> str:
    """Join the validate view's wrapped panel rows back into plain text."""
    return _normalized(text.replace("│", " "))


def _even_panel_warning() -> str:
    """The advisory warning the even panel resolves with (the CLI must print it verbatim)."""
    panel = resolve_judge_panel_config({**os.environ, "SKILL_EVAL_JUDGE_PANEL": EVEN_PANEL})
    assert panel is not None
    (warning,) = panel.warnings
    assert "odd number of judges" in warning
    return warning


def _completed_tier3() -> ValidationResult:
    """A Tier 3 result the validate view summarizes as a finished run."""
    result = _passing("AGENT_EVAL")
    result.metadata["agent_eval"] = {"summary": {"agents_run": ["opencode"], "execution_status": "succeeded"}}
    return result


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_judge_panel_reaches_the_engine_from_every_entry_point(entry, skill, tmp_path, engine) -> None:
    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", PANEL))

    assert result.exit_code == 0, result.output
    assert [call["judge_panel"] for call in engine] == [PANEL]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_omitted_judge_panel_reaches_the_engine_as_none(entry, skill, tmp_path, engine) -> None:
    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path))

    assert result.exit_code == 0, result.output
    assert [call["judge_panel"] for call in engine] == [None]


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
def test_invalid_judge_panel_fails_before_the_engine_runs(entry, skill, tmp_path, engine) -> None:
    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", "nope"))

    assert result.exit_code == 1, result.output
    output = _normalized(result.output)
    assert "Invalid judge panel configuration: --judge-panel entry 'nope' must use provider:model form" in output
    assert "Traceback" not in result.output
    assert engine == []


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("openai:gpt-5.6-sol,,anthropic:claude-opus-5", "--judge-panel entry 2 is empty"),
        ("foo:bar", "--judge-panel provider 'foo' must be one of"),
        ("openai:gpt-5.6-sol,openai:gpt-5.6-sol", "--judge-panel lists openai:gpt-5.6-sol more than once"),
        (",".join(f"openai:model-{index}" for index in range(6)), "--judge-panel supports at most 5 judges"),
    ],
)
def test_invalid_cli_panel_errors_name_the_flag(entry, value, expected, skill, tmp_path, engine) -> None:
    # The operator typed the value on the command line, so the error points
    # there rather than at an environment variable they may never have set.
    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", value))

    assert result.exit_code == 1, result.output
    output = _normalized(result.output)
    assert expected in output
    assert "SKILL_EVAL_JUDGE_PANEL" not in output
    assert engine == []


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("options", "remove_from"),
    [
        (("--judge-panel", PANEL), "remove anthropic:claude-opus-5 from --judge-panel."),
        ((), "remove anthropic:claude-opus-5 from SKILL_EVAL_JUDGE_PANEL (or --judge-panel)."),
    ],
)
def test_missing_member_credential_fails_fast_and_names_the_variable(
    entry, options, remove_from, skill, tmp_path, engine, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    if not options:
        monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, *options))

    assert result.exit_code == 1, result.output
    output = _normalized(result.output)
    assert "anthropic:claude-opus-5 requires ANTHROPIC_API_KEY" in output
    assert remove_from in output
    assert engine == []


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
def test_knob_errors_keep_their_environment_names_with_the_flag(entry, skill, tmp_path, engine, monkeypatch) -> None:
    # The knobs still come from the environment, so their errors name it.
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_QUORUM", "9")

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", PANEL))

    assert result.exit_code == 1, result.output
    assert "SKILL_EVAL_JUDGE_PANEL_QUORUM must be an integer between 1 and 3" in _normalized(result.output)
    assert engine == []


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
def test_gateway_member_with_a_native_primary_fails_fast_naming_the_flag(
    entry, skill, tmp_path, engine, monkeypatch
) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("SKILL_EVAL_LLM_API_KEY", "sk-gateway-test-key-never-sent")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example.test/v1")

    result = CliRunner().invoke(
        cli, _argv(entry, skill, tmp_path, "--judge-panel", "openai-compatible:gateway-model,openai:gpt-5.6-sol")
    )

    assert result.exit_code == 1, result.output
    output = _normalized(result.output)
    assert "--judge-panel cannot include an openai-compatible member while the openai primary is selected" in output
    assert "sk-gateway-test-key-never-sent" not in result.output
    assert _CREDENTIALS["OPENAI_API_KEY"] not in result.output
    assert engine == []


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"SKILL_EVAL_JUDGE_PANEL": "nope"}, "SKILL_EVAL_JUDGE_PANEL entry 'nope' must use provider:model form"),
        ({"SKILL_EVAL_JUDGE_PANEL_QUORUM": "2"}, "SKILL_EVAL_JUDGE_PANEL is not set"),
        # A blank panel in the environment does not excuse its knobs; only an
        # explicit --judge-panel '' opts out of them.
        (
            {"SKILL_EVAL_JUDGE_PANEL": "   ", "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "median"},
            "SKILL_EVAL_JUDGE_PANEL_AGGREGATION configure(s) a judge panel",
        ),
    ],
)
def test_invalid_environment_panel_fails_fast_without_the_flag(
    entry, environment, expected, skill, tmp_path, engine, monkeypatch
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path))

    assert result.exit_code == 1, result.output
    output = _normalized(result.output)
    assert "Invalid judge panel configuration" in output
    assert expected in output
    assert engine == []


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("flag_value", [PANEL, ""])
def test_cli_value_replaces_the_environment_panel(entry, flag_value, skill, tmp_path, engine, monkeypatch) -> None:
    # The flag replaces SKILL_EVAL_JUDGE_PANEL outright, so an unusable host
    # value is never consulted; a blank flag runs the single standard judge.
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", "nope")

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", flag_value))

    assert result.exit_code == 0, result.output
    assert [call["judge_panel"] for call in engine] == [flag_value]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_cli_panel_opts_out_of_the_host_panel_and_its_knobs(
    entry, blank, skill, tmp_path, engine, monkeypatch
) -> None:
    # An operator who exports a panel together with its knobs (and even a
    # single-judge model override) can still run one single-judge evaluation.
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    for name, value in _PANEL_KNOBS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_JUDGE_MODEL", "openai/gpt-5.6-sol")

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", blank))

    assert result.exit_code == 0, result.output
    assert "judge panel" not in _normalized(result.output).lower()
    assert [call["judge_panel"] for call in engine] == [blank]


@pytest.mark.parametrize("blank", ["", " \t "])
def test_resolve_judge_panel_option_treats_a_blank_value_as_a_full_opt_out(blank, credentials, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    # Knobs that would be invalid for any panel are ignored once the panel is off.
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_AGGREGATION", "majority")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_QUORUM", "9")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT", "2")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    assert tier3_commands.resolve_judge_panel_option(blank) is None


def test_resolve_judge_panel_option_returns_the_resolved_panel_and_its_warnings(credentials) -> None:
    panel = tier3_commands.resolve_judge_panel_option(f" {EVEN_PANEL.replace('openai:', 'OpenAI:')} ")

    assert panel is not None
    assert [member.label for member in panel.members] == ["openai:gpt-5.6-sol", "anthropic:claude-opus-5"]
    assert panel.warnings == (_even_panel_warning(),)
    assert tier3_commands.resolve_judge_panel_option(None) is None


def test_resolve_judge_panel_option_keeps_a_blank_environment_panel_with_knobs_an_error(
    credentials, monkeypatch
) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", "  ")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL_AGGREGATION", "median")

    with pytest.raises(ValueError, match="SKILL_EVAL_JUDGE_PANEL_AGGREGATION configure"):
        tier3_commands.resolve_judge_panel_option(None)


def test_direct_workflow_rejects_an_invalid_panel_before_generating_a_dataset(tmp_path, engine) -> None:
    # The direct workflow generates a missing dataset with a paid LLM call, so
    # its preflight must reject the panel first.
    bare_skill = _skill(tmp_path / "bare-skill", with_dataset=False)

    result = CliRunner().invoke(cli, _argv("tier3 PATH", bare_skill, tmp_path, "--judge-panel", "nope"))

    assert result.exit_code == 1, result.output
    assert "Invalid judge panel configuration" in _normalized(result.output)
    assert not (bare_skill / "evals").exists()
    assert engine == []


def _custom_grader_skill(path: Path, *, config_name: str = "config.yml") -> Path:
    """A skill without a dataset whose own config selects its custom grader."""
    skill = _skill(path, with_dataset=False)
    (skill / "evals").mkdir()
    (skill / "evals" / config_name).write_text("schema_version: 1\ngrading:\n  mode: default_plus_custom\n")
    (skill / "evals" / "grader.py").write_text("print('{}')\n")
    return skill


def _record_generation(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    generated: list[Path] = []

    def _generate(_service, skill_path: Path, *, use_llm: bool) -> None:
        generated.append(skill_path)
        case = {"id": "demo-001", "prompt": "Validate this config.", "expected_output": "Validation result."}
        (skill_path / "evals" / "evals.json").write_text(json.dumps({"skill_name": skill_path.name, "evals": [case]}))

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", _generate)
    return generated


@pytest.mark.parametrize(
    ("panel_input", "config_name"),
    [("flag", "config.yml"), ("environment", "config.yml"), ("flag", "config.yaml")],
)
def test_direct_workflow_refuses_a_skill_selected_custom_grader_before_generating_a_dataset(
    panel_input, config_name, tmp_path, engine, monkeypatch
) -> None:
    # The engine refuses this too, but only after the workflow's paid starter-case generation.
    from skillevaluator.tier3.harbor.runner import _skill_selected_custom_grader_error

    generated = _record_generation(monkeypatch)
    custom_skill = _custom_grader_skill(tmp_path / "custom-skill", config_name=config_name)
    if panel_input == "environment":
        monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    options = ("--judge-panel", PANEL) if panel_input == "flag" else ()

    result = CliRunner().invoke(cli, _argv("tier3 PATH", custom_skill, tmp_path, *options))

    assert result.exit_code == 1, result.output
    source = "--judge-panel" if panel_input == "flag" else "SKILL_EVAL_JUDGE_PANEL"
    expected = _skill_selected_custom_grader_error(source, f"evals/{config_name}")
    assert _normalized(expected) in _normalized(result.output)
    assert generated == []
    assert not (custom_skill / "evals" / "evals.json").exists()
    assert engine == []


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(("--judge-panel", PANEL, "--grading-mode", "default_plus_custom"), id="operator-accepts"),
        pytest.param(("--judge-panel", PANEL, "--grading-mode", "default"), id="operator-skips-the-grader"),
        pytest.param(("--judge-panel", PANEL, "--grading-mode", "custom_only"), id="no-standard-judges"),
        pytest.param(("--judge-panel", ""), id="panel-off"),
        pytest.param((), id="no-panel"),
    ],
)
def test_direct_workflow_keeps_a_skill_selected_custom_grader_the_operator_allows(
    options, tmp_path, engine, monkeypatch
) -> None:
    generated = _record_generation(monkeypatch)
    custom_skill = _custom_grader_skill(tmp_path / "custom-skill")

    result = CliRunner().invoke(cli, _argv("tier3 PATH", custom_skill, tmp_path, *options))

    assert result.exit_code == 0, result.output
    assert generated == [custom_skill]
    assert len(engine) == 1


@pytest.mark.parametrize("entry", ["tier3 evaluate", "evaluate"])
def test_evaluate_autopilot_rejects_an_invalid_panel_before_generating_a_dataset(
    entry, tmp_path, engine, monkeypatch
) -> None:
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("an invalid panel must stop autopilot before any generation"),
    )
    bare_skill = _skill(tmp_path / "bare-skill", with_dataset=False)

    result = CliRunner().invoke(cli, _argv(entry, bare_skill, tmp_path, "--judge-panel", "nope", autopilot=True))

    assert result.exit_code == 1, result.output
    assert "Invalid judge panel configuration: --judge-panel entry 'nope'" in _normalized(result.output)
    assert "Autopilot" not in result.output
    assert not (bare_skill / "evals").exists()
    assert engine == []


@pytest.mark.parametrize(("blocking", "expected_exit_code"), [((), 0), (("--block-on-agent-eval",), 1)])
def test_validate_rejects_an_invalid_panel_before_autopilot(
    tmp_path, engine, monkeypatch, blocking, expected_exit_code
) -> None:
    # Tier 3 is advisory in validate: the panel error skips it, without a
    # paid dataset generation, and is reported like any other Tier 3 failure.
    monkeypatch.setattr(
        EvaluationService,
        "create_autopilot_dataset",
        lambda *_args, **_kwargs: pytest.fail("an invalid panel must stop autopilot before any generation"),
    )
    bare_skill = _skill(tmp_path / "bare-skill", with_dataset=False)

    result = CliRunner().invoke(
        cli, _argv("validate", bare_skill, tmp_path, "--judge-panel", "nope", *blocking, autopilot=True)
    )

    assert result.exit_code == expected_exit_code, result.output
    assert not (bare_skill / "evals").exists()
    assert engine == []
    (report,) = (tmp_path / "reports").glob("*.json")
    report_text = report.read_text(encoding="utf-8")
    assert "Tier 3 live evaluation skipped: Invalid judge panel configuration: --judge-panel entry 'nope'" in (
        report_text
    )


@pytest.mark.parametrize("entry", ["tier3 evaluate", "evaluate", "tier3 PATH", "validate"])
def test_autopilot_still_prepares_a_dataset_for_a_valid_panel(entry, tmp_path, engine, monkeypatch) -> None:
    generated: list[Path] = []

    def _generate(_service, skill_path: Path, *, use_llm: bool) -> Path:
        generated.append(skill_path)
        return _skill(tmp_path / "template", with_dataset=True).joinpath("evals").rename(skill_path / "evals")

    monkeypatch.setattr(EvaluationService, "create_autopilot_dataset", _generate)
    bare_skill = _skill(tmp_path / "bare-skill", with_dataset=False)

    result = CliRunner().invoke(cli, _argv(entry, bare_skill, tmp_path, "--judge-panel", PANEL, autopilot=True))

    assert result.exit_code == 0, result.output
    assert generated == [bare_skill]
    assert [call["judge_panel"] for call in engine] == [PANEL]


@pytest.mark.parametrize(("blocking", "expected_exit_code"), [((), 0), (("--block-on-agent-eval",), 1)])
def test_validate_reports_an_invalid_panel_as_tier3_failure_without_running_the_engine(
    skill, tmp_path, engine, blocking, expected_exit_code
) -> None:
    result = CliRunner().invoke(cli, _argv("validate", skill, tmp_path, "--judge-panel", "nope", *blocking))

    assert result.exit_code == expected_exit_code, result.output
    assert engine == []
    (report,) = (tmp_path / "reports").glob("*.json")
    assert "Invalid judge panel configuration: --judge-panel entry 'nope'" in report.read_text(encoding="utf-8")


# --- panel warnings ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", FAIL_FAST_ENTRY_POINTS)
@pytest.mark.parametrize("progress", ["off", "plain"])
def test_evaluate_prints_panel_warnings_in_every_progress_mode(entry, progress, skill, tmp_path, engine) -> None:
    # The engine also reports the warning as progress, which --progress off drops.
    warning = _even_panel_warning()

    result = CliRunner().invoke(cli, _argv(entry, skill, tmp_path, "--judge-panel", EVEN_PANEL, "--progress", progress))

    assert result.exit_code == 0, result.output
    assert result.stderr.count(f"Warning: {warning}") == 1
    assert [call["judge_panel"] for call in engine] == [EVEN_PANEL]


def test_evaluate_prints_warnings_for_an_environment_panel(skill, tmp_path, engine, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", EVEN_PANEL)

    result = CliRunner().invoke(cli, _argv("tier3 evaluate", skill, tmp_path))

    assert result.exit_code == 0, result.output
    assert result.stderr.count(f"Warning: {_even_panel_warning()}") == 1


@pytest.mark.parametrize(
    "options",
    [(), ("--judge-panel", PANEL), ("--judge-panel", EVEN_PANEL, "--grading-mode", "custom_only")],
    ids=["no-panel", "odd-panel", "custom-only"],
)
def test_evaluate_prints_no_panel_warning_without_one_to_give(options, skill, tmp_path, engine) -> None:
    # custom_only grading runs no standard judges, so the panel's advice does not apply.
    result = CliRunner().invoke(cli, _argv("tier3 evaluate", skill, tmp_path, *options))

    assert result.exit_code == 0, result.output
    assert "Warning:" not in result.stderr
    assert len(engine) == 1


@pytest.fixture
def completed_tier3(monkeypatch: pytest.MonkeyPatch, credentials: None) -> list[object]:
    """Let validate's Tier 3 finish, so its view keeps the rows a finished tier shows."""
    calls: list[object] = []

    def _tier3(_path: Path, **kwargs) -> ValidationResult:
        calls.append(kwargs.get("judge_panel"))
        return _completed_tier3()

    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [_passing("SCHEMA")])
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)
    return calls


def test_validate_keeps_panel_warnings_in_the_finished_tier3_view(skill, tmp_path, completed_tier3) -> None:
    warning = _even_panel_warning()

    result = CliRunner().invoke(cli, _argv("validate", skill, tmp_path, "--judge-panel", EVEN_PANEL))

    assert result.exit_code == 0, result.output
    assert completed_tier3 == [EVEN_PANEL]
    assert _plain(result.stdout).count(f"judge panel {warning}") == 1
    assert "Warning:" not in result.stderr


def test_validate_verbose_prints_panel_warnings_to_stderr(skill, tmp_path, completed_tier3, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", EVEN_PANEL)

    result = CliRunner().invoke(cli, _argv("validate", skill, tmp_path, "--verbose"))

    assert result.exit_code == 0, result.output
    assert completed_tier3 == [None]
    assert result.stderr.count(f"Warning: {_even_panel_warning()}") == 1


@pytest.mark.parametrize("verbose", [(), ("--verbose",)], ids=["view", "verbose"])
@pytest.mark.parametrize(
    "options",
    [(), ("--judge-panel", PANEL), ("--judge-panel", EVEN_PANEL, "--grading-mode", "custom_only")],
    ids=["no-panel", "odd-panel", "custom-only"],
)
def test_validate_shows_no_panel_warning_without_one_to_give(
    verbose, options, skill, tmp_path, completed_tier3
) -> None:
    result = CliRunner().invoke(cli, _argv("validate", skill, tmp_path, *options, *verbose))

    assert result.exit_code == 0, result.output
    assert "judge panel" not in _plain(result.output).lower()
    assert "Warning:" not in result.stderr


@pytest.mark.parametrize("args", [["doctor", "--help"], ["tier3", "doctor", "--help"]])
def test_verify_models_help_mentions_the_judge_panel_members(args: list[str]) -> None:
    result = CliRunner().invoke(cli, args, terminal_width=240)

    assert result.exit_code == 0, result.output
    assert "with a judge panel, each panel member" in _normalized(result.output)


@pytest.mark.parametrize(
    "args",
    [["tier3", "evaluate", "--help"], ["evaluate", "--help"], ["tier3", "--help"], ["validate", "--help"]],
)
def test_help_describes_the_judge_panel_option(args: list[str]) -> None:
    result = CliRunner().invoke(cli, args, terminal_width=240)

    assert result.exit_code == 0, result.output
    output = _normalized(result.output)
    assert "--judge-panel TEXT" in output
    assert "Cross-model judge panel for the three LLM metrics" in output
    assert "overrides SKILL_EVAL_JUDGE_PANEL" in output


def test_validate_lists_judge_panel_in_the_tier3_option_group() -> None:
    (param,) = [param for param in cli_module.validate.params if param.name == "judge_panel"]
    assert isinstance(param, GroupedOption)
    assert param.help_group == cli_module._TIER3_GROUP

    text = render_help(cli_module.validate, click.Context(cli_module.validate, info_name="validate"), width=240)
    # The Tier 3 section ends at the next option heading or at the epilog.
    tier3_start = text.index(f"{cli_module._TIER3_GROUP}:")
    boundaries = [text.index("Content types (--type):")] + [
        text.index(f"{heading}:")
        for heading in ("Options", cli_module._RUN_GROUP, cli_module._TIER1_GROUP, cli_module._TIER2_GROUP)
    ]
    tier3_end = min(position for position in boundaries if position > tier3_start)
    assert text.count("--judge-panel") == 1
    assert "--judge-panel" in text[tier3_start:tier3_end]


def test_judge_panel_option_is_shared_by_every_evaluate_registration() -> None:
    from skillevaluator.tier3.workflow import build_tier3_workflow

    for command in (cli_module.evaluate, cli_module._tier3_evaluate_visible, build_tier3_workflow(cli_module.evaluate)):
        (param,) = [param for param in command.params if param.name == "judge_panel"]
        assert param.opts == ["--judge-panel"]
        assert param.default is None


@pytest.mark.parametrize(
    ("options", "expected"),
    [((), None), (("--judge-panel", PANEL), PANEL), (("--judge-panel", ""), "")],
)
def test_catalog_child_argv_forwards_the_judge_panel(tmp_path: Path, options, expected) -> None:
    skill_dir = tmp_path / "catalog" / "demo"
    skill_dir.mkdir(parents=True)
    ctx = cli_module.validate.make_context("validate", [str(skill_dir), "--workers", "2", *options])

    argv = cli_module._catalog_child_argv_from_ctx(ctx, skill_dir, tmp_path / "out" / "demo")

    if expected is None:
        assert "--judge-panel" not in argv
    else:
        assert argv[argv.index("--judge-panel") + 1] == expected
        # The child parses the rebuilt argv back to the same panel value.
        child = cli_module.validate.make_context("validate", argv[1:])
        assert child.params["judge_panel"] == expected


def test_sequential_catalog_forwards_the_judge_panel_to_every_skill(
    tmp_path: Path, monkeypatch, credentials: None
) -> None:
    # Each child resolves the panel before Tier 3, so the members need credentials.
    catalog = tmp_path / "catalog"
    for name in ("alpha", "beta"):
        _skill(catalog / name, with_dataset=True)
    captured: list[tuple[str, object]] = []

    def _tier3(path: Path, **kwargs) -> ValidationResult:
        captured.append((path.name, kwargs.get("judge_panel")))
        return _passing("AGENT_EVAL")

    monkeypatch.setattr(cli_module, "run_validation", lambda *_args, **_kwargs: [_passing("SCHEMA")])
    monkeypatch.setattr(cli_module, "_run_agent_eval_or_skip", _tier3)

    result = CliRunner().invoke(
        cli,
        [
            "validate",
            str(catalog),
            "--no-tier2",
            "--no-autopilot",
            "--tier3",
            "--judge-panel",
            PANEL,
            "-o",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(captured) == [("alpha", PANEL), ("beta", PANEL)]


@pytest.mark.parametrize(
    ("options", "environment", "expected_source"),
    [
        (("--judge-panel", " openai:gpt-5.6-sol , ANTHROPIC:claude-opus-5 "), {}, "--judge-panel"),
        ((), {"SKILL_EVAL_JUDGE_PANEL": "openai:gpt-5.6-sol,anthropic:claude-opus-5"}, "SKILL_EVAL_JUDGE_PANEL"),
    ],
)
def test_tier3_evaluate_delivers_the_panel_to_the_harbor_verifier(
    skill, tmp_path, credentials, monkeypatch, options, environment, expected_source
) -> None:
    """Run the real engine and task staging; only Harbor's process, probes, and collection are faked."""
    import subprocess

    from skillevaluator.tier3.harbor import runner, runtime_preflight

    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    harbor_calls: list[tuple[list[str], dict[str, str]]] = []
    probed: list[str] = []
    real_run = subprocess.run

    def _harbor_run(command, *args, **kwargs):
        if isinstance(command, list) and command[1:2] == ["run"] and "--job-name" in command:
            harbor_calls.append((list(command), dict(kwargs["env"])))
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    def _probe(provider):
        probed.append(f"{provider.provider}:{provider.model}")
        return runtime_preflight.ModelProbeResult(True, provider.provider, provider.model, "model is available")

    def _render(_skill_path, run_dir, **_kwargs):
        report = run_dir / "report.html"
        report.write_text("<html></html>\n", encoding="utf-8")
        return report

    monkeypatch.setattr(runner.subprocess, "run", _harbor_run)
    monkeypatch.setattr(runner, "_check_prerequisites", lambda **_kwargs: [])
    monkeypatch.setattr(runner, "_validate_harbor_job_result", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        runner,
        "collect_harbor_results",
        lambda **_kwargs: {"execution_status": "succeeded", "execution_errors": [], "metrics": [], "agents": {}},
    )
    monkeypatch.setattr(runner, "render_agent_eval_html_report", _render)
    monkeypatch.setattr(runtime_preflight, "probe_model", _probe)

    result = CliRunner().invoke(
        cli,
        [
            "tier3",
            "evaluate",
            str(skill),
            "--agents",
            "opencode",
            "--skip-baseline",
            "--no-agent-runtime-preflight",
            "--harbor-keep-jobs",
            "--progress",
            "off",
            "--results-dir",
            str(tmp_path / "results"),
            *options,
        ],
    )

    assert result.exit_code == 0, result.output
    ((command, harbor_env),) = harbor_calls
    verifier_env = dict(
        command[index + 1].split("=", 1) for index, part in enumerate(command) if part == "--verifier-env"
    )
    # The normalized panel reaches the verifier through job-level placeholders.
    assert harbor_env["SKILL_EVAL_JUDGE_PANEL"] == "openai:gpt-5.6-sol,anthropic:claude-opus-5"
    assert harbor_env["SKILL_EVAL_JUDGE_PANEL_AGGREGATION"] == "vote"
    assert harbor_env["SKILL_EVAL_JUDGE_PANEL_QUORUM"] == "2"
    assert verifier_env["SKILL_EVAL_JUDGE_PANEL"] == "${SKILL_EVAL_JUDGE_PANEL}"
    # Judge-only keys never sit under the standard names Harbor's agents read.
    assert "OPENAI_API_KEY" not in harbor_env
    assert "ANTHROPIC_API_KEY" not in harbor_env
    assert verifier_env["OPENAI_API_KEY"].startswith("${SKILLEVALUATOR_JUDGE_PANEL__")
    assert sorted(probed) == [
        "anthropic:claude-opus-5",
        "nv_build:nvidia/nemotron-3-super-120b-a12b",
        "openai:gpt-5.6-sol",
    ]

    (run_config_path,) = (tmp_path / "results").rglob("run_config.json")
    run_config_text = run_config_path.read_text(encoding="utf-8")
    judge = json.loads(run_config_text)["judge"]
    assert judge["mode"] == "panel"
    assert judge["source"] == expected_source
    assert [member["label"] for member in judge["panel"]] == ["openai:gpt-5.6-sol", "anthropic:claude-opus-5"]
    for secret in _CREDENTIALS.values():
        assert secret not in run_config_text


# --- doctor -----------------------------------------------------------------------------------------------------


_DOCTOR_PANEL = "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:meta/llama-3.3-70b-instruct"


@pytest.fixture
def doctor_console(monkeypatch: pytest.MonkeyPatch, credentials: None) -> Console:
    console = Console(file=io.StringIO(), record=True, width=240)
    monkeypatch.setattr(tier3_commands, "console", console)
    monkeypatch.setattr(tier3_commands, "_check_prerequisites", lambda **_kwargs: [])
    return console


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Fake the live catalog probe: classify by model, defaulting to verified."""
    from skillevaluator.tier3.harbor import runtime_preflight
    from skillevaluator.tier3.harbor.runtime_preflight import CredentialProbeDisposition, ModelProbeResult

    calls: list[object] = []
    dispositions = {
        "gpt-5.6-sol": CredentialProbeDisposition.VERIFIED,
        "claude-opus-5": CredentialProbeDisposition.DEGRADED,
        "meta/llama-3.3-70b-instruct": CredentialProbeDisposition.FATAL,
    }

    def _probe(provider):
        calls.append(provider)
        ok = provider.model != "meta/llama-3.3-70b-instruct"
        detail = "model is available" if ok else "model catalog returned HTTP 401"
        return ModelProbeResult(ok, provider.provider, provider.model, detail)

    monkeypatch.setattr(runtime_preflight, "probe_model", _probe)
    monkeypatch.setattr(
        runtime_preflight,
        "credential_probe_disposition",
        lambda provider, _probe_result: dispositions.get(provider.model, CredentialProbeDisposition.VERIFIED),
    )
    return calls


def _doctor_row(text: str, check: str) -> tuple[str, str]:
    """Return ``(status, details)`` for the doctor row whose check name is ``check``."""
    lines = [line.strip() for line in text.splitlines()]
    matches = [line for line in lines if line.startswith(f"{check} ")]
    assert len(matches) == 1, text
    status, _, details = matches[0][len(check) :].strip().partition(" ")
    return status, details.strip()


def test_doctor_omits_the_judge_panel_without_a_panel(doctor_console, probes) -> None:
    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode", "--verify-models"])

    assert result.exit_code == 0, result.output
    text = doctor_console.export_text()
    assert "Judge panel" not in text
    assert "judge " not in text
    assert [provider.provider for provider in probes] == ["nv_build"]


def test_doctor_lists_the_configured_judge_panel(doctor_console, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", f" {PANEL.replace(',', ' , ')} ")

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode"])

    assert result.exit_code == 0, result.output
    status, details = _doctor_row(doctor_console.export_text(), "Judge panel")
    assert status == "pass"
    assert details.startswith("openai:gpt-5.6-sol, anthropic:claude-opus-5, nv_build:nvidia/nemotron-3-super-120b-a12b")
    assert "vote, quorum 2" in details


def test_health_check_also_lists_the_judge_panel(doctor_console, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)

    result = CliRunner().invoke(cli, ["health-check", "--agents", "opencode"])

    assert result.exit_code == 0, result.output
    assert _doctor_row(doctor_console.export_text(), "Judge panel")[0] == "pass"


def test_doctor_warns_about_an_even_vote_panel(doctor_console, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", "openai:gpt-5.6-sol,anthropic:claude-opus-5")

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode"])

    assert result.exit_code == 0, result.output
    status, details = _doctor_row(doctor_console.export_text(), "Judge panel")
    assert status == "warn"
    assert "even number of members" in details


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        ("SKILL_EVAL_JUDGE_PANEL", "nope", "'nope' must use provider:model form"),
        ("SKILL_EVAL_JUDGE_PANEL_AGGREGATION", "vote", "SKILL_EVAL_JUDGE_PANEL is not set"),
    ],
)
def test_doctor_fails_on_an_invalid_judge_panel(doctor_console, probes, monkeypatch, variable, value, expected) -> None:
    monkeypatch.setenv(variable, value)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode", "--verify-models"])

    assert result.exit_code == 1, result.output
    text = doctor_console.export_text()
    status, _details = _doctor_row(text, "Judge panel")
    assert status == "fail"
    assert expected in _normalized(text)
    # Nothing to probe: the members could not be resolved.
    assert [provider.provider for provider in probes] == ["nv_build"]


def test_doctor_fails_when_a_member_credential_is_missing(doctor_console, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    monkeypatch.delenv("OPENAI_API_KEY")

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode"])

    assert result.exit_code == 1, result.output
    text = doctor_console.export_text()
    assert _doctor_row(text, "Judge panel")[0] == "fail"
    assert "openai:gpt-5.6-sol requires OPENAI_API_KEY" in _normalized(text)
    assert _CREDENTIALS["OPENAI_API_KEY"] not in text


def test_doctor_verify_models_probes_every_panel_member(doctor_console, probes, monkeypatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", _DOCTOR_PANEL)

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode", "--verify-models"])

    assert result.exit_code == 1, result.output
    text = doctor_console.export_text()
    assert _doctor_row(text, "opencode model")[0] == "pass"
    assert _doctor_row(text, "judge openai:gpt-5.6-sol") == ("pass", "model is available")
    assert _doctor_row(text, "judge anthropic:claude-opus-5") == (
        "warn",
        "model is available; catalog access does not verify runtime credentials for this endpoint",
    )
    assert _doctor_row(text, "judge nv_build:meta/llama-3.3-70b-instruct") == (
        "fail",
        "model catalog returned HTTP 401",
    )
    for secret in _CREDENTIALS.values():
        assert secret not in text

    panel = resolve_judge_panel_config()
    assert panel is not None
    # The agent is probed first, then each member through its own resolved route.
    assert probes[1:] == [member.provider_config() for member in panel.members]


def test_doctor_verify_models_probes_the_panel_without_an_agent_plan(doctor_console, probes, monkeypatch) -> None:
    # Ambiguous primary provider: the agent plan is unavailable, but every
    # member still has its own credential and can be probed.
    monkeypatch.delenv("SKILL_EVAL_LLM_PROVIDER")
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", "openai:gpt-5.6-sol")

    result = CliRunner().invoke(cli, ["doctor", "--agents", "opencode", "--verify-models"])

    assert result.exit_code == 1, result.output
    text = doctor_console.export_text()
    assert _doctor_row(text, "provider model")[0] == "fail"
    assert _doctor_row(text, "Judge panel")[0] == "pass"
    assert _doctor_row(text, "judge openai:gpt-5.6-sol") == ("pass", "model is available")
    assert [(provider.provider, provider.model) for provider in probes] == [("openai", "gpt-5.6-sol")]
