# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-staging hardening around the cross-model judge panel.

- Every staged Dockerfile blanks the four ``SKILL_EVAL_JUDGE_PANEL*`` names after all
  authored layers, so image ENV from a skill cannot switch on a judge panel. The
  verifier reads a blank value as unset, so runs without a panel keep their artifacts.
- With a panel, a native task must keep its verifier in the evaluator-staged agent
  container: a separate verifier environment would receive every member credential.
- The native ``env`` table walk is iterative, so deep dotted TOML headers still stage.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
import tomllib
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import MockUrllibResponse
from tests.test_harbor_runner_judge_panel import _Harness, _nv_build_primary, _panel_environment

from skillevaluator.provider_config import JUDGE_PANEL_ENV_VARS
from skillevaluator.tier3.harbor.adapter import (
    _dockerfile_logical_lines_from_content,
    _native_task_workdir,
    _toml_env_tables,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)

_EVAL_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
)

# The exact evaluator-owned reset line (post-review decision D4).
_JUDGE_PANEL_RESET = (
    'ENV SKILL_EVAL_JUDGE_PANEL="" SKILL_EVAL_JUDGE_PANEL_AGGREGATION="" '
    'SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT="" SKILL_EVAL_JUDGE_PANEL_QUORUM=""'
)
_POLICY_RESET = 'ENV CLAUDE_CODE_DISABLE_POLICY_SKILLS="1"'
_SKILL_JUDGE = "meta/llama-3.2-1b-instruct"
# What a hostile skill image would like the verifier to see.
_HOSTILE_PANEL_ENV = {
    "SKILL_EVAL_JUDGE_PANEL": f"nv_build:{_SKILL_JUDGE}",
    "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "mean",
    "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "1",
    "SKILL_EVAL_JUDGE_PANEL_QUORUM": "1",
}
# Both ENV forms Docker accepts, including a continued multi-pair instruction.
_HOSTILE_DOCKERFILE = (
    "FROM python:3.12-slim\n"
    f"ENV SKILL_EVAL_JUDGE_PANEL=nv_build:{_SKILL_JUDGE}\n"
    "ENV SKILL_EVAL_JUDGE_PANEL_QUORUM 1\n"
    'ENV SKILL_EVAL_JUDGE_PANEL_AGGREGATION="mean" \\\n'
    "    SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT=1\n"
)
_CONTROL_DOCKERFILE = "FROM python:3.12-slim\n"
_BASE_IMAGE = "skillevaluator-base:hardening-test"
_STAGED_VERIFIER_ENV = {"SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}", "NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}
_PANEL_VERIFIER_TIMEOUT = 1800.0
_SKILL_MD = "---\nname: demo-skill\ndescription: Demo skill.\n---\n# Demo\n"
_ENTRIES = [{"id": "case-001", "question": "Run the case.", "expected_answer": "ok", "files": []}]
_NATIVE_TASK_HEAD = (
    'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
)


def _generated_skill(root: Path, dockerfile: str | None) -> Path:
    skill = root / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(json.dumps(_ENTRIES), encoding="utf-8")
    if dockerfile is not None:
        (skill / "evals" / "environment").mkdir()
        (skill / "evals" / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return skill


def _native_skill(root: Path, task_body: str, *, dockerfile: str | None = None, custom_only: bool = False) -> Path:
    skill = root / "demo-skill"
    task = skill / "evals" / "harbor" / "case-001"
    task.mkdir(parents=True)
    (skill / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (skill / "evals" / "evals.json").write_text(json.dumps(_ENTRIES), encoding="utf-8")
    (skill / "evals" / "config.yml").write_text(
        "schema_version: 1\nharbor:\n  task_source: native_harbor\n",
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (task / "task.toml").write_text(_NATIVE_TASK_HEAD + task_body, encoding="utf-8")
    if dockerfile is not None:
        (task / "environment").mkdir()
        (task / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    if custom_only:
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8")
    return skill


def _image_env(dockerfile: str, *, inherited: Mapping[str, str]) -> dict[str, str]:
    """Fold a Dockerfile's ENV instructions into the environment of its final image.

    Every stage starts from ``inherited``, which stands in for whatever ENV its base
    image already carries.
    """
    environment: dict[str, str] = {}
    for line in _dockerfile_logical_lines_from_content(dockerfile):
        instruction, *payload = line.split(None, 1)
        if instruction.upper() == "FROM":
            environment = dict(inherited)
        elif instruction.upper() == "ENV":
            words = shlex.split(payload[0])
            if all("=" in word for word in words):
                environment.update(dict(word.split("=", 1) for word in words))
            else:
                environment[words[0]] = " ".join(words[1:])
    return environment


def _panel_names(environment: Mapping[str, str]) -> dict[str, str | None]:
    return {name: environment.get(name) for name in sorted(JUDGE_PANEL_ENV_VARS)}


# --- Staged Dockerfiles blank image judge-panel ENV (D4) -------------------------------------------


def _stage_generated(*, custom_dockerfile: bool = True, **options: Any) -> Callable[[Path, str], Path]:
    def stage(root: Path, dockerfile: str) -> Path:
        skill = _generated_skill(root, dockerfile if custom_dockerfile else None)
        (task,) = generate_harbor_tasks(skill, root / "staged", verifier_env=dict(_STAGED_VERIFIER_ENV), **options)
        return task

    return stage


def _stage_native(*, custom_dockerfile: bool = True, **options: Any) -> Callable[[Path, str], Path]:
    def stage(root: Path, dockerfile: str) -> Path:
        custom_only = options.get("grading_mode") == "custom_only"
        skill = _native_skill(
            root,
            "[environment]\n",
            dockerfile=dockerfile if custom_dockerfile else None,
            custom_only=custom_only,
        )
        (task,) = stage_native_harbor_tasks(skill, root / "staged", verifier_env=dict(_STAGED_VERIFIER_ENV), **options)
        return task

    return stage


# One case per evaluator Dockerfile writer: both projection helpers, every caller. Cases
# without a skill Dockerfile meet the hostile values only through inherited image ENV.
_STAGING_PATHS = {
    "generated": _stage_generated(custom_dockerfile=False),
    "generated-base-image": _stage_generated(custom_dockerfile=False, base_image=_BASE_IMAGE),
    "generated-custom": _stage_generated(),
    "generated-custom-preserve": _stage_generated(base_image=_BASE_IMAGE, custom_dockerfile_mode="preserve"),
    "generated-custom-rebase": _stage_generated(base_image=_BASE_IMAGE, custom_dockerfile_mode="rebase"),
    "native-default": _stage_native(custom_dockerfile=False),
    "native-default-base-image": _stage_native(custom_dockerfile=False, base_image=_BASE_IMAGE),
    "native-custom": _stage_native(),
    "native-custom-rebase": _stage_native(base_image=_BASE_IMAGE),
    "native-custom-only": _stage_native(grading_mode="custom_only"),
}


@pytest.mark.parametrize("staging_path", list(_STAGING_PATHS))
def test_staged_dockerfile_blanks_image_judge_panel_env(tmp_path: Path, staging_path: str) -> None:
    hostile = _STAGING_PATHS[staging_path](tmp_path / "hostile", _HOSTILE_DOCKERFILE)
    control = _STAGING_PATHS[staging_path](tmp_path / "control", _CONTROL_DOCKERFILE)

    dockerfile = (hostile / "environment" / "Dockerfile").read_text(encoding="utf-8")
    lines = dockerfile.splitlines()
    assert lines.count(_JUDGE_PANEL_RESET) == 1
    assert lines[lines.index(_POLICY_RESET) + 1] == _JUDGE_PANEL_RESET
    # Even an image that already carries a skill-chosen panel ends with the names blank.
    image_env = _image_env(dockerfile, inherited=_HOSTILE_PANEL_ENV)
    assert _panel_names(image_env) == dict.fromkeys(sorted(JUDGE_PANEL_ENV_VARS), "")
    # The reset lives only in the Dockerfile: task.toml matches a skill without the hostile ENV.
    assert (hostile / "task.toml").read_bytes() == (control / "task.toml").read_bytes()
    assert "JUDGE_PANEL" not in (hostile / "task.toml").read_text(encoding="utf-8")


def test_skill_dockerfile_env_survives_staging_until_the_final_reset(tmp_path: Path) -> None:
    task = _STAGING_PATHS["generated-custom"](tmp_path, _HOSTILE_DOCKERFILE)

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    authored_only = dockerfile[: dockerfile.index(_JUDGE_PANEL_RESET)]

    assert _panel_names(_image_env(_HOSTILE_DOCKERFILE, inherited={})) == _HOSTILE_PANEL_ENV
    assert _panel_names(_image_env(authored_only, inherited={})) == _HOSTILE_PANEL_ENV
    assert _panel_names(_image_env(dockerfile, inherited={})) == dict.fromkeys(sorted(JUDGE_PANEL_ENV_VARS), "")


# --- The verifier reads blank judge-panel names as unset (D4) ---------------------------------------

_VERIFIER_ENVIRONMENT = (
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
    "AWS_REGION",
    *sorted(JUDGE_PANEL_ENV_VARS),
)
_PRIMARY_KEYS = {
    "nv_build": ("NVIDIA_API_KEY", "nvapi-HardeningPrimaryKey0001"),
    "openai": ("OPENAI_API_KEY", "sk-Hardening0002"),
}
_VERDICTS = {
    "accuracy": {
        "criteria": {
            "SKILL_IDENTIFIED": True,
            "ACTION_CORRECT": True,
            "FACTUALLY_ACCURATE": True,
            "TASK_ADDRESSED": False,
            "ACTIONABLE": True,
        },
        "score": 0.8,
        "reason": "mostly accurate",
    },
    "goal_accuracy": {"user_goal": "finish", "end_state": "finished", "achieved": True, "score": 1.0, "reason": "met"},
    "behavior_check": {"results": [{"step": 1, "passed": True, "reason": "reported"}], "score": 1.0, "summary": "ok"},
}


def _verdict_reply(request: urllib.request.Request) -> dict[str, Any]:
    prompt = json.loads(request.data)["messages"][0]["content"]
    if "SKILL_IDENTIFIED" in prompt:
        verdict = _VERDICTS["accuracy"]
    elif "EXPECTED BEHAVIORS" in prompt:
        verdict = _VERDICTS["behavior_check"]
    else:
        verdict = _VERDICTS["goal_accuracy"]
    return {"choices": [{"message": {"content": json.dumps(verdict)}}]}


def _run_verifier(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary: str,
    panel_env: Mapping[str, str],
) -> dict[str, Any]:
    """Run the standalone verifier's ``main()`` once and return everything it emitted."""
    module_name = f"harbor_eval_adapter_hardening_{root.parent.name}_{root.name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    verifier = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, verifier)
    spec.loader.exec_module(verifier)

    for directory in ("logs/agent", "logs/verifier", "tests"):
        (root / directory).mkdir(parents=True)
    verifier.LOGS_DIR = root / "logs"
    verifier.AGENT_LOGS_DIR = root / "logs" / "agent"
    verifier.VERIFIER_DIR = root / "logs" / "verifier"
    verifier.TESTS_DIR = root / "tests"
    verifier.ATIF_PATH = verifier.AGENT_LOGS_DIR / "trajectory.json"
    verifier.ENTRY_PATH = verifier.TESTS_DIR / "entry.json"
    verifier.REWARD_JSON = verifier.VERIFIER_DIR / "reward.json"
    verifier.REWARD_TXT = verifier.VERIFIER_DIR / "reward.txt"
    verifier.SKILL_EVALUATOR_REWARD_JSON = verifier.VERIFIER_DIR / "skill_evaluator_reward.json"
    verifier.ATIF_PATH.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "Finish the task."},
                    {"source": "agent", "message": "The task is finished."},
                ]
            }
        ),
        encoding="utf-8",
    )
    verifier.ENTRY_PATH.write_text(
        json.dumps(
            {
                "id": "blank-panel-case",
                "question": "Finish the task.",
                "ground_truth": "The task is finished.",
                "expected_behavior": ["Report that the task is finished"],
                "should_trigger": False,
                "evaluated_skill": "demo",
                "has_skill": True,
            }
        ),
        encoding="utf-8",
    )

    for name in _VERIFIER_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    key_name, key = _PRIMARY_KEYS[primary]
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", primary)
    monkeypatch.setenv(key_name, key)
    for name, value in panel_env.items():
        monkeypatch.setenv(name, value)

    requests: list[urllib.request.Request] = []
    ragas_calls: list[tuple[Any, ...]] = []

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> MockUrllibResponse:
        requests.append(request)
        return MockUrllibResponse(_verdict_reply(request))

    def ragas_goal_accuracy(*args: Any) -> dict[str, Any]:
        ragas_calls.append(args)
        return {"score": 0.9, "reason": "ragas verdict"}

    monkeypatch.setattr(verifier.urllib.request, "urlopen", urlopen)
    # An OpenAI primary takes the RAGAS goal judge only while no panel is configured.
    monkeypatch.setattr(verifier, "_judge_goal_accuracy_ragas", ragas_goal_accuracy)

    exit_code = verifier.main()

    return {
        "exit_code": exit_code,
        "requests": [(request.full_url, sorted(request.header_items()), request.data) for request in requests],
        "ragas_calls": ragas_calls,
        "reward_json": verifier.REWARD_JSON.read_bytes(),
        "reward_txt": verifier.REWARD_TXT.read_bytes(),
        "skill_evaluator_reward_json": verifier.SKILL_EVALUATOR_REWARD_JSON.read_bytes(),
    }


@pytest.mark.parametrize("primary", ["nv_build", "openai"])
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
def test_blank_judge_panel_names_behave_like_unset_ones_in_the_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    blank: str,
) -> None:
    unset = _run_verifier(tmp_path / "unset", monkeypatch, primary=primary, panel_env={})
    blanked = _run_verifier(
        tmp_path / "blank",
        monkeypatch,
        primary=primary,
        panel_env=dict.fromkeys(JUDGE_PANEL_ENV_VARS, blank),
    )

    assert blanked == unset
    assert len(unset["requests"]) == (2 if primary == "openai" else 3)
    assert len(unset["ragas_calls"]) == (1 if primary == "openai" else 0)
    assert b'"panel"' not in unset["skill_evaluator_reward_json"]


def test_staged_image_env_keeps_the_verifier_on_its_single_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _STAGING_PATHS["generated-custom"](tmp_path / "skill", _HOSTILE_DOCKERFILE)
    staged_dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    staged_panel_env = {
        name: value
        for name, value in _panel_names(_image_env(staged_dockerfile, inherited={})).items()
        if value is not None
    }

    unset = _run_verifier(tmp_path / "unset", monkeypatch, primary="nv_build", panel_env={})
    staged = _run_verifier(tmp_path / "staged", monkeypatch, primary="nv_build", panel_env=staged_panel_env)
    unreset = _run_verifier(tmp_path / "unreset", monkeypatch, primary="nv_build", panel_env=_HOSTILE_PANEL_ENV)

    assert staged == unset
    # Without the evaluator's reset, the skill image would have chosen its own judge.
    assert [json.loads(body)["model"] for _url, _headers, body in unreset["requests"]] == [_SKILL_JUDGE] * 3
    assert b'"panel"' in unreset["skill_evaluator_reward_json"]


# --- A judge panel keeps native verifiers in the evaluator-staged container (D2b) -------------------

_SEPARATE_VERIFIER_TASKS = [
    pytest.param('[verifier]\nenvironment_mode = "separate"\n\n[environment]\n', "[verifier]", id="task-mode"),
    pytest.param('[verifier]\nenvironment_mode = "SHARED"\n\n[environment]\n', "[verifier]", id="task-mode-not-shared"),
    pytest.param(
        '[verifier.environment]\ndocker_image = "attacker.example/verifier:latest"\n\n[environment]\n',
        "[verifier.environment]",
        id="task-environment",
    ),
    pytest.param(
        '[verifier]\nenvironment_mode = "separate"\n\n'
        '[verifier.environment]\ndocker_image = "attacker.example/verifier:latest"\n\n[environment]\n',
        "[verifier]",
        id="task-mode-and-environment",
    ),
    pytest.param(
        '[[steps]]\nname = "step-one"\n\n[steps.verifier]\nenvironment_mode = "separate"\n\n[environment]\n',
        "[steps[0].verifier]",
        id="step-mode",
    ),
    pytest.param(
        '[[steps]]\nname = "step-one"\n\n[[steps]]\nname = "step-two"\n\n'
        '[steps.verifier.environment]\ndocker_image = "attacker.example/verifier:latest"\n\n[environment]\n',
        "[steps[1].verifier.environment]",
        id="step-environment",
    ),
]


def _stage_native_task(output: Path, skill: Path, *, verifier_timeout_sec: float | None) -> Path:
    (task,) = stage_native_harbor_tasks(
        skill,
        output,
        grading_mode="default",
        verifier_env=dict(_STAGED_VERIFIER_ENV),
        verifier_timeout_sec=verifier_timeout_sec,
    )
    return task


@pytest.mark.parametrize(("task_body", "table"), _SEPARATE_VERIFIER_TASKS)
def test_panel_staging_rejects_a_separate_native_verifier_environment(
    tmp_path: Path,
    task_body: str,
    table: str,
) -> None:
    skill = _native_skill(tmp_path, task_body)

    with pytest.raises(ValueError, match="judge panel") as excinfo:
        _stage_native_task(tmp_path / "staged", skill, verifier_timeout_sec=_PANEL_VERIFIER_TIMEOUT)

    message = str(excinfo.value)
    assert f"Native Harbor task {table} " in message
    assert "member credential" in message
    assert "attacker.example" not in message
    assert not (tmp_path / "staged").exists()


@pytest.mark.parametrize(("task_body", "table"), _SEPARATE_VERIFIER_TASKS)
def test_separate_native_verifier_environment_still_stages_without_a_panel(
    tmp_path: Path,
    task_body: str,
    table: str,
) -> None:
    skill = _native_skill(tmp_path, task_body)

    task = _stage_native_task(tmp_path / "staged", skill, verifier_timeout_sec=None)

    authored = tomllib.loads(_NATIVE_TASK_HEAD + task_body)
    staged = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    for key in ("environment_mode", "environment"):
        assert staged["verifier"].get(key) == authored.get("verifier", {}).get(key)
    assert staged.get("steps") == authored.get("steps")


@pytest.mark.parametrize(
    "task_body",
    [
        pytest.param("[environment]\n", id="no-verifier-table"),
        pytest.param('[verifier]\nenvironment_mode = "shared"\n\n[environment]\n', id="explicit-shared"),
        pytest.param(
            '[[steps]]\nname = "step-one"\n\n[steps.verifier]\nenvironment_mode = "shared"\n\n[environment]\n',
            id="explicit-shared-step",
        ),
    ],
)
def test_panel_staging_accepts_a_shared_native_verifier(tmp_path: Path, task_body: str) -> None:
    skill = _native_skill(tmp_path, task_body)

    task = _stage_native_task(tmp_path / "staged", skill, verifier_timeout_sec=_PANEL_VERIFIER_TIMEOUT)

    staged = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert staged["verifier"]["timeout_sec"] == _PANEL_VERIFIER_TIMEOUT
    assert "environment" not in staged["verifier"]


_SEPARATE_VERIFIER_TABLE = (
    '[verifier]\nenvironment_mode = "separate"\n\n'
    '[verifier.environment]\ndocker_image = "attacker.example/verifier:latest"\n\n'
)


def test_panel_run_rejects_a_separate_native_verifier_before_harbor_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    _native_skill(tmp_path, _SEPARATE_VERIFIER_TABLE + "[environment]\n")

    result = harness.run(_panel_environment())

    assert harness.harbor_calls == []
    (error,) = result["error"]
    assert "Native Harbor task [verifier] " in error
    assert "judge panel" in error


def test_run_without_a_panel_keeps_a_separate_native_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    _native_skill(tmp_path, _SEPARATE_VERIFIER_TABLE + "[environment]\n")

    result = harness.run(_nv_build_primary())

    assert "error" not in result
    assert len(harness.harbor_calls) == 1
    (task_toml,) = harness.staged_task_tomls(result)
    assert tomllib.loads(task_toml)["verifier"]["environment"] == {"docker_image": "attacker.example/verifier:latest"}


# --- Deep native task configs (D20) ------------------------------------------------------------------


def _recursive_env_tables(value: object, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """The previous recursive walk, kept as the reference for order and paths."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key == "env" and isinstance(child, dict):
                yield child_path, child
            yield from _recursive_env_tables(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _recursive_env_tables(child, f"{path}[{index}]")


_ENV_TABLE_CONFIGS = [
    pytest.param(
        '[environment.env]\nA = "1"\n\n[verifier]\ntimeout_sec = 1.0\n\n[verifier.env]\nB = "2"\n\n'
        '[verifier.environment.env]\nC = "3"\n\n[[steps]]\nname = "one"\n\n[steps.verifier.env]\nD = "4"\n\n'
        '[[steps]]\nname = "two"\n\n[steps.agent]\nenv = {E = "5"}\n',
        id="task-steps-and-verifier",
    ),
    pytest.param(
        'env = {outer = "1", env = {inner = "2", env = {}}}\n'
        'matrix = [[{env = {F = "6"}}, "text"], [{nested = [{env = {G = "7"}}]}]]\n'
        'listed = {env = [{env = {H = "8"}}]}\n'
        'scalar = {env = "not a table"}\n',
        id="nested-lists-and-env-tables",
    ),
    pytest.param("", id="empty"),
]


@pytest.mark.parametrize("config", _ENV_TABLE_CONFIGS)
@pytest.mark.parametrize("prefix", ["", "task"])
def test_env_table_walk_keeps_the_recursive_order_and_paths(config: str, prefix: str) -> None:
    data = tomllib.loads(config)

    walked = [(path, id(table)) for path, table in _toml_env_tables(data, prefix)]

    assert walked == [(path, id(table)) for path, table in _recursive_env_tables(data, prefix)]


def test_env_table_walk_finds_env_tables_inside_lists_and_other_env_tables() -> None:
    data = tomllib.loads(_ENV_TABLE_CONFIGS[1].values[0])

    assert [path for path, _table in _toml_env_tables(data)] == [
        "env",
        "env.env",
        "env.env.env",
        "matrix[0][0].env",
        "matrix[1][0].nested[0].env",
        "listed.env[0].env",
    ]


def _deep_header(depth: int) -> str:
    return ".".join(f"k{index}" for index in range(depth))


def test_deep_dotted_native_task_header_does_not_exhaust_the_recursion_limit(tmp_path: Path) -> None:
    task = tmp_path / "deep-task"
    task.mkdir()
    (task / "task.toml").write_text(f"[{_deep_header(3000)}]\nx = 1\n", encoding="utf-8")

    assert _native_task_workdir(task) is None


def test_deep_dotted_env_table_naming_a_panel_control_is_still_rejected(tmp_path: Path) -> None:
    task = tmp_path / "deep-task"
    task.mkdir()
    header = f"{_deep_header(3000)}.env"
    (task / "task.toml").write_text(f'[{header}]\nSKILL_EVAL_JUDGE_PANEL = "openai:lenient-judge"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="judge panel") as excinfo:
        _native_task_workdir(task)

    assert f"[{header}]" in str(excinfo.value)
    assert "SKILL_EVAL_JUDGE_PANEL" in str(excinfo.value)


def test_deep_dotted_native_task_header_stages_with_and_without_a_panel(tmp_path: Path) -> None:
    skill = _native_skill(tmp_path, f"[environment]\n\n[{_deep_header(3000)}]\nx = 1\n")

    for label, verifier_timeout_sec in (("no-panel", None), ("panel", _PANEL_VERIFIER_TIMEOUT)):
        task = _stage_native_task(tmp_path / label, skill, verifier_timeout_sec=verifier_timeout_sec)
        assert tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))["k0"]["k1"]["k2"]
