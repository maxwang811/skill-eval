# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared cross-model judge panel helpers in ``eval_core.llm_judge``.

Covers panel settings parsing, model-family inference, verdict aggregation,
per-member credential routing through ``call_public_llm``, and the
``LLMClient(provider_config=...)`` hook that routing relies on. Aggregation
tests also run against an isolated execution of the verbatim shared block,
proving it works without the rest of the module (the Harbor verifier copy
cannot import skillevaluator).
"""

from __future__ import annotations

import builtins
import dis
import json
import math
import re
import statistics
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from skillevaluator.inference import client as client_module
from skillevaluator.inference.client import LLMClient
from skillevaluator.provider_config import OPENAI_BASE_URL, PUBLIC_NVIDIA_BUILD_BASE_URL, ProviderConfig
from skillevaluator.tier3.eval_core import llm_judge
from skillevaluator.tier3.eval_core.llm_judge import JudgePanelSettings, JudgeTarget

PANEL = "SKILL_EVAL_JUDGE_PANEL"
AGGREGATION = "SKILL_EVAL_JUDGE_PANEL_AGGREGATION"
QUORUM = "SKILL_EVAL_JUDGE_PANEL_QUORUM"
DISAGREEMENT = "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT"
ANTHROPIC_API_ROOT = "https://api.anthropic.com"

_BLOCK_BEGIN = "# --- BEGIN SHARED JUDGE PANEL HELPERS (verbatim copy in harbor/templates/eval.py) ---"
_BLOCK_END = "# --- END SHARED JUDGE PANEL HELPERS ---"
# The only non-builtin names the verbatim block may use (see the panel contract).
_BLOCK_IMPORTS: dict[str, Any] = {
    "math": math,
    "re": re,
    "statistics": statistics,
    "ContextVar": ContextVar,
    "NamedTuple": NamedTuple,
    "Any": Any,
    "Mapping": Mapping,
}
_CRITERIA_KEYS = ("SKILL_IDENTIFIED", "ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE")
_OPENAI = ("openai", "gpt-5.6-sol")
_ANTHROPIC = ("anthropic", "claude-opus-5")
_NVIDIA = ("nv_build", "nvidia/nemotron-3-super-120b-a12b")
_BEDROCK = ("bedrock", "us.anthropic.claude-opus-5")
_O_SERIES = ("openai", "o3")
_IDENTITIES = (_OPENAI, _ANTHROPIC, _NVIDIA, _BEDROCK, _O_SERIES)
_PROVIDER_ENV_VARS = (
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
    PANEL,
    AGGREGATION,
    QUORUM,
    DISAGREEMENT,
)


# ---------------------------------------------------------------------------
# The verbatim shared block, executed on its own
# ---------------------------------------------------------------------------


def _shared_block_code() -> CodeType:
    source = Path(llm_judge.__file__).read_text(encoding="utf-8")
    assert source.count(_BLOCK_BEGIN) == 1
    assert source.count(_BLOCK_END) == 1
    block = source[source.index(_BLOCK_BEGIN) : source.index(_BLOCK_END) + len(_BLOCK_END)]
    # Both host modules use postponed annotations; compile the block the same way.
    return compile(f"from __future__ import annotations\n{block}", "<shared judge panel block>", "exec")


def _isolated_block() -> SimpleNamespace:
    namespace: dict[str, Any] = {"__name__": "isolated_judge_panel_block", **_BLOCK_IMPORTS}
    exec(_shared_block_code(), namespace)
    return SimpleNamespace(**namespace)


def _global_loads(code: CodeType) -> set[str]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"}
    }
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names |= _global_loads(constant)
    return names


@pytest.fixture(params=["module", "isolated-block"])
def panel(request: pytest.FixtureRequest) -> Any:
    return llm_judge if request.param == "module" else _isolated_block()


def test_shared_block_references_only_allowed_globals() -> None:
    code = _shared_block_code()
    namespace: dict[str, Any] = {"__name__": "isolated_judge_panel_block", **_BLOCK_IMPORTS}
    exec(code, namespace)

    # Class bodies load __annotations__ from their own namespace.
    allowed = set(dir(builtins)) | set(namespace) | {"__annotations__"}
    assert _global_loads(code) <= allowed, sorted(_global_loads(code) - allowed)


def test_panel_constants_match_the_host_contract(panel: Any) -> None:
    assert panel.JUDGE_PANEL_ENV == PANEL
    assert panel.JUDGE_PANEL_AGGREGATION_ENV == AGGREGATION
    assert panel.JUDGE_PANEL_QUORUM_ENV == QUORUM
    assert panel.JUDGE_PANEL_DISAGREEMENT_ENV == DISAGREEMENT
    assert panel.JUDGE_PANEL_AGGREGATIONS == ("vote", "median", "mean")
    assert panel.JUDGE_PANEL_MAX_MEMBERS == 5
    assert panel.DEFAULT_JUDGE_PANEL_DISAGREEMENT == 0.4
    assert panel._ACTIVE_JUDGE_TARGET.get() is None
    assert panel.JudgeTarget._fields == ("provider", "model")
    assert panel.JudgePanelSettings._fields == ("members", "aggregation", "quorum", "disagreement_threshold")


# ---------------------------------------------------------------------------
# parse_judge_panel_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({}, id="unset"),
        pytest.param({PANEL: ""}, id="empty"),
        pytest.param({PANEL: " \t "}, id="whitespace"),
        # The verifier can receive empty placeholders for every allowlisted name.
        pytest.param({PANEL: "", AGGREGATION: "", QUORUM: " ", DISAGREEMENT: ""}, id="blank-knobs"),
    ],
)
def test_parse_without_panel_returns_none(panel: Any, environ: dict[str, str]) -> None:
    assert panel.parse_judge_panel_env(environ) is None


def test_parse_normalizes_entries_and_applies_defaults(panel: Any) -> None:
    settings = panel.parse_judge_panel_env(
        {PANEL: " OpenAI : gpt-5.6-sol ,anthropic:claude-opus-5, NV_BUILD:nvidia/nemotron-3-super-120b-a12b "}
    )

    assert settings == JudgePanelSettings(
        members=(
            JudgeTarget("openai", "gpt-5.6-sol"),
            JudgeTarget("anthropic", "claude-opus-5"),
            JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
        ),
        aggregation="vote",
        quorum=2,
        disagreement_threshold=0.4,
    )
    assert type(settings).__name__ == "JudgePanelSettings"
    assert type(settings.members[0]).__name__ == "JudgeTarget"


def test_parse_splits_on_the_first_colon_so_bedrock_ids_stay_intact(panel: Any) -> None:
    settings = panel.parse_judge_panel_env({PANEL: "bedrock:us.anthropic.claude-opus-5-v1:0"})

    assert settings.members == (JudgeTarget("bedrock", "us.anthropic.claude-opus-5-v1:0"),)
    assert settings.quorum == 1


def test_parse_accepts_printable_non_ascii_model_ids(panel: Any) -> None:
    settings = panel.parse_judge_panel_env({PANEL: "openai-compatible:équipe/modèle-1"})

    assert settings.members == (JudgeTarget("openai-compatible", "équipe/modèle-1"),)


@pytest.mark.parametrize(("count", "quorum"), [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)])
def test_parse_default_quorum_is_a_strict_majority(panel: Any, count: int, quorum: int) -> None:
    entries = ",".join(f"openai:model-{index}" for index in range(count))

    assert panel.parse_judge_panel_env({PANEL: entries}).quorum == quorum


def test_parse_reads_knobs_case_insensitively(panel: Any) -> None:
    settings = panel.parse_judge_panel_env(
        {
            PANEL: "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b",
            AGGREGATION: " MEDIAN ",
            QUORUM: " 3 ",
            DISAGREEMENT: " 0.25 ",
        }
    )

    assert (settings.aggregation, settings.quorum, settings.disagreement_threshold) == ("median", 3, 0.25)


@pytest.mark.parametrize(
    ("knobs", "expected"),
    [
        pytest.param({AGGREGATION: "mean"}, "mean", id="mean"),
        pytest.param({QUORUM: "1"}, 1, id="quorum-one"),
        pytest.param({QUORUM: "2"}, 2, id="quorum-n"),
        pytest.param({DISAGREEMENT: "0"}, 0.0, id="threshold-zero"),
        pytest.param({DISAGREEMENT: "1"}, 1.0, id="threshold-one"),
    ],
)
def test_parse_accepts_boundary_knobs(panel: Any, knobs: dict[str, str], expected: object) -> None:
    settings = panel.parse_judge_panel_env({PANEL: "openai:gpt-5.6-sol,anthropic:claude-opus-5", **knobs})

    name = next(iter(knobs))
    field = {AGGREGATION: "aggregation", QUORUM: "quorum", DISAGREEMENT: "disagreement_threshold"}[name]
    assert getattr(settings, field) == expected


_TWO_JUDGES = "openai:gpt-5.6-sol,anthropic:claude-opus-5"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        pytest.param({PANEL: "openai:gpt-5.6-sol,,anthropic:claude-opus-5"}, "empty entry", id="empty-entry"),
        pytest.param({PANEL: "openai:gpt-5.6-sol,"}, "empty entry", id="trailing-comma"),
        pytest.param({PANEL: "openai"}, "provider:model", id="missing-colon"),
        pytest.param({PANEL: ":gpt-5.6-sol"}, "provider:model", id="empty-provider"),
        pytest.param({PANEL: "openai:"}, "provider:model", id="empty-model"),
        pytest.param({PANEL: "openai:   "}, "provider:model", id="blank-model"),
        pytest.param({PANEL: "azure:gpt-5.6-sol"}, "unsupported provider 'azure'", id="unknown-provider"),
        pytest.param({PANEL: "openai:gpt 5.6"}, "whitespace or control characters", id="space-in-model"),
        pytest.param({PANEL: "openai:gpt-5.6\x07"}, "whitespace or control characters", id="control-in-model"),
        pytest.param({PANEL: "openai:gpt-5.6\u200b"}, "whitespace or control characters", id="format-char-in-model"),
        pytest.param({PANEL: "openai:gpt-5.6-sol, OPENAI : gpt-5.6-sol"}, "more than once", id="duplicate"),
        pytest.param(
            {PANEL: "openai-compatible:model-a,openai-compatible:model-b"},
            "at most one openai-compatible",
            id="two-gateways",
        ),
        pytest.param(
            {PANEL: "openai:a,openai:b,anthropic:c,nv_build:d,bedrock:e,openai:f"},
            "at most 5",
            id="six-members",
        ),
        pytest.param({PANEL: _TWO_JUDGES, QUORUM: "0"}, QUORUM, id="quorum-zero"),
        pytest.param({PANEL: _TWO_JUDGES, QUORUM: "3"}, QUORUM, id="quorum-above-n"),
        pytest.param({PANEL: _TWO_JUDGES, QUORUM: "two"}, QUORUM, id="quorum-word"),
        pytest.param({PANEL: _TWO_JUDGES, QUORUM: "1.5"}, QUORUM, id="quorum-fraction"),
        pytest.param({PANEL: _TWO_JUDGES, AGGREGATION: "majority"}, AGGREGATION, id="bad-aggregation"),
        pytest.param({PANEL: _TWO_JUDGES, DISAGREEMENT: "-0.1"}, DISAGREEMENT, id="threshold-negative"),
        pytest.param({PANEL: _TWO_JUDGES, DISAGREEMENT: "1.5"}, DISAGREEMENT, id="threshold-above-one"),
        pytest.param({PANEL: _TWO_JUDGES, DISAGREEMENT: "nan"}, DISAGREEMENT, id="threshold-nan"),
        pytest.param({PANEL: _TWO_JUDGES, DISAGREEMENT: "inf"}, DISAGREEMENT, id="threshold-inf"),
        pytest.param({PANEL: _TWO_JUDGES, DISAGREEMENT: "high"}, DISAGREEMENT, id="threshold-word"),
        pytest.param({AGGREGATION: "median"}, f"without {PANEL}", id="aggregation-without-panel"),
        pytest.param({PANEL: " ", QUORUM: "2"}, f"without {PANEL}", id="quorum-without-panel"),
        pytest.param({DISAGREEMENT: "0.3"}, f"without {PANEL}", id="threshold-without-panel"),
    ],
)
def test_parse_rejects_invalid_configuration(panel: Any, environ: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        panel.parse_judge_panel_env(environ)


# ---------------------------------------------------------------------------
# _model_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "model", "family"),
    [
        # Leaf prefixes.
        ("openai", "gpt-5.6-sol", "openai"),
        ("openai-compatible", "openai/openai/gpt-5.6-sol", "openai"),
        ("openai", "chatgpt-4o-latest", "openai"),
        ("openai", "codex-mini-latest", "openai"),
        ("openai", "davinci-002", "openai"),
        ("openai", "o3-mini", "openai"),
        ("openai", "o1", "openai"),
        ("openai-compatible", "openai/o4-mini", "openai"),
        ("nv_build", "o1", "openai"),
        ("anthropic", "claude-opus-5", "anthropic"),
        ("nv_build", "nvidia/nemotron-3-super-120b-a12b", "nvidia"),
        (None, "nvidia/nvidia/nemotron-3-super-120b-a12b", "nvidia"),
        ("openai-compatible", "nvidia/nvidia/nemotron-3-super-120b-long-ctx", "nvidia"),
        ("nv_build", "nvidia/nvidia-nemotron-nano-9b-v2", "nvidia"),
        ("nv_build", "nvidia/llama-3.1-nemotron-70b-instruct", "meta"),
        ("nv_build", "meta/llama-3.3-70b-instruct", "meta"),
        ("openai-compatible", "meta-llama-3.1-405b-instruct", "meta"),
        ("nv_build", "mistralai/mistral-large-2-instruct", "mistral"),
        ("nv_build", "mistralai/mixtral-8x22b-instruct-v0.1", "mistral"),
        ("nv_build", "mistralai/codestral-22b-instruct-v0.1", "mistral"),
        ("openai-compatible", "ministral-8b-latest", "mistral"),
        ("openai-compatible", "magistral-medium-latest", "mistral"),
        ("openai-compatible", "devstral-small-2505", "mistral"),
        ("openai-compatible", "pixtral-large-latest", "mistral"),
        ("openai-compatible", "gemini-2.5-pro", "google"),
        ("nv_build", "google/gemma-3-27b-it", "google"),
        ("nv_build", "qwen/qwen3-235b-a22b", "qwen"),
        ("nv_build", "qwen/qwq-32b", "qwen"),
        ("nv_build", "deepseek-ai/deepseek-r1", "deepseek"),
        ("nv_build", "microsoft/phi-4-mini-instruct", "microsoft"),
        ("nv_build", "phi3-medium-128k", "microsoft"),
        ("nv_build", "ibm/granite-3.3-8b-instruct", "ibm"),
        ("openai-compatible", "grok-4", "xai"),
        ("nv_build", "moonshotai/kimi-k2-instruct", "moonshot"),
        ("openai-compatible", "glm-4.5", "zhipu"),
        ("bedrock", "nova-pro-v1:0", "amazon"),
        ("openai-compatible", "titan-text-premier", "amazon"),
        ("openai-compatible", "command-a-03-2025", "cohere"),
        ("openai-compatible", "jamba-large-1.7", "ai21"),
        # Bedrock vendor ids, with and without a region prefix.
        ("bedrock", "us.anthropic.claude-opus-5", "anthropic"),
        ("bedrock", "us.anthropic.claude-opus-5-v1:0", "anthropic"),
        ("bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic"),
        ("bedrock", "global.anthropic.claude-sonnet-4-5", "anthropic"),
        ("bedrock", "us-gov.anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),
        ("bedrock", "jp.anthropic.claude-sonnet-4-5", "anthropic"),
        ("bedrock", "au.anthropic.claude-sonnet-4-5", "anthropic"),
        ("bedrock", "ca.meta.llama3-1-8b-instruct-v1:0", "meta"),
        ("bedrock", "ap.amazon.nova-lite-v1:0", "amazon"),
        ("bedrock", "apac.amazon.nova-micro-v1:0", "amazon"),
        ("bedrock", "eu.meta.llama3-2-3b-instruct-v1:0", "meta"),
        ("bedrock", "mistral.mistral-large-2407-v1:0", "mistral"),
        ("bedrock", "amazon.titan-text-express-v1", "amazon"),
        ("bedrock", "cohere.command-r-plus-v1:0", "cohere"),
        ("bedrock", "ai21.jamba-1-5-large-v1:0", "ai21"),
        ("bedrock", "us.deepseek.r1-v1:0", "deepseek"),
        ("bedrock", "qwen.qwen3-32b-v1:0", "qwen"),
        ("bedrock", "openai.gpt-oss-120b-1:0", "openai"),
        ("bedrock", "google.gemma-3-12b-it", "google"),
        ("bedrock", "nvidia.nemotron-nano-12b-v2", "nvidia"),
        # Bedrock ids of unlisted vendors fall back to the model name after the vendor.
        ("bedrock", "moonshot.kimi-k2-thinking", "moonshot"),
        ("bedrock", "us.moonshot.kimi-k2-thinking", "moonshot"),
        ("bedrock", "apac.moonshot.kimi-k2-thinking", "moonshot"),
        ("bedrock", "moonshotai.kimi-k2.5", "moonshot"),
        ("bedrock", "US.Moonshot.Kimi-K2-Thinking", "moonshot"),
        ("bedrock", "zai.glm-4.6", "zhipu"),
        ("bedrock", "eu.zai.glm-4.6", "zhipu"),
        ("bedrock", "global.zai.glm-4.6", "zhipu"),
        ("bedrock", "us-gov.zai.glm-4.6", "zhipu"),
        ("openai-compatible", "bedrock/moonshot.kimi-k2-thinking", "moonshot"),
        # An unlisted vendor with an unknown model name stays unknown.
        ("bedrock", "minimax.minimax-m2", "unknown"),
        ("bedrock", "us.minimax.minimax-m2", "unknown"),
        ("bedrock", "stability.sd3-5-large-v1:0", "unknown"),
        ("bedrock", "us.twelvelabs.pegasus-1-2-v1:0", "unknown"),
        # Gateway ids that wrap a Bedrock model.
        ("openai-compatible", "aws/anthropic/bedrock-claude-opus-5", "anthropic"),
        # Organization segments decide when the leaf has no known prefix.
        ("nv_build", "nvidia/some-new-model", "nvidia"),
        ("nv_build", "deepseek-ai/janus-pro-7b", "deepseek"),
        ("openai-compatible", "aws/anthropic/custom-alias", "anthropic"),
        ("nv_build", "thudm/chatglm3-6b", "zhipu"),
        ("nv_build", "ibm-granite/custom-model", "ibm"),
        ("nv_build", "microsoft/orca-2-13b", "microsoft"),
        ("nv_build", "meta-llama/custom-model", "meta"),
        ("nv_build", "mistralai/custom-model", "mistral"),
        ("nv_build", "moonshotai/custom-model", "moonshot"),
        ("nv_build", "zhipuai/custom-model", "zhipu"),
        ("nv_build", "xai/custom-model", "xai"),
        # A dotted version is not a vendor prefix that hides a known model name.
        ("nv_build", "nvidia/some-new-model-v1.5", "nvidia"),
        ("nv_build", "upstage/solar-10.7b-instruct", "unknown"),
        # A prefix must not run into another letter.
        ("nv_build", "philosopher-7b", "unknown"),
        ("nv_build", "gptx-1", "unknown"),
        ("nv_build", "commander-7b", "unknown"),
        ("nv_build", "novak-1", "unknown"),
        ("nv_build", "omni-7b", "unknown"),
        ("bedrock", "us.custom-model", "unknown"),
        # The provider is only a fallback, and only for single-family providers.
        ("openai", "philosopher-7b", "openai"),
        ("openai", "my-finetune", "openai"),
        ("anthropic", "my-finetune", "anthropic"),
        (" Anthropic ", "", "anthropic"),
        ("openai", None, "openai"),
        ("nv_build", "my-finetune", "unknown"),
        ("bedrock", "my-finetune", "unknown"),
        ("openai-compatible", "my-finetune", "unknown"),
        (None, "my-finetune", "unknown"),
        (None, None, "unknown"),
        # Model ids are compared case-insensitively after trimming.
        ("nv_build", "  Meta/Llama-3.1-8B-Instruct  ", "meta"),
        ("bedrock", "US.Anthropic.Claude-Opus-5", "anthropic"),
        ("nv_build", "nvidia//nemotron-mini/", "nvidia"),
    ],
)
def test_model_family(panel: Any, provider: str | None, model: str | None, family: str) -> None:
    assert panel._model_family(provider, model) == family


# ---------------------------------------------------------------------------
# aggregate_panel
# ---------------------------------------------------------------------------


def _accuracy(score: float, *flags: bool, reason: str = "accuracy verdict") -> dict[str, Any]:
    return {"score": score, "reason": reason, "criteria": dict(zip(_CRITERIA_KEYS, flags, strict=True))}


def _goal(achieved: bool, score: float, reason: str = "goal verdict") -> dict[str, Any]:
    return {
        "score": score,
        "reason": reason,
        "user_goal": "finish the task",
        "end_state": "task state",
        "method": "custom",
        "achieved": achieved,
    }


def _behavior(*passed: bool) -> dict[str, Any]:
    return {
        "score": round(sum(passed) / len(passed), 4),
        "reason": "behavior summary",
        "results": [
            {"step": index + 1, "passed": value, "reason": f"note {index}"} for index, value in enumerate(passed)
        ],
    }


def _failed(reason: str = "LLM judge error: timeout") -> dict[str, Any]:
    return {"score": None, "status": "error", "reason": reason}


def _members(*results: Any) -> list[tuple[str, str, Any]]:
    return [(provider, model, result) for (provider, model), result in zip(_IDENTITIES, results, strict=False)]


def test_accuracy_vote_uses_per_criterion_majority(panel: Any) -> None:
    first = _accuracy(0.8, True, True, True, True, False, reason="openai reason")
    second = _accuracy(1.0, True, True, True, True, True, reason="anthropic reason")
    third = _accuracy(0.4, True, False, True, False, False, reason="nvidia reason")

    result = panel.aggregate_panel("accuracy", _members(first, second, third))

    assert result == {
        "score": 0.8,
        "reason": "panel vote (3/3 judges)",
        "criteria": {
            "SKILL_IDENTIFIED": True,
            "ACTION_CORRECT": True,
            "FACTUALLY_ACCURATE": True,
            "TASK_ADDRESSED": True,
            "ACTIONABLE": False,
        },
        "panel": {
            "aggregation": "vote",
            "quorum": 2,
            "members": [
                {
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "family": "openai",
                    "status": "ok",
                    "score": 0.8,
                    "reason": "openai reason",
                    "criteria": first["criteria"],
                },
                {
                    "provider": "anthropic",
                    "model": "claude-opus-5",
                    "family": "anthropic",
                    "status": "ok",
                    "score": 1.0,
                    "reason": "anthropic reason",
                    "criteria": second["criteria"],
                },
                {
                    "provider": "nv_build",
                    "model": "nvidia/nemotron-3-super-120b-a12b",
                    "family": "nvidia",
                    "status": "ok",
                    "score": 0.4,
                    "reason": "nvidia reason",
                    "criteria": third["criteria"],
                },
            ],
            "spread": 0.6,
            "agreement": 0.8,
            "disagreement": True,
            "disagreement_threshold": 0.4,
            "failed_members": 0,
        },
    }
    json.dumps(result, allow_nan=False)


def test_accuracy_vote_tie_counts_half_and_leaves_criterion_undecided(panel: Any) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members(_accuracy(1.0, True, True, True, True, True), _accuracy(0.4, True, True, False, False, False)),
    )

    assert result["criteria"] == {
        "SKILL_IDENTIFIED": True,
        "ACTION_CORRECT": True,
        "FACTUALLY_ACCURATE": None,
        "TASK_ADDRESSED": None,
        "ACTIONABLE": None,
    }
    assert result["score"] == 0.7
    assert result["panel"]["agreement"] == 0.7
    assert result["panel"]["quorum"] == 2


@pytest.mark.parametrize(("aggregation", "score"), [("median", 0.8), ("mean", 0.7333)])
def test_accuracy_median_and_mean_use_member_scores(panel: Any, aggregation: str, score: float) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members(
            _accuracy(0.8, True, True, True, True, False),
            _accuracy(1.0, True, True, True, True, True),
            _accuracy(0.4, True, False, True, False, False),
        ),
        aggregation=aggregation,
    )

    assert result["score"] == score
    assert result["reason"] == f"panel {aggregation} (3/3 judges)"
    # The per-criterion vote is still reported for explainability.
    assert result["criteria"]["ACTIONABLE"] is False
    assert result["panel"]["aggregation"] == aggregation


@pytest.mark.parametrize(
    "criteria",
    [
        pytest.param({}, id="empty"),
        pytest.param(dict.fromkeys(_CRITERIA_KEYS[:4], True), id="missing-key"),
        pytest.param({**dict.fromkeys(_CRITERIA_KEYS, True), "ACTIONABLE": "true"}, id="non-bool"),
        pytest.param(None, id="absent"),
    ],
)
def test_accuracy_without_complete_criteria_fails_vote_but_not_median(panel: Any, criteria: object) -> None:
    score_only = {"score": 0.6, "reason": "score only"}
    if criteria is not None:
        score_only["criteria"] = criteria
    others = (_accuracy(1.0, True, True, True, True, True), _accuracy(0.4, True, False, True, False, False))

    voted = panel.aggregate_panel("accuracy", _members(score_only, *others))
    median = panel.aggregate_panel("accuracy", _members(score_only, *others), aggregation="median")

    assert voted["panel"]["members"][0] == {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "family": "openai",
        "status": "error",
        "reason": "Judge returned incomplete accuracy criteria",
    }
    assert voted["score"] == 0.7
    assert voted["reason"] == "panel vote (2/3 judges; 1 failed)"
    assert voted["panel"]["failed_members"] == 1

    assert median["panel"]["members"][0]["status"] == "ok"
    assert median["panel"]["members"][0]["criteria"] == {}
    assert median["score"] == 0.6
    assert median["criteria"] == {
        "SKILL_IDENTIFIED": True,
        "ACTION_CORRECT": None,
        "FACTUALLY_ACCURATE": True,
        "TASK_ADDRESSED": None,
        "ACTIONABLE": None,
    }
    assert median["panel"]["agreement"] == 0.7
    assert median["panel"]["failed_members"] == 0


def test_accuracy_median_reports_empty_criteria_when_no_member_supplied_them(panel: Any) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members({"score": 0.6, "reason": "a"}, {"score": 0.2, "reason": "b", "criteria": {}}),
        aggregation="mean",
    )

    assert result["score"] == 0.4
    assert result["criteria"] == {}
    assert result["panel"]["agreement"] is None


def test_behavior_vote_is_per_position(panel: Any) -> None:
    result = panel.aggregate_panel(
        "behavior_check",
        _members(_behavior(True, True, False), _behavior(True, False, False), _behavior(True, True, True)),
        expected_count=3,
    )

    assert result["score"] == 0.6667
    assert result["reason"] == "panel vote (3/3 judges)"
    assert result["results"] == [
        {"step": 1, "passed": True, "reason": "3/3 judges observed this behavior"},
        {"step": 2, "passed": True, "reason": "2/3 judges observed this behavior"},
        {"step": 3, "passed": False, "reason": "1/3 judges observed this behavior"},
    ]
    assert result["panel"]["agreement"] == 0.7778
    assert result["panel"]["spread"] == 0.6667
    assert result["panel"]["members"][0] == {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "family": "openai",
        "status": "ok",
        "score": 0.6667,
        "reason": "behavior summary",
        "results": [
            {"step": 1, "passed": True, "reason": "note 0"},
            {"step": 2, "passed": True, "reason": "note 1"},
            {"step": 3, "passed": False, "reason": "note 2"},
        ],
    }


def test_behavior_vote_tie_counts_half(panel: Any) -> None:
    result = panel.aggregate_panel(
        "behavior_check",
        _members(_behavior(True, True), _behavior(True, False)),
        expected_count=2,
    )

    assert result["score"] == 0.75
    assert result["results"][1] == {"step": 2, "passed": None, "reason": "1/2 judges observed this behavior"}
    assert result["panel"]["agreement"] == 0.75


@pytest.mark.parametrize("aggregation", ["vote", "median", "mean"])
def test_behavior_count_mismatch_is_an_invalid_member_in_every_mode(panel: Any, aggregation: str) -> None:
    result = panel.aggregate_panel(
        "behavior_check",
        _members(_behavior(True, True, False), _behavior(True, False, False), _behavior(True, True)),
        aggregation=aggregation,
        expected_count=3,
    )

    assert result["panel"]["members"][2] == {
        "provider": "nv_build",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "family": "nvidia",
        "status": "error",
        "reason": "behavior result count 2 does not match expected 3",
    }
    assert result["panel"]["failed_members"] == 1
    assert result["reason"] == f"panel {aggregation} (2/3 judges; 1 failed)"
    assert len(result["results"]) == 3


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({"score": 1.0, "reason": "x"}, id="no-results"),
        pytest.param({"score": 1.0, "reason": "x", "results": "all passed"}, id="not-a-list"),
        pytest.param({"score": 1.0, "reason": "x", "results": [{"passed": "yes"}]}, id="non-bool-passed"),
        pytest.param({"score": 1.0, "reason": "x", "results": ["passed"]}, id="non-dict-entry"),
    ],
)
@pytest.mark.parametrize("aggregation", ["vote", "median"])
def test_behavior_malformed_results_are_invalid(panel: Any, malformed: dict[str, Any], aggregation: str) -> None:
    result = panel.aggregate_panel(
        "behavior_check",
        _members(malformed, _behavior(True), _behavior(True)),
        aggregation=aggregation,
        expected_count=1,
    )

    assert result["panel"]["members"][0]["status"] == "error"
    assert result["panel"]["members"][0]["reason"] == "Judge returned malformed behavior results"
    assert result["score"] == 1.0


def test_behavior_without_expected_count_uses_the_most_common_length(panel: Any) -> None:
    result = panel.aggregate_panel(
        "behavior_check",
        _members(_behavior(True, False), _behavior(True), _behavior(False, False)),
    )

    assert result["panel"]["members"][1]["reason"] == "behavior result count 1 does not match expected 2"
    assert result["results"] == [
        {"step": 1, "passed": None, "reason": "1/2 judges observed this behavior"},
        {"step": 2, "passed": False, "reason": "0/2 judges observed this behavior"},
    ]
    assert result["score"] == 0.25


def test_goal_vote_takes_the_median_score_of_the_majority_side(panel: Any) -> None:
    result = panel.aggregate_panel(
        "goal_accuracy",
        _members(_goal(True, 1.0), _goal(True, 0.8), _goal(False, 0.0)),
    )

    assert result["score"] == 0.9
    assert result["achieved"] is True
    assert result["method"] == "custom"
    assert result["reason"] == "panel vote (3/3 judges)"
    assert result["panel"]["agreement"] == 0.6667
    assert result["panel"]["spread"] == 1.0
    assert result["panel"]["disagreement"] is True
    assert result["panel"]["members"][2] == {
        "provider": "nv_build",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "family": "nvidia",
        "status": "ok",
        "score": 0.0,
        "reason": "goal verdict",
        "achieved": False,
        "method": "custom",
    }


def test_goal_vote_majority_not_achieved(panel: Any) -> None:
    result = panel.aggregate_panel(
        "goal_accuracy",
        _members(_goal(False, 0.2), _goal(False, 0.0), _goal(True, 1.0)),
    )

    assert result["achieved"] is False
    assert result["score"] == 0.1


def test_goal_vote_tie_is_undecided_at_half(panel: Any) -> None:
    result = panel.aggregate_panel("goal_accuracy", _members(_goal(True, 1.0), _goal(False, 0.0)))

    assert result["achieved"] is None
    assert result["score"] == 0.5
    assert result["panel"]["agreement"] == 0.5


def test_goal_without_boolean_achieved_fails_vote_but_not_median(panel: Any) -> None:
    unscored = {"score": 1.0, "reason": "no verdict", "method": "custom"}
    members = _members(unscored, _goal(True, 0.8), _goal(False, 0.0))

    voted = panel.aggregate_panel("goal_accuracy", members)
    median = panel.aggregate_panel("goal_accuracy", members, aggregation="median")

    assert voted["panel"]["members"][0]["status"] == "error"
    assert voted["panel"]["members"][0]["reason"] == "Judge returned no boolean achieved verdict"
    assert voted["achieved"] is None
    assert voted["score"] == 0.5

    assert median["panel"]["members"][0]["status"] == "ok"
    assert median["panel"]["members"][0]["achieved"] is None
    assert median["score"] == 0.8
    assert median["achieved"] is None
    assert median["method"] == "custom"


@pytest.mark.parametrize(
    "score",
    [None, math.nan, math.inf, -math.inf, -0.1, 1.5, True, "0.5"],
    ids=["none", "nan", "inf", "-inf", "negative", "above-one", "bool", "string"],
)
@pytest.mark.parametrize("aggregation", ["vote", "median"])
def test_non_finite_or_out_of_range_scores_are_invalid(panel: Any, score: object, aggregation: str) -> None:
    bad = _accuracy(0.0, True, True, True, True, True)
    bad["score"] = score

    result = panel.aggregate_panel(
        "accuracy",
        _members(bad, _accuracy(1.0, True, True, True, True, True), _accuracy(1.0, True, True, True, True, True)),
        aggregation=aggregation,
    )

    assert result["panel"]["members"][0]["status"] == "error"
    assert result["panel"]["members"][0]["reason"] == "Judge returned no finite score between 0 and 1"
    assert result["score"] == 1.0
    json.dumps(result, allow_nan=False)


def test_integer_scores_are_accepted(panel: Any) -> None:
    result = panel.aggregate_panel(
        "goal_accuracy",
        _members(_goal(True, 1), _goal(False, 0)),
        aggregation="mean",
    )

    assert [member["score"] for member in result["panel"]["members"]] == [1.0, 0.0]
    assert result["score"] == 0.5


@pytest.mark.parametrize(
    "result_value",
    [pytest.param(None, id="none"), pytest.param("0.9", id="string"), pytest.param([0.9], id="list")],
)
def test_non_dict_member_results_are_invalid(panel: Any, result_value: object) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members(result_value, _accuracy(1.0, True, True, True, True, True)),
        quorum=1,
    )

    assert result["panel"]["members"][0]["reason"] == "Judge returned an invalid result"
    assert result["score"] == 1.0


def test_quorum_exactly_met_aggregates_the_successful_members(panel: Any) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members(
            _accuracy(0.8, True, True, True, True, False),
            _accuracy(1.0, True, True, True, True, True),
            _failed(),
        ),
    )

    assert result["score"] == 0.9
    assert result["reason"] == "panel vote (2/3 judges; 1 failed)"
    assert result["criteria"]["ACTIONABLE"] is None
    assert result["panel"]["quorum"] == 2
    assert result["panel"]["failed_members"] == 1
    assert result["panel"]["members"][2] == {
        "provider": "nv_build",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "family": "nvidia",
        "status": "error",
        "reason": "LLM judge error: timeout",
    }
    assert "status" not in result


@pytest.mark.parametrize(
    ("metric", "extra"),
    [
        ("accuracy", {}),
        ("behavior_check", {"results": []}),
        ("goal_accuracy", {"method": "custom"}),
    ],
)
def test_one_short_of_quorum_fails_closed_with_the_full_panel(
    panel: Any,
    metric: str,
    extra: dict[str, Any],
) -> None:
    ok_result = {
        "accuracy": _accuracy(0.8, True, True, True, True, False),
        "behavior_check": _behavior(True, False),
        "goal_accuracy": _goal(True, 1.0),
    }[metric]

    result = panel.aggregate_panel(
        metric,
        _members(ok_result, ok_result, _failed("Required judge raised TimeoutError")),
        quorum=3,
        expected_count=2 if metric == "behavior_check" else None,
    )

    panel_block = result.pop("panel")
    assert result == {
        "score": None,
        "status": "error",
        "reason": f"Judge panel quorum not met for {metric}: 2/3 judges succeeded (quorum 3)",
        **extra,
    }
    assert [member["status"] for member in panel_block["members"]] == ["ok", "ok", "error"]
    assert panel_block["quorum"] == 3
    assert panel_block["failed_members"] == 1
    assert panel_block["spread"] == 0.0
    assert panel_block["disagreement"] is False
    assert set(panel_block) == {
        "aggregation",
        "quorum",
        "members",
        "spread",
        "agreement",
        "disagreement",
        "disagreement_threshold",
        "failed_members",
    }


def test_all_members_failed(panel: Any) -> None:
    result = panel.aggregate_panel("goal_accuracy", _members(_failed("a"), _failed("b"), _failed("c")))

    assert result["score"] is None
    assert result["status"] == "error"
    assert result["reason"] == "Judge panel quorum not met for goal_accuracy: 0/3 judges succeeded (quorum 2)"
    assert result["method"] == "custom"
    assert result["panel"]["spread"] is None
    assert result["panel"]["agreement"] is None
    assert result["panel"]["disagreement"] is False
    assert result["panel"]["failed_members"] == 3
    assert [member["reason"] for member in result["panel"]["members"]] == ["a", "b", "c"]
    json.dumps(result, allow_nan=False)


def test_explicit_zero_quorum_still_requires_one_successful_member(panel: Any) -> None:
    result = panel.aggregate_panel("accuracy", _members(_failed(), _failed()), quorum=0)

    assert result["score"] is None
    assert result["status"] == "error"


@pytest.mark.parametrize(("count", "quorum"), [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)])
def test_default_quorum_is_a_strict_majority(panel: Any, count: int, quorum: int) -> None:
    members = _members(*[_accuracy(1.0, True, True, True, True, True)] * count)

    assert panel.aggregate_panel("accuracy", members)["panel"]["quorum"] == quorum


def test_default_quorum_rejects_half_of_an_even_panel(panel: Any) -> None:
    ok = _accuracy(1.0, True, True, True, True, True)

    result = panel.aggregate_panel("accuracy", _members(ok, ok, _failed(), _failed()))

    assert result["status"] == "error"
    assert result["reason"].endswith("2/4 judges succeeded (quorum 3)")


def test_explicit_quorum_of_one_accepts_a_single_success(panel: Any) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        _members(_failed(), _accuracy(0.6, True, True, True, False, False), _failed()),
        quorum=1,
    )

    assert result["score"] == 0.6
    assert result["reason"] == "panel vote (1/3 judges; 2 failed)"
    assert result["panel"]["spread"] == 0.0
    assert result["panel"]["agreement"] == 1.0


@pytest.mark.parametrize(
    ("scores", "threshold", "flagged"),
    [
        ((0.8, 0.4), 0.4, True),
        ((0.6, 0.2), 0.4, True),
        ((0.8, 0.4), 0.41, False),
        ((0.8, 0.5), 0.4, False),
        ((0.5, 0.5), 0.0, True),
        ((1.0, 0.0), 1.0, True),
    ],
)
def test_disagreement_flag_includes_a_spread_equal_to_the_threshold(
    panel: Any,
    scores: tuple[float, float],
    threshold: float,
    flagged: bool,
) -> None:
    result = panel.aggregate_panel(
        "goal_accuracy",
        _members(*(_goal(True, score) for score in scores)),
        aggregation="median",
        disagreement_threshold=threshold,
    )

    assert result["panel"]["spread"] == round(scores[0] - scores[1], 4)
    assert result["panel"]["disagreement"] is flagged
    assert result["panel"]["disagreement_threshold"] == threshold


def test_scores_are_rounded_to_four_decimals(panel: Any) -> None:
    result = panel.aggregate_panel(
        "goal_accuracy",
        _members(_goal(True, 0.123456), _goal(True, 0.0), _goal(True, 1.0)),
        aggregation="mean",
    )

    assert [member["score"] for member in result["panel"]["members"]] == [0.1235, 0.0, 1.0]
    assert result["score"] == 0.3745
    assert result["panel"]["spread"] == 1.0


def test_member_reasons_are_bounded_text(panel: Any) -> None:
    long_reason = "x" * 2000
    noisy = _behavior(True)
    noisy["reason"] = long_reason
    noisy["results"][0]["reason"] = {"nested": "object"}

    result = panel.aggregate_panel(
        "behavior_check",
        _members(noisy, _failed(long_reason), {**_behavior(True), "reason": 42}),
        expected_count=1,
    )

    first, second, third = result["panel"]["members"]
    assert first["reason"] == "x" * 509 + "..."
    assert first["results"][0]["reason"] == ""
    assert second["reason"] == "x" * 509 + "..."
    assert third["reason"] == ""


@pytest.mark.parametrize("aggregation", ["vote", "median", "mean"])
@pytest.mark.parametrize(
    ("metric", "results", "expected_count"),
    [
        (
            "accuracy",
            (
                _accuracy(0.8, True, True, True, True, False),
                {"score": 0.6, "reason": "score only", "criteria": {}},
                _failed(),
                _accuracy(0.2, False, False, True, False, False),
            ),
            None,
        ),
        (
            "behavior_check",
            (_behavior(True, False), _behavior(True), _behavior(False, False), _failed()),
            2,
        ),
        (
            "goal_accuracy",
            (_goal(True, 1.0), {"score": 0.5, "reason": "x"}, _goal(False, 0.25), _failed()),
            None,
        ),
    ],
)
def test_reaggregating_member_entries_reproduces_the_result(
    panel: Any,
    metric: str,
    results: tuple[Any, ...],
    expected_count: int | None,
    aggregation: str,
) -> None:
    first = panel.aggregate_panel(
        metric,
        _members(*results),
        aggregation=aggregation,
        quorum=1,
        expected_count=expected_count,
    )

    again = panel.aggregate_panel(
        metric,
        [(member["provider"], member["model"], member) for member in first["panel"]["members"]],
        aggregation=aggregation,
        quorum=first["panel"]["quorum"],
    )

    assert again == first


def test_member_families_come_from_the_model_id(panel: Any) -> None:
    result = panel.aggregate_panel(
        "accuracy",
        [
            ("bedrock", "us.anthropic.claude-opus-5", _accuracy(1.0, True, True, True, True, True)),
            ("nv_build", "nvidia/llama-3.1-nemotron-70b-instruct", _failed()),
            ("openai-compatible", "my-finetune", _accuracy(1.0, True, True, True, True, True)),
        ],
    )

    assert [member["family"] for member in result["panel"]["members"]] == ["anthropic", "meta", "unknown"]


@pytest.mark.parametrize(
    ("metric", "aggregation"),
    [("accuracy", "majority"), ("accuracy", "VOTE"), ("security", "vote")],
)
def test_aggregate_rejects_unknown_metric_or_aggregation(panel: Any, metric: str, aggregation: str) -> None:
    with pytest.raises(ValueError, match="Unsupported judge panel"):
        panel.aggregate_panel(metric, _members(_failed()), aggregation=aggregation)


# ---------------------------------------------------------------------------
# call_public_llm routing
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the LLMClient that call_public_llm imports with a recorder returning canned text."""
    record = SimpleNamespace(
        constructed=[],
        completions=[],
        respond=lambda _kwargs: "canned judge text",
        error=None,
    )

    class _RecordingLLMClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            record.constructed.append((args, kwargs))
            self.kwargs = kwargs

        def completions(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            response_schema: dict[str, Any] | None = None,
            schema_name: str = "judge_response",
        ) -> str:
            record.completions.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response_schema": response_schema,
                    "schema_name": schema_name,
                }
            )
            if record.error is not None:
                raise record.error
            return record.respond(self.kwargs)

    monkeypatch.setattr(client_module, "LLMClient", _RecordingLLMClient)
    return record


def _routed_config(record: SimpleNamespace) -> ProviderConfig:
    [(args, kwargs)] = record.constructed
    assert args == ()
    assert set(kwargs) == {"max_tokens", "temperature", "provider_config"}
    config = kwargs["provider_config"]
    assert isinstance(config, ProviderConfig)
    return config


@pytest.mark.parametrize(
    ("target", "environ", "expected"),
    [
        pytest.param(
            JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
            {"NVIDIA_API_KEY": "nvapi-member-key-0001"},
            ("nv_build", "nvidia/nemotron-3-super-120b-a12b", "nvapi-member-key-0001", PUBLIC_NVIDIA_BUILD_BASE_URL),
            id="nv_build",
        ),
        pytest.param(
            JudgeTarget("openai", "gpt-5.6-sol"),
            {"OPENAI_API_KEY": "sk-openai-member-0001"},
            ("openai", "gpt-5.6-sol", "sk-openai-member-0001", OPENAI_BASE_URL),
            id="openai-official",
        ),
        pytest.param(
            JudgeTarget("openai", "gpt-5.6-sol"),
            {"OPENAI_API_KEY": "sk-openai-member-0001", "OPENAI_BASE_URL": "https://openai-proxy.example/v1/"},
            ("openai", "gpt-5.6-sol", "sk-openai-member-0001", "https://openai-proxy.example/v1"),
            id="openai-own-base-url",
        ),
        pytest.param(
            JudgeTarget("openai-compatible", "nvidia/nvidia/nemotron-3-super-120b-long-ctx"),
            {
                "SKILL_EVAL_LLM_API_KEY": "gateway-member-key-0001",
                "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/v1/",
            },
            (
                "openai-compatible",
                "nvidia/nvidia/nemotron-3-super-120b-long-ctx",
                "gateway-member-key-0001",
                "https://gateway.example/v1",
            ),
            id="openai-compatible",
        ),
        pytest.param(
            JudgeTarget("anthropic", "claude-opus-5"),
            {"ANTHROPIC_API_KEY": "sk-ant-member-key-0001"},
            ("anthropic", "claude-opus-5", "sk-ant-member-key-0001", ANTHROPIC_API_ROOT),
            id="anthropic-official",
        ),
        pytest.param(
            JudgeTarget("anthropic", "claude-opus-5"),
            {
                "ANTHROPIC_API_KEY": "sk-ant-member-key-0001",
                "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1/",
            },
            ("anthropic", "claude-opus-5", "sk-ant-member-key-0001", "https://anthropic-proxy.example"),
            id="anthropic-own-base-url",
        ),
        pytest.param(
            JudgeTarget("bedrock", "us.anthropic.claude-opus-5"),
            {"AWS_REGION": "us-east-1"},
            ("bedrock", "us.anthropic.claude-opus-5", None, None),
            id="bedrock",
        ),
    ],
)
def test_target_routes_through_the_member_provider_config(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
    target: JudgeTarget,
    environ: dict[str, str],
    expected: tuple[str, str, str | None, str | None],
) -> None:
    for name, value in environ.items():
        clean_provider_env.setenv(name, value)

    content, error = llm_judge.call_public_llm("Judge this response", target=target)

    assert (content, error) == ("canned judge text", None)
    config = _routed_config(recording_client)
    assert (config.provider, config.model, config.api_key, config.base_url) == expected
    assert recording_client.constructed[0][1]["max_tokens"] == 1024
    assert recording_client.constructed[0][1]["temperature"] == 0.0


@pytest.mark.parametrize(("region", "expected"), [("eu-central-1", "eu-central-1"), (None, "us-west-2")])
def test_bedrock_member_uses_the_aws_region(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
    region: str | None,
    expected: str,
) -> None:
    if region:
        clean_provider_env.setenv("AWS_REGION", region)

    llm_judge.call_public_llm("Judge this response", target=JudgeTarget("bedrock", "us.anthropic.claude-opus-5"))

    config = _routed_config(recording_client)
    assert config.region == expected
    assert config.litellm_model == "bedrock/us.anthropic.claude-opus-5"


@pytest.mark.parametrize(
    ("target", "key_env", "official_base_url"),
    [
        (JudgeTarget("openai", "gpt-5.6-sol"), "OPENAI_API_KEY", OPENAI_BASE_URL),
        (JudgeTarget("anthropic", "claude-opus-5"), "ANTHROPIC_API_KEY", ANTHROPIC_API_ROOT),
        (JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"), "NVIDIA_API_KEY", PUBLIC_NVIDIA_BUILD_BASE_URL),
    ],
)
def test_gateway_base_url_never_reaches_a_native_member(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
    target: JudgeTarget,
    key_env: str,
    official_base_url: str,
) -> None:
    # Even when the native provider is also the primary provider, the gateway
    # endpoint and key belong to the openai-compatible member alone.
    clean_provider_env.setenv("SKILL_EVAL_LLM_PROVIDER", target.provider)
    clean_provider_env.setenv("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example/v1")
    clean_provider_env.setenv("SKILL_EVAL_LLM_API_KEY", "gateway-key-not-for-members")
    clean_provider_env.setenv(key_env, "member-native-key-0001")

    content, error = llm_judge.call_public_llm("Judge this response", target=target)

    assert error is None
    assert content == "canned judge text"
    config = _routed_config(recording_client)
    assert config.base_url == official_base_url
    assert config.api_key == "member-native-key-0001"


@pytest.mark.parametrize(
    ("target", "variable", "decoys"),
    [
        (
            JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
            "NVIDIA_API_KEY",
            {"OPENAI_API_KEY": "sk-decoy-openai-0001", "SKILL_EVAL_LLM_API_KEY": "decoy-gateway-0001"},
        ),
        (
            JudgeTarget("openai", "gpt-5.6-sol"),
            "OPENAI_API_KEY",
            {"SKILL_EVAL_LLM_API_KEY": "decoy-gateway-0001", "NVIDIA_API_KEY": "nvapi-decoy-0001"},
        ),
        (
            JudgeTarget("openai-compatible", "gateway-model"),
            "SKILL_EVAL_LLM_API_KEY",
            {"OPENAI_API_KEY": "sk-decoy-openai-0001", "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/v1"},
        ),
        (
            JudgeTarget("anthropic", "claude-opus-5"),
            "ANTHROPIC_API_KEY",
            {"OPENAI_API_KEY": "sk-decoy-openai-0001", "SKILL_EVAL_LLM_API_KEY": "decoy-gateway-0001"},
        ),
    ],
)
@pytest.mark.parametrize("blank", [False, True], ids=["unset", "blank"])
def test_missing_member_key_is_an_error_without_a_fallback(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
    target: JudgeTarget,
    variable: str,
    decoys: dict[str, str],
    blank: bool,
) -> None:
    for name, value in decoys.items():
        clean_provider_env.setenv(name, value)
    if blank:
        clean_provider_env.setenv(variable, "   ")

    content, error = llm_judge.call_public_llm("Judge this response", target=target)

    assert content is None
    assert error == f"No API key configured for {target.provider} judge panel member ({variable})"
    assert recording_client.constructed == []


@pytest.mark.parametrize(
    ("target", "environ", "message"),
    [
        (
            JudgeTarget("openai-compatible", "gateway-model"),
            {"SKILL_EVAL_LLM_API_KEY": "gateway-member-key-0001"},
            "No base URL configured for openai-compatible judge panel member (SKILL_EVAL_LLM_BASE_URL)",
        ),
        (
            JudgeTarget("openai-compatible", "gateway-model"),
            {"SKILL_EVAL_LLM_API_KEY": "gateway-member-key-0001", "SKILL_EVAL_LLM_BASE_URL": "ftp://gateway.example"},
            "SKILL_EVAL_LLM_BASE_URL must be an absolute HTTP or HTTPS URL",
        ),
        (
            JudgeTarget("openai", "gpt-5.6-sol"),
            {"OPENAI_API_KEY": "sk-openai-member-0001", "OPENAI_BASE_URL": "file:///etc/passwd"},
            "OPENAI_BASE_URL must be an absolute HTTP or HTTPS URL",
        ),
        (
            JudgeTarget("anthropic", "claude-opus-5"),
            {"ANTHROPIC_API_KEY": "sk-ant-member-key-0001", "ANTHROPIC_BASE_URL": "https://user:pw@proxy.example"},
            "ANTHROPIC_BASE_URL must be an absolute HTTP or HTTPS URL",
        ),
        (
            JudgeTarget("azure", "gpt-5.6-sol"),
            {"OPENAI_API_KEY": "sk-openai-member-0001"},
            "Unsupported judge panel provider: azure",
        ),
    ],
)
def test_unusable_member_endpoint_is_an_error(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
    target: JudgeTarget,
    environ: dict[str, str],
    message: str,
) -> None:
    for name, value in environ.items():
        clean_provider_env.setenv(name, value)

    content, error = llm_judge.call_public_llm("Judge this response", target=target)

    assert content is None
    assert error is not None
    assert message in error
    assert "pw@" not in error
    assert recording_client.constructed == []


def test_active_target_contextvar_routes_without_an_explicit_target(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
) -> None:
    clean_provider_env.setenv("ANTHROPIC_API_KEY", "sk-ant-member-key-0001")
    clean_provider_env.setenv("NVIDIA_API_KEY", "nvapi-member-key-0001")

    token = llm_judge._ACTIVE_JUDGE_TARGET.set(JudgeTarget("anthropic", "claude-opus-5"))
    try:
        assert llm_judge.call_public_llm("Judge this response") == ("canned judge text", None)
        # An explicit target wins over the active one.
        llm_judge.call_public_llm("Judge this response", target=JudgeTarget("nv_build", "nvidia/nemotron-mini"))
    finally:
        llm_judge._ACTIVE_JUDGE_TARGET.reset(token)

    providers = [kwargs["provider_config"].provider for _args, kwargs in recording_client.constructed]
    assert providers == ["anthropic", "nv_build"]

    llm_judge.call_public_llm("Judge this response")
    assert "provider_config" not in recording_client.constructed[-1][1]


def test_target_ignores_model_overrides_fallbacks_and_explicit_credentials(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
) -> None:
    clean_provider_env.setenv("OPENAI_API_KEY", "sk-openai-member-0001")
    clean_provider_env.setenv("LLM_JUDGE_MODEL", "judge-model-override")
    clean_provider_env.setenv("SKILL_EVAL_JUDGE_MODEL", "judge-model-override")
    clean_provider_env.setenv("SKILL_EVAL_LLM_MODEL", "llm-model-override")
    clean_provider_env.setenv("LLM_JUDGE_FALLBACK_MODELS", "fallback-a,fallback-b")

    content, error = llm_judge.call_public_llm(
        "Judge this response",
        model="explicit-model",
        api_key="explicit-key-0001",
        max_tokens=4096,
        temperature=0.5,
        response_schema=llm_judge.ACCURACY_JSON_SCHEMA,
        schema_name="accuracy_judgment",
        target=JudgeTarget("openai", "gpt-5.6-sol"),
    )

    assert (content, error) == ("canned judge text", None)
    config = _routed_config(recording_client)
    assert (config.model, config.api_key) == ("gpt-5.6-sol", "sk-openai-member-0001")
    assert recording_client.constructed[0][1]["max_tokens"] == 4096
    assert recording_client.constructed[0][1]["temperature"] == 0.5
    assert recording_client.completions == [
        {
            "system_prompt": "You are a precise evaluation judge.",
            "user_prompt": "Judge this response",
            "response_schema": llm_judge.ACCURACY_JSON_SCHEMA,
            "schema_name": "accuracy_judgment",
        }
    ]


_GATEWAY_PRIMARY = {
    "SKILL_EVAL_LLM_PROVIDER": "openai-compatible",
    "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/v1",
    "SKILL_EVAL_LLM_API_KEY": "gateway-member-key-0001",
}


@pytest.mark.parametrize(
    ("target", "member_env", "sdk", "expected_sdk_kwargs"),
    [
        pytest.param(
            JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
            {"NVIDIA_API_KEY": "nvapi-member-key-0001"},
            "openai.OpenAI",
            {"api_key": "nvapi-member-key-0001", "base_url": PUBLIC_NVIDIA_BUILD_BASE_URL, "max_retries": 0},
            id="nv_build",
        ),
        pytest.param(
            JudgeTarget("openai", "gpt-5.6-sol"),
            {"OPENAI_API_KEY": "sk-openai-member-0001"},
            "openai.OpenAI",
            {"api_key": "sk-openai-member-0001", "base_url": OPENAI_BASE_URL, "max_retries": 0},
            id="openai",
        ),
        pytest.param(
            JudgeTarget("openai-compatible", "nvidia/nvidia/nemotron-3-super-120b-long-ctx"),
            {},
            "openai.OpenAI",
            {"api_key": "gateway-member-key-0001", "base_url": "https://gateway.example/v1", "max_retries": 0},
            id="openai-compatible",
        ),
        pytest.param(
            JudgeTarget("anthropic", "claude-opus-5"),
            {"ANTHROPIC_API_KEY": "sk-ant-member-key-0001"},
            "anthropic.Anthropic",
            {"api_key": "sk-ant-member-key-0001", "base_url": ANTHROPIC_API_ROOT, "max_retries": 0},
            id="anthropic",
        ),
    ],
)
def test_target_reaches_the_sdk_with_only_the_member_endpoint(
    clean_provider_env: pytest.MonkeyPatch,
    target: JudgeTarget,
    member_env: dict[str, str],
    sdk: str,
    expected_sdk_kwargs: dict[str, Any],
) -> None:
    for name, value in {**_GATEWAY_PRIMARY, **member_env}.items():
        clean_provider_env.setenv(name, value)
    sdk_client = MagicMock()
    sdk_client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Done"
    sdk_client.messages.create.return_value = MagicMock(content=[text_block])

    with patch(sdk, return_value=sdk_client) as sdk_cls:
        assert llm_judge.call_public_llm("Judge this response", max_tokens=4096, target=target) == ("Done", None)

    sdk_cls.assert_called_once_with(**expected_sdk_kwargs)
    create = sdk_client.messages.create if sdk == "anthropic.Anthropic" else sdk_client.chat.completions.create
    assert create.call_args.kwargs["model"] == target.model
    if target.provider == "openai":
        # The native OpenAI endpoint keeps its gpt-5 request shape under a panel target.
        assert create.call_args.kwargs["max_completion_tokens"] == 4096
        assert "temperature" not in create.call_args.kwargs


def test_target_provider_errors_are_redacted(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
) -> None:
    clean_provider_env.setenv("NVIDIA_API_KEY", "nvapi-member-secret-0001")
    recording_client.error = RuntimeError("401 for key nvapi-member-secret-0001")

    content, error = llm_judge.call_public_llm(
        "Judge this response",
        target=JudgeTarget("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
    )

    assert content is None
    assert error == "Public provider call failed: 401 for key [REDACTED]"


def test_no_target_constructs_llm_client_exactly_as_before(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
) -> None:
    clean_provider_env.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    clean_provider_env.setenv("NVIDIA_API_KEY", "nvapi-primary-key-0001")

    assert llm_judge.call_public_llm("Judge this response") == ("canned judge text", None)
    llm_judge.call_public_llm(
        "Judge again",
        model="custom-judge",
        api_key="explicit-key-0001",
        max_tokens=4096,
        temperature=0.25,
        timeout=5,
        allow_model_fallback=False,
        response_schema=llm_judge.BEHAVIOR_CHECK_JSON_SCHEMA,
        schema_name="behavior_check_judgment",
    )

    assert recording_client.constructed == [
        ((), {"model": None, "api_key": None, "max_tokens": 1024, "temperature": 0.0}),
        ((), {"model": "custom-judge", "api_key": "explicit-key-0001", "max_tokens": 4096, "temperature": 0.25}),
    ]
    assert recording_client.completions == [
        {
            "system_prompt": "You are a precise evaluation judge.",
            "user_prompt": "Judge this response",
            "response_schema": None,
            "schema_name": "judge_response",
        },
        {
            "system_prompt": "You are a precise evaluation judge.",
            "user_prompt": "Judge again",
            "response_schema": llm_judge.BEHAVIOR_CHECK_JSON_SCHEMA,
            "schema_name": "behavior_check_judgment",
        },
    ]


# ---------------------------------------------------------------------------
# LLMClient with an explicit provider_config
# ---------------------------------------------------------------------------


def _forbid_environment_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: Any, **_kwargs: Any) -> ProviderConfig:
        raise AssertionError("resolve_llm_provider must not run when provider_config is given")

    monkeypatch.setattr(client_module, "resolve_llm_provider", _fail)


def test_llm_client_uses_an_explicit_provider_config_verbatim(clean_provider_env: pytest.MonkeyPatch) -> None:
    _forbid_environment_resolution(clean_provider_env)
    # Ambient primary-provider settings must not leak into an explicit config.
    clean_provider_env.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    clean_provider_env.setenv("OPENAI_API_KEY", "sk-ambient-key-0001")
    clean_provider_env.setenv("SKILL_EVAL_LLM_MODEL", "ambient-model")
    config = ProviderConfig(
        provider="nv_build",
        model="nvidia/nemotron-3-super-120b-a12b",
        api_key="nvapi-member-key-0001",
        base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
        litellm_model="openai/nvidia/nemotron-3-super-120b-a12b",
        credential_env="NVIDIA_API_KEY",
    )
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content="Done"))]

    client = LLMClient(max_tokens=64, temperature=0.0, provider_config=config)

    assert client.model == config.model
    assert client.base_url == config.base_url
    assert client.api_key == config.api_key
    assert client._resolved_config() is config
    with patch("openai.OpenAI", return_value=mock_openai) as openai_cls:
        assert client.completions("system", "user") == "Done"
    openai_cls.assert_called_once_with(
        api_key="nvapi-member-key-0001", base_url=PUBLIC_NVIDIA_BUILD_BASE_URL, max_retries=0
    )
    assert mock_openai.chat.completions.create.call_args.kwargs["model"] == config.model


def test_llm_client_provider_config_routes_anthropic_to_its_root(clean_provider_env: pytest.MonkeyPatch) -> None:
    _forbid_environment_resolution(clean_provider_env)
    config = ProviderConfig(
        provider="anthropic",
        model="claude-opus-5",
        api_key="sk-ant-member-key-0001",
        base_url=ANTHROPIC_API_ROOT,
        litellm_model="anthropic/claude-opus-5",
        credential_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
    )
    block = MagicMock()
    block.type = "text"
    block.text = "Done"
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = MagicMock(content=[block])

    with patch("anthropic.Anthropic", return_value=mock_anthropic) as anthropic_cls:
        assert LLMClient(provider_config=config).completions("system", "user") == "Done"

    anthropic_cls.assert_called_once_with(api_key="sk-ant-member-key-0001", max_retries=0, base_url=ANTHROPIC_API_ROOT)
    assert mock_anthropic.messages.create.call_args.kwargs["model"] == "claude-opus-5"


@pytest.mark.parametrize(
    "conflict",
    [{"model": "other-model"}, {"base_url": "https://other.example/v1"}, {"api_key": "other-key-0001"}],
    ids=["model", "base_url", "api_key"],
)
def test_llm_client_rejects_provider_config_with_separate_endpoint_arguments(conflict: dict[str, str]) -> None:
    config = ProviderConfig(
        provider="openai",
        model="gpt-5.6-sol",
        api_key="sk-openai-member-0001",
        base_url=OPENAI_BASE_URL,
        litellm_model="openai/gpt-5.6-sol",
    )

    with pytest.raises(ValueError, match="provider_config"):
        LLMClient(provider_config=config, **conflict)


# ---------------------------------------------------------------------------
# Shared judges under a panel target
# ---------------------------------------------------------------------------

_GOAL_RESPONSE = json.dumps(
    {"user_goal": "finish", "end_state": "unfinished", "achieved": False, "score": 0.0, "reason": "not done"}
)


def test_shared_goal_judge_reports_achieved_only_for_panel_members(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_call(prompt: str, **kwargs: Any) -> tuple[str, None]:
        calls.append(kwargs)
        return _GOAL_RESPONSE, None

    monkeypatch.setattr(llm_judge, "call_public_llm", fake_call)

    single = llm_judge.judge_goal_accuracy("question", "ground truth", "agent response")
    token = llm_judge._ACTIVE_JUDGE_TARGET.set(JudgeTarget("openai", "gpt-5.6-sol"))
    try:
        active = llm_judge.judge_goal_accuracy("question", "ground truth", "agent response")
    finally:
        llm_judge._ACTIVE_JUDGE_TARGET.reset(token)
    explicit = llm_judge.judge_goal_accuracy(
        "question",
        "ground truth",
        "agent response",
        target=JudgeTarget("anthropic", "claude-opus-5"),
    )

    assert single == {"score": 0.0, "reason": "not done", "user_goal": "finish", "end_state": "unfinished"}
    assert active == {**single, "achieved": False}
    assert explicit == {**single, "achieved": False}
    assert calls[-1]["target"] == JudgeTarget("anthropic", "claude-opus-5")


def test_shared_goal_judge_skip_result_is_unchanged_under_a_target() -> None:
    token = llm_judge._ACTIVE_JUDGE_TARGET.set(JudgeTarget("openai", "gpt-5.6-sol"))
    try:
        result = llm_judge.judge_goal_accuracy("question", "", "agent response")
    finally:
        llm_judge._ACTIVE_JUDGE_TARGET.reset(token)

    assert result == {"score": 1.0, "reason": "No ground_truth -- skipped"}


def test_shared_judges_route_each_member_and_aggregate(
    clean_provider_env: pytest.MonkeyPatch,
    recording_client: SimpleNamespace,
) -> None:
    clean_provider_env.setenv("OPENAI_API_KEY", "sk-openai-member-0001")
    clean_provider_env.setenv("ANTHROPIC_API_KEY", "sk-ant-member-key-0001")
    clean_provider_env.setenv("NVIDIA_API_KEY", "nvapi-member-key-0001")
    verdicts = {
        "openai": {"criteria": dict.fromkeys(_CRITERIA_KEYS, True), "score": 1.0, "reason": "openai"},
        "anthropic": {
            "criteria": {**dict.fromkeys(_CRITERIA_KEYS, True), "ACTIONABLE": False},
            "score": 0.8,
            "reason": "anthropic",
        },
        "nv_build": {"criteria": dict.fromkeys(_CRITERIA_KEYS, False), "score": 0.0, "reason": "nvidia"},
    }
    goals = {
        "openai": {"user_goal": "g", "end_state": "e", "achieved": True, "score": 1.0, "reason": "openai"},
        "anthropic": {"user_goal": "g", "end_state": "e", "achieved": False, "score": 0.0, "reason": "anthropic"},
        "nv_build": {"user_goal": "g", "end_state": "e", "achieved": True, "score": 0.9, "reason": "nvidia"},
    }
    settings = llm_judge.parse_judge_panel_env(
        {PANEL: "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b"}
    )

    def run(judge: Any, payloads: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
        recording_client.respond = lambda kwargs: json.dumps(payloads[kwargs["provider_config"].provider])
        member_results = []
        for target in settings.members:
            token = llm_judge._ACTIVE_JUDGE_TARGET.set(target)
            try:
                member_results.append((target.provider, target.model, judge("question", "ground truth", "agent")))
            finally:
                llm_judge._ACTIVE_JUDGE_TARGET.reset(token)
        return llm_judge.aggregate_panel(
            metric,
            member_results,
            aggregation=settings.aggregation,
            quorum=settings.quorum,
            disagreement_threshold=settings.disagreement_threshold,
        )

    accuracy = run(llm_judge.judge_accuracy, verdicts, "accuracy")
    goal = run(llm_judge.judge_goal_accuracy, goals, "goal_accuracy")

    routed = [kwargs["provider_config"] for _args, kwargs in recording_client.constructed]
    assert [(config.provider, config.model) for config in routed] == [*settings.members, *settings.members]
    assert accuracy["score"] == 0.8
    assert accuracy["criteria"]["ACTIONABLE"] is False
    assert accuracy["panel"]["spread"] == 1.0
    assert [member["reason"] for member in accuracy["panel"]["members"]] == ["openai", "anthropic", "nvidia"]
    assert goal["achieved"] is True
    assert goal["score"] == 0.95
    assert goal["method"] == "custom"
