# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential routing for cross-model judge panel members in the Harbor verifier.

Every panel member must reach only its own provider endpoint with only its own
credential. These tests capture the real ``urllib.request.Request`` objects the
standalone verifier builds, so URLs, headers, and bodies are asserted exactly
as they would leave the container. A golden test pins the single-judge request
sequence that must not change when no panel is configured.
"""

from __future__ import annotations

import email.message
import importlib.util
import io
import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import pytest
from tests.conftest import MockUrllibResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_TEMPLATE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GATEWAY_BASE_URL = "https://gateway.example"
GATEWAY_URL = "https://gateway.example/chat/completions"

# Distinct fake credentials, one per provider variable, so any cross-member leak is detectable.
MEMBER_KEYS = {
    "nv_build": ("NVIDIA_API_KEY", "nvapi-PanelMemberNvidiaKey0001"),
    "openai": ("OPENAI_API_KEY", "sk-PanelMemberOpenaiKey0002"),
    "anthropic": ("ANTHROPIC_API_KEY", "sk-ant-PanelMemberAnthropicKey0003"),
    "openai-compatible": ("SKILL_EVAL_LLM_API_KEY", "gw-PanelMemberGatewayKey0004"),
}
MEMBERS = (
    ("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
    ("openai", "gpt-5.6-sol"),
    ("anthropic", "claude-opus-5"),
    ("openai-compatible", "gateway/judge-model"),
)
PANEL = ",".join(f"{provider}:{model}" for provider, model in MEMBERS)
METRICS = ("accuracy", "goal_accuracy", "behavior_check")
EXPECTED_BEHAVIOR = ["Report that the task is complete", "Stay on the requested task"]

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
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
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

_ACCURACY_VERDICT = {
    "criteria": {
        "SKILL_IDENTIFIED": True,
        "ACTION_CORRECT": True,
        "FACTUALLY_ACCURATE": True,
        "TASK_ADDRESSED": True,
        "ACTIONABLE": False,
    },
    "score": 0.8,
    "reason": "mostly accurate",
}
_GOAL_VERDICT = {
    "user_goal": "complete the task",
    "end_state": "the task is complete",
    "achieved": True,
    "score": 1.0,
    "reason": "goal met",
}
_BEHAVIOR_VERDICT = {
    "results": [
        {"step": 1, "passed": True, "reason": "reported completion"},
        {"step": 2, "passed": True, "reason": "stayed on task"},
    ],
    "score": 1.0,
    "summary": "both behaviors observed",
}
_VERDICTS = {"accuracy": _ACCURACY_VERDICT, "goal_accuracy": _GOAL_VERDICT, "behavior_check": _BEHAVIOR_VERDICT}


@pytest.fixture(autouse=True)
def _hermetic_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep routing assertions independent of credentials in the invoking shell."""
    for name in _ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _load_verifier(tmp_path: Path) -> ModuleType:
    """Load the standalone verifier template against a temporary Harbor workspace."""
    module_name = f"harbor_eval_panel_routing_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, _EVAL_TEMPLATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

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
    module.ENTRY_PATH.write_text(
        json.dumps(
            {
                "id": "panel-routing-case",
                "question": "Complete the task.",
                "ground_truth": "The task is complete.",
                "expected_behavior": EXPECTED_BEHAVIOR,
                "should_trigger": False,
                "evaluated_skill": "demo",
                "has_skill": True,
            }
        ),
        encoding="utf-8",
    )
    return module


def _judge_kind(request: urllib.request.Request) -> str | None:
    """Identify which LLM metric built a request from its prompt text."""
    prompt = json.loads(request.data)["messages"][0]["content"]
    if "SKILL_IDENTIFIED" in prompt:
        return "accuracy"
    if "EXPECTED BEHAVIORS" in prompt:
        return "behavior_check"
    if "Did the agent achieve the expected goal?" in prompt:
        return "goal_accuracy"
    return None


def _provider_reply(request: urllib.request.Request, text: str) -> dict[str, Any]:
    """Shape a completion the way the request's provider would return it."""
    if urlsplit(request.full_url).path.endswith("/v1/messages"):
        return {"content": [{"type": "text", "text": text}]}
    return {"choices": [{"message": {"content": text}}]}


def _default_reply(request: urllib.request.Request) -> dict[str, Any]:
    kind = _judge_kind(request)
    return _provider_reply(request, json.dumps(_VERDICTS[kind]) if kind else "plain judge reply")


class _Transport:
    """Record each real Request the verifier opens and answer like its provider."""

    def __init__(self, reply: Callable[[urllib.request.Request], Any] = _default_reply) -> None:
        self.requests: list[urllib.request.Request] = []
        self._reply = reply

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> MockUrllibResponse:
        assert isinstance(request, urllib.request.Request)
        assert timeout is not None and timeout > 0
        self.requests.append(request)
        return MockUrllibResponse(self._reply(request))


def _http_error(url: str, code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "Error", hdrs=email.message.Message(), fp=io.BytesIO(body.encode()))


def _set_member_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable, key in MEMBER_KEYS.values():
        monkeypatch.setenv(variable, key)


def _wire(request: urllib.request.Request) -> str:
    """Everything a request puts on the wire: URL, every header, and the body."""
    return "\n".join(
        [request.full_url, *(f"{name}: {value}" for name, value in request.header_items()), request.data.decode()]
    )


def _assert_carries_only_its_own_key(request: urllib.request.Request, provider: str) -> None:
    own_key = MEMBER_KEYS[provider][1]
    if provider == "anthropic":
        assert request.get_header("X-api-key") == own_key
        assert request.get_header("Authorization") is None
    else:
        assert request.get_header("Authorization") == f"Bearer {own_key}"
        assert request.get_header("X-api-key") is None
    assert own_key not in request.full_url
    assert own_key not in request.data.decode()
    wire = _wire(request)
    for other_provider, (_variable, other_key) in MEMBER_KEYS.items():
        if other_provider != provider:
            assert other_key not in wire, f"{provider} member request carried the {other_provider} key"


def _keys_named(value: Any, name: str) -> list[Any]:
    """Return every value stored under ``name`` anywhere in a JSON document."""
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


# ---------------------------------------------------------------------------
# Panel members reach only their own endpoint with only their own key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("native_base_urls", "expected_urls"),
    [
        pytest.param(
            {},
            {
                "nv_build": NVIDIA_URL,
                "openai": OPENAI_URL,
                "anthropic": ANTHROPIC_URL,
                "openai-compatible": GATEWAY_URL,
            },
            id="official-endpoints",
        ),
        pytest.param(
            {
                "OPENAI_BASE_URL": "https://openai-proxy.example/v1/",
                "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1",
            },
            {
                "nv_build": NVIDIA_URL,
                "openai": "https://openai-proxy.example/v1/chat/completions",
                "anthropic": "https://anthropic-proxy.example/v1/messages",
                "openai-compatible": GATEWAY_URL,
            },
            id="own-base-url-variables",
        ),
    ],
)
def test_each_panel_member_reaches_only_its_own_endpoint_with_only_its_own_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_base_urls: dict[str, str],
    expected_urls: dict[str, str],
) -> None:
    """Route every member of a four-provider panel to its own URL with its own credential only."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    # The primary is nv_build with a gateway base URL: only the openai-compatible member may use it.
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    for variable, value in native_base_urls.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("SKILL_EVAL_JUDGE_PANEL", PANEL)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    verifier.main()

    url_providers = {url: provider for provider, url in expected_urls.items()}
    assert [
        (_judge_kind(request), request.full_url, json.loads(request.data)["model"]) for request in transport.requests
    ] == [(metric, expected_urls[provider], model) for metric in METRICS for provider, model in MEMBERS]
    for request in transport.requests:
        _assert_carries_only_its_own_key(request, url_providers[request.full_url])

    rich_text = verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8")
    rich = json.loads(rich_text)
    for metric in METRICS:
        members = rich["details"][metric]["panel"]["members"]
        assert [(member["provider"], member["model"], member["status"]) for member in members] == [
            (provider, model, "ok") for provider, model in MEMBERS
        ]
    artifacts = rich_text + verifier.REWARD_JSON.read_text(encoding="utf-8")
    assert all(key not in artifacts for _variable, key in MEMBER_KEYS.values())
    assert verifier._ACTIVE_JUDGE_TARGET.get() is None


@pytest.mark.parametrize(
    ("provider", "model", "expected_url"),
    [
        ("nv_build", "nvidia/nemotron-3-super-120b-a12b", NVIDIA_URL),
        ("openai", "gpt-5.6-sol", OPENAI_URL),
        ("anthropic", "claude-opus-5", ANTHROPIC_URL),
    ],
)
def test_primary_gateway_base_url_never_redirects_a_native_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    expected_url: str,
) -> None:
    """Keep a native member on its own endpoint even when it is the primary with a gateway base URL."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget(provider, model)
    )

    assert (content, error) == ("plain judge reply", None)
    assert provenance == {"provider": provider, "model": model}
    assert [request.full_url for request in transport.requests] == [expected_url]
    _assert_carries_only_its_own_key(transport.requests[0], provider)


def test_gateway_member_uses_the_gateway_base_url_and_gateway_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send the openai-compatible member to SKILL_EVAL_LLM_BASE_URL with SKILL_EVAL_LLM_API_KEY."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", "https://gateway.example/v1/")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-proxy.example/v1")
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    _content, error, _provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget("openai-compatible", "gateway/judge-model")
    )

    assert error is None
    assert [request.full_url for request in transport.requests] == ["https://gateway.example/v1/chat/completions"]
    _assert_carries_only_its_own_key(transport.requests[0], "openai-compatible")


@pytest.mark.parametrize("placeholder", [None, "", "   "], ids=["unset", "empty", "blank"])
@pytest.mark.parametrize(("provider", "model"), MEMBERS)
def test_member_without_its_own_key_fails_without_borrowing_another_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    placeholder: str | None,
) -> None:
    """Fail a member whose own credential is missing, even when every other provider key is present."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    variable = MEMBER_KEYS[provider][0]
    if placeholder is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, placeholder)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget(provider, model)
    )

    assert content is None
    assert error == f"No API key configured for {provider} judge panel member ({variable})"
    assert provenance == {"provider": provider, "model": model}
    assert transport.requests == []


@pytest.mark.parametrize(
    ("provider", "model", "environment", "expected_error"),
    [
        pytest.param(
            "openai",
            "gpt-5.6-sol",
            {"OPENAI_BASE_URL": "file:///etc/passwd"},
            "absolute HTTP or HTTPS URL",
            id="openai-file-url",
        ),
        pytest.param(
            "openai-compatible",
            "gateway/judge-model",
            {},
            "No base URL configured for openai-compatible judge panel member (SKILL_EVAL_LLM_BASE_URL)",
            id="gateway-without-base-url",
        ),
        pytest.param(
            "openai-compatible",
            "gateway/judge-model",
            {"SKILL_EVAL_LLM_BASE_URL": "ftp://gateway.example"},
            "absolute HTTP or HTTPS URL",
            id="gateway-ftp-url",
        ),
        pytest.param(
            "anthropic",
            "claude-opus-5",
            {"ANTHROPIC_BASE_URL": "https://member:url-password@anthropic-proxy.example"},
            "ANTHROPIC_BASE_URL must be an absolute HTTP or HTTPS URL",
            id="anthropic-credentials-in-url",
        ),
        pytest.param(
            "anthropic",
            "claude-opus-5",
            {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1/messages"},
            "ANTHROPIC_BASE_URL must be an absolute HTTP or HTTPS URL",
            id="anthropic-endpoint-not-root",
        ),
    ],
)
def test_member_with_an_unusable_endpoint_fails_before_any_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    environment: dict[str, str],
    expected_error: str,
) -> None:
    """Reject a member endpoint that is not a plain absolute HTTP(S) URL before sending its key."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget(provider, model)
    )

    assert content is None
    assert expected_error in error
    assert "url-password" not in error
    assert provenance == {"provider": provider, "model": model}
    assert transport.requests == []


def test_unknown_member_provider_fails_without_a_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a target whose provider has no credential route instead of guessing one."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget("mystery", "model-x")
    )

    assert content is None
    assert error == "Unsupported judge panel provider: mystery"
    assert provenance == {"provider": "mystery", "model": "model-x"}
    assert transport.requests == []


def test_bedrock_member_uses_its_aws_region_and_never_an_http_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call Bedrock members through the AWS chain in AWS_REGION, never through urllib."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    clients: list[dict[str, Any]] = []

    class _FakeBedrockRuntime:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            clients[-1]["converse"] = kwargs
            return {"output": {"message": {"content": [{"text": "bedrock verdict"}]}}}

    def _client(service: str, **kwargs: Any) -> _FakeBedrockRuntime:
        clients.append({"service": service, **kwargs})
        return _FakeBedrockRuntime()

    fake_boto3 = ModuleType("boto3")
    fake_boto3.client = _client
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setattr(
        verifier.urllib.request, "urlopen", lambda *_args, **_kwargs: pytest.fail("bedrock must not use urllib")
    )

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response", target=verifier.JudgeTarget("bedrock", "us.anthropic.claude-opus-5-v1:0")
    )

    assert (content, error) == ("bedrock verdict", None)
    assert provenance == {"provider": "bedrock", "model": "us.anthropic.claude-opus-5-v1:0"}
    assert len(clients) == 1
    assert clients[0]["service"] == "bedrock-runtime"
    assert clients[0]["region_name"] == "eu-central-1"
    assert clients[0]["converse"]["modelId"] == "us.anthropic.claude-opus-5-v1:0"


# ---------------------------------------------------------------------------
# Members never change model: no overrides, no fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("provider", "model"), MEMBERS)
def test_model_fallbacks_never_apply_to_a_panel_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
) -> None:
    """Make exactly one request for a member whose model is missing, despite configured fallbacks."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "fallback-model-a,fallback-model-b")
    requests: list[urllib.request.Request] = []

    def model_not_found(request: urllib.request.Request, timeout: float | None = None) -> None:
        requests.append(request)
        raise _http_error(request.full_url, 404, '{"error": {"message": "model not found"}}')

    monkeypatch.setattr(verifier.urllib.request, "urlopen", model_not_found)

    content, error, provenance = verifier._call_public_llm_with_provenance(
        "Judge this response",
        allow_model_fallback=True,
        target=verifier.JudgeTarget(provider, model),
    )

    assert content is None
    assert error.startswith("HTTP 404")
    assert provenance == {"provider": provider, "model": model}
    assert [json.loads(request.data)["model"] for request in requests] == [model]


def test_single_judge_still_walks_the_fallback_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: without a target the same failure still tries every configured fallback model."""
    verifier = _load_verifier(tmp_path)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", MEMBER_KEYS["openai"][1])
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "fallback-model-a,fallback-model-b")
    requests: list[urllib.request.Request] = []

    def model_not_found(request: urllib.request.Request, timeout: float | None = None) -> None:
        requests.append(request)
        raise _http_error(request.full_url, 404, '{"error": {"message": "model not found"}}')

    monkeypatch.setattr(verifier.urllib.request, "urlopen", model_not_found)

    content, error, _provenance = verifier._call_public_llm_with_provenance("Judge this response", model="gpt-missing")

    assert content is None
    assert error.startswith("LLM judge model fallback exhausted:")
    assert [json.loads(request.data)["model"] for request in requests] == [
        "gpt-missing",
        "fallback-model-a",
        "fallback-model-b",
    ]


def test_judge_model_overrides_never_rename_a_panel_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore LLM_JUDGE_MODEL, SKILL_EVAL_JUDGE_MODEL, SKILL_EVAL_LLM_MODEL, and model= for members."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_BASE_URL", GATEWAY_BASE_URL)
    for variable in ("LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL", "SKILL_EVAL_LLM_MODEL"):
        monkeypatch.setenv(variable, f"override-from-{variable.lower()}")
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    provenances = [
        verifier._call_public_llm_with_provenance(
            "Judge this response", model="explicit-override", target=verifier.JudgeTarget(provider, model)
        )[2]
        for provider, model in MEMBERS
    ]

    assert provenances == [{"provider": provider, "model": model} for provider, model in MEMBERS]
    assert [json.loads(request.data)["model"] for request in transport.requests] == [model for _, model in MEMBERS]


def test_active_target_routes_judge_calls_and_an_explicit_target_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route call_public_llm through the active target, let an explicit target win, then restore the judge."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "single-judge-model")
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    token = verifier._ACTIVE_JUDGE_TARGET.set(verifier.JudgeTarget("anthropic", "claude-opus-5"))
    try:
        assert verifier.call_public_llm("Judge this response") == ("plain judge reply", None)
        explicit = verifier._call_public_llm_with_provenance(
            "Judge this response", target=verifier.JudgeTarget("openai", "gpt-5.6-sol")
        )
    finally:
        verifier._ACTIVE_JUDGE_TARGET.reset(token)
    single = verifier._call_public_llm_with_provenance("Judge this response")

    assert explicit[2] == {"provider": "openai", "model": "gpt-5.6-sol"}
    assert single[2] == {"provider": "nv_build", "model": "single-judge-model"}
    assert [(request.full_url, json.loads(request.data)["model"]) for request in transport.requests] == [
        (ANTHROPIC_URL, "claude-opus-5"),
        (OPENAI_URL, "gpt-5.6-sol"),
        (NVIDIA_URL, "single-judge-model"),
    ]


def test_panel_member_failure_makes_one_request_per_metric_in_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail a member whose model is missing after one request per metric while the quorum still scores."""
    verifier = _load_verifier(tmp_path)
    _set_member_keys(monkeypatch)
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_JUDGE_FALLBACK_MODELS", "fallback-model-a")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "override-model")
    monkeypatch.setenv(
        "SKILL_EVAL_JUDGE_PANEL",
        "openai:gpt-missing,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b",
    )

    def reply(request: urllib.request.Request) -> dict[str, Any]:
        if request.full_url == OPENAI_URL:
            raise _http_error(request.full_url, 404, '{"error": {"message": "model not found"}}')
        return _default_reply(request)

    transport = _Transport(reply)
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    verifier.main()

    openai_requests = [request for request in transport.requests if request.full_url == OPENAI_URL]
    assert [(_judge_kind(request), json.loads(request.data)["model"]) for request in openai_requests] == [
        (metric, "gpt-missing") for metric in METRICS
    ]
    assert {json.loads(request.data)["model"] for request in transport.requests} == {
        "gpt-missing",
        "claude-opus-5",
        "nvidia/nemotron-3-super-120b-a12b",
    }
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    for metric in METRICS:
        panel = rich["details"][metric]["panel"]
        assert panel["failed_members"] == 1
        assert panel["members"][0]["status"] == "error"
        assert "HTTP 404" in panel["members"][0]["reason"]


# ---------------------------------------------------------------------------
# Host-to-verifier boundary: the environment the runner delivers routes correctly
# ---------------------------------------------------------------------------


def _harbor_verifier_environment(monkeypatch: pytest.MonkeyPatch, host_env: dict[str, str]) -> dict[str, str]:
    """Rebuild the verifier environment Harbor assembles for a panel run on this host environment.

    Task-level ``[verifier.env]`` forwards the primary provider's allowlisted
    variables; the job-level ``--verifier-env`` entries the runner adds for the
    panel override them, with each ``${NAME}`` resolved from the Harbor parent
    process environment exactly once.
    """
    # Host imports stay local so the container-only routing tests never depend on runner internals.
    from skillevaluator.provider_config import resolve_judge_panel_config, resolve_llm_provider
    from skillevaluator.tier3.harbor.adapter import _VERIFIER_PROVIDER_ENV_VARS
    from skillevaluator.tier3.harbor.runner import _judge_panel_harbor_environment, _provider_environment

    for name, value in host_env.items():
        monkeypatch.setenv(name, value)
    provider_env = _provider_environment(resolve_llm_provider())
    panel = resolve_judge_panel_config()
    assert panel is not None
    subprocess_additions, job_verifier_env = _judge_panel_harbor_environment(panel, provider_env)
    harbor_parent = {**provider_env, **subprocess_additions}
    task_level = {name: value for name, value in provider_env.items() if name in _VERIFIER_PROVIDER_ENV_VARS}
    job_level = {
        name: re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda match: harbor_parent[match.group(1)], value)
        for name, value in job_verifier_env.items()
    }
    for name in host_env:
        monkeypatch.delenv(name)
    return {**task_level, **job_level}


@pytest.mark.parametrize(
    "primary_env",
    [
        pytest.param({"SKILL_EVAL_LLM_PROVIDER": "nv_build"}, id="nv-build-primary"),
        # A gateway primary arrives as OPENAI_*; the native openai member must not inherit it.
        pytest.param(
            {"SKILL_EVAL_LLM_PROVIDER": "openai-compatible", "SKILL_EVAL_LLM_MODEL": "gateway/agent-model"},
            id="gateway-primary",
        ),
    ],
)
def test_host_delivered_panel_environment_routes_each_member_to_its_own_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_env: dict[str, str],
) -> None:
    """Route every member correctly in the exact environment the runner hands the verifier."""
    host_env = {
        **dict(MEMBER_KEYS.values()),
        "SKILL_EVAL_LLM_BASE_URL": "https://gateway.example/v1",
        "SKILL_EVAL_JUDGE_PANEL": PANEL,
        **primary_env,
    }
    container_env = _harbor_verifier_environment(monkeypatch, host_env)
    verifier = _load_verifier(tmp_path)
    for name, value in container_env.items():
        monkeypatch.setenv(name, value)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    verifier.main()

    expected_urls = {
        "nv_build": NVIDIA_URL,
        "openai": OPENAI_URL,
        "anthropic": ANTHROPIC_URL,
        "openai-compatible": "https://gateway.example/v1/chat/completions",
    }
    url_providers = {url: provider for provider, url in expected_urls.items()}
    assert [(request.full_url, json.loads(request.data)["model"]) for request in transport.requests] == [
        (expected_urls[provider], model) for _metric in METRICS for provider, model in MEMBERS
    ]
    for request in transport.requests:
        _assert_carries_only_its_own_key(request, url_providers[request.full_url])
    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    assert all(rich["details"][metric]["panel"]["failed_members"] == 0 for metric in METRICS)


# ---------------------------------------------------------------------------
# Golden: no panel means the single-judge path is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel_value", [None, "", "   "], ids=["unset", "empty-placeholder", "blank"])
def test_without_a_panel_the_single_judge_request_sequence_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    panel_value: str | None,
) -> None:
    """Pin the single nv_build judge's requests and artifacts when no panel is configured."""
    verifier = _load_verifier(tmp_path)
    key = "nvapi-SingleJudgeGoldenKey0005"
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "nv_build")
    monkeypatch.setenv("NVIDIA_API_KEY", key)
    if panel_value is not None:
        for name in (
            "SKILL_EVAL_JUDGE_PANEL",
            "SKILL_EVAL_JUDGE_PANEL_AGGREGATION",
            "SKILL_EVAL_JUDGE_PANEL_QUORUM",
            "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT",
        ):
            monkeypatch.setenv(name, panel_value)
    transport = _Transport()
    monkeypatch.setattr(verifier.urllib.request, "urlopen", transport)

    verifier.main()

    assert [
        (
            request.full_url,
            sorted(request.header_items()),
            list(json.loads(request.data)),
            json.loads(request.data)["model"],
            json.loads(request.data)["max_tokens"],
            json.loads(request.data)["response_format"]["json_schema"]["name"],
        )
        for request in transport.requests
    ] == [
        (
            NVIDIA_URL,
            [("Authorization", f"Bearer {key}"), ("Content-type", "application/json")],
            ["model", "max_tokens", "messages", "stream", "response_format"],
            "gpt-5.6-sol",
            4096,
            schema_name,
        )
        for schema_name in ("accuracy_judgment", "goal_accuracy_judgment", "behavior_check_judgment")
    ]

    rich = json.loads(verifier.SKILL_EVALUATOR_REWARD_JSON.read_text(encoding="utf-8"))
    numeric = json.loads(verifier.REWARD_JSON.read_text(encoding="utf-8"))
    assert numeric == {
        "security": 1.0,
        "skill_execution": 1.0,
        "skill_efficiency": 1.0,
        "accuracy": 0.8,
        "goal_accuracy": 1.0,
        "behavior_check": 1.0,
        "overall": 0.9667,
    }
    assert list(rich) == [
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
    assert _keys_named(rich, "panel") == []
    assert _keys_named(rich["details"], "achieved") == []
    assert rich["details"]["accuracy"] == {
        "score": 0.8,
        "reason": "mostly accurate",
        "criteria": _ACCURACY_VERDICT["criteria"],
        **{key: rich["details"]["accuracy"][key] for key in ("evidence_refs", "omitted")},
    }
    assert list(rich["details"]["goal_accuracy"]) == [
        "score",
        "reason",
        "user_goal",
        "end_state",
        "method",
        "provider",
        "model",
        "evidence_refs",
        "omitted",
    ]
    assert {key: rich["details"]["goal_accuracy"][key] for key in ("score", "method", "provider", "model")} == {
        "score": 1.0,
        "method": "custom",
        "provider": "nv_build",
        "model": "gpt-5.6-sol",
    }
    assert list(rich["details"]["behavior_check"]) == ["score", "reason", "results", "evidence_refs", "omitted"]
    assert rich["details"]["behavior_check"]["results"] == _BEHAVIOR_VERDICT["results"]
    assert verifier.REWARD_TXT.read_text(encoding="utf-8") == "0.9667"
