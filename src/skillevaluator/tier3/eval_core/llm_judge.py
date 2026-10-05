# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM judge prompt builders and public-provider caller.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, NamedTuple
from urllib.parse import urlparse

from skillevaluator.inference.types import EmptyLLMResponseError
from skillevaluator.provider_config import (
    CHAT_DEFAULT_OPENAI,
    OPENAI_BASE_URL,
    PUBLIC_NVIDIA_BUILD_BASE_URL,
    ProviderConfig,
    ProviderConfigurationError,
    _model_leaf,
    _normalize_anthropic_base_url,
    _supports_custom_temperature,
)
from skillevaluator.tier3.eval_core.atif_helpers import (
    _SECTION_COMPACT_TOOL_HISTORY,
    _SECTION_FINAL_RESPONSE,
    _SECTION_USER_REQUEST,
    _behavior_check_budget,
    _behavior_final_response_limit,
    _truncate_for_behavior,
)

logger = logging.getLogger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_BUILD_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_JUDGE_MODEL = CHAT_DEFAULT_OPENAI

_ERROR_REDACTION_MARKER = "[REDACTED]"
_JUDGE_ERROR_REASON_LIMIT = 512
_JUDGE_TEXT_LIMIT = 512
# Match verifier log redaction; shorter placeholders can corrupt ordinary diagnostic text.
_MIN_EXACT_SECRET_LENGTH = 8
_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "NVIDIA_API_KEY",
    "ANTHROPIC_API_KEY",
    "SKILL_EVAL_LLM_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)


# ---------------------------------------------------------------------------
# Cross-model judge panel
# ---------------------------------------------------------------------------

# --- BEGIN SHARED JUDGE PANEL HELPERS (verbatim copy in harbor/templates/eval.py) ---
# A judge panel scores each LLM metric with several provider:model judges and
# aggregates their verdicts. The standalone verifier cannot import skillevaluator,
# so templates/eval.py carries a byte-for-byte copy of this block, enforced by a
# parity test. Use only builtins, math, re, statistics, ContextVar, NamedTuple,
# Any, and Mapping here.


class JudgeTarget(NamedTuple):
    """Name one judge panel member by provider and model id."""

    provider: str
    model: str


class JudgePanelSettings(NamedTuple):
    """Hold a validated judge panel and its aggregation settings."""

    members: tuple[JudgeTarget, ...]
    aggregation: str
    quorum: int
    disagreement_threshold: float


# Set while one panel member judges so provider calls route to that member only.
_ACTIVE_JUDGE_TARGET: ContextVar[JudgeTarget | None] = ContextVar("active_judge_target", default=None)

JUDGE_PANEL_ENV = "SKILL_EVAL_JUDGE_PANEL"
JUDGE_PANEL_AGGREGATION_ENV = "SKILL_EVAL_JUDGE_PANEL_AGGREGATION"
JUDGE_PANEL_QUORUM_ENV = "SKILL_EVAL_JUDGE_PANEL_QUORUM"
JUDGE_PANEL_DISAGREEMENT_ENV = "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT"
JUDGE_PANEL_AGGREGATIONS = ("vote", "median", "mean")
JUDGE_PANEL_MAX_MEMBERS = 5
DEFAULT_JUDGE_PANEL_DISAGREEMENT = 0.4

_JUDGE_PANEL_PROVIDERS = ("anthropic", "bedrock", "nv_build", "openai", "openai-compatible")
_JUDGE_PANEL_METRICS = ("accuracy", "goal_accuracy", "behavior_check")
_JUDGE_PANEL_CRITERIA = ("SKILL_IDENTIFIED", "ACTION_CORRECT", "FACTUALLY_ACCURATE", "TASK_ADDRESSED", "ACTIONABLE")
_JUDGE_PANEL_TEXT_LIMIT = 512
# Bedrock ids name the vendor after an optional inference-profile region, as in us.anthropic.claude-opus-5.
_MODEL_FAMILY_VENDOR_RE = re.compile(
    r"^(?:(?:us|eu|apac|ap|ca|jp|au|global|us-gov)\.)?"
    r"(anthropic|meta|mistral|amazon|cohere|ai21|deepseek|qwen|openai|google|nvidia)\."
)
# Any vendor in that position, so ids of unlisted Bedrock vendors such as zai.glm-4.6 reach the model name.
_MODEL_FAMILY_ANY_VENDOR_RE = re.compile(r"^(?:(?:us|eu|apac|ap|ca|jp|au|global|us-gov)\.)?[a-z0-9-]+\.")
_MODEL_FAMILY_O_SERIES_RE = re.compile(r"^o\d+(?:$|[-_.:])")
_MODEL_FAMILY_PREFIXES = {
    "anthropic": ("claude",),
    "openai": ("gpt", "chatgpt", "codex", "davinci"),
    "nvidia": ("nemotron", "nvidia"),
    "meta": ("llama", "meta-llama"),
    "mistral": ("mistral", "mixtral", "codestral", "ministral", "magistral", "devstral", "pixtral"),
    "google": ("gemini", "gemma"),
    "qwen": ("qwen", "qwq"),
    "deepseek": ("deepseek",),
    "microsoft": ("phi",),
    "ibm": ("granite",),
    "xai": ("grok",),
    "moonshot": ("kimi",),
    "zhipu": ("glm",),
    "amazon": ("nova", "titan"),
    "cohere": ("command",),
    "ai21": ("jamba",),
}
_MODEL_FAMILY_ORGS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "nvidia": "nvidia",
    "meta": "meta",
    "meta-llama": "meta",
    "mistralai": "mistral",
    "mistral": "mistral",
    "google": "google",
    "qwen": "qwen",
    "deepseek-ai": "deepseek",
    "deepseek": "deepseek",
    "microsoft": "microsoft",
    "ibm": "ibm",
    "ibm-granite": "ibm",
    "xai": "xai",
    "moonshotai": "moonshot",
    "zhipuai": "zhipu",
    "thudm": "zhipu",
    "amazon": "amazon",
    "cohere": "cohere",
}
# nv_build, bedrock, and gateways host many families, so only these providers imply one.
_MODEL_FAMILY_PROVIDERS = {"openai": "openai", "anthropic": "anthropic"}


def parse_judge_panel_env(environ: Mapping[str, str]) -> JudgePanelSettings | None:
    """Return the configured judge panel, or ``None`` when ``SKILL_EVAL_JUDGE_PANEL`` is unset or blank.

    Raises ``ValueError`` for any invalid setting so callers fail closed rather
    than silently falling back to a single judge.
    """
    panel_text = str(environ.get(JUDGE_PANEL_ENV) or "").strip()
    knobs = {
        name: str(environ.get(name) or "").strip()
        for name in (JUDGE_PANEL_AGGREGATION_ENV, JUDGE_PANEL_QUORUM_ENV, JUDGE_PANEL_DISAGREEMENT_ENV)
    }
    if not panel_text:
        if orphans := [name for name, value in knobs.items() if value]:
            raise ValueError(f"{', '.join(orphans)} set without {JUDGE_PANEL_ENV}; name the judges or unset the knobs")
        return None

    members: list[JudgeTarget] = []
    for raw_entry in panel_text.split(","):
        entry = raw_entry.strip()
        if not entry:
            raise ValueError(f"{JUDGE_PANEL_ENV} contains an empty entry")
        # Split on the first colon only: Bedrock model ids such as ...-v1:0 contain colons.
        provider, _, model = entry.partition(":")
        target = JudgeTarget(provider.strip().lower(), model.strip())
        if not target.provider or not target.model:
            raise ValueError(f"{JUDGE_PANEL_ENV} entry {entry!r} must have the form provider:model")
        if target.provider not in _JUDGE_PANEL_PROVIDERS:
            raise ValueError(
                f"{JUDGE_PANEL_ENV} entry {entry!r} has unsupported provider {target.provider!r}; "
                f"expected one of: {', '.join(_JUDGE_PANEL_PROVIDERS)}"
            )
        # isprintable() is False exactly for Unicode "Other" and "Separator" characters.
        if any(character.isspace() for character in target.model) or not target.model.isprintable():
            raise ValueError(
                f"{JUDGE_PANEL_ENV} model {target.model!r} must not contain whitespace or control characters"
            )
        if target in members:
            raise ValueError(f"{JUDGE_PANEL_ENV} lists {target.provider}:{target.model} more than once")
        if target.provider == "openai-compatible" and any(member.provider == target.provider for member in members):
            raise ValueError(
                f"{JUDGE_PANEL_ENV} may name at most one openai-compatible judge because "
                "SKILL_EVAL_LLM_BASE_URL and SKILL_EVAL_LLM_API_KEY configure a single gateway"
            )
        members.append(target)
    if len(members) > JUDGE_PANEL_MAX_MEMBERS:
        raise ValueError(
            f"{JUDGE_PANEL_ENV} names {len(members)} judges; at most {JUDGE_PANEL_MAX_MEMBERS} are allowed"
        )

    aggregation = knobs[JUDGE_PANEL_AGGREGATION_ENV].lower() or "vote"
    if aggregation not in JUDGE_PANEL_AGGREGATIONS:
        raise ValueError(f"{JUDGE_PANEL_AGGREGATION_ENV} must be one of: {', '.join(JUDGE_PANEL_AGGREGATIONS)}")
    quorum = len(members) // 2 + 1
    if quorum_text := knobs[JUDGE_PANEL_QUORUM_ENV]:
        try:
            quorum = int(quorum_text)
        except ValueError:
            quorum = 0  # rejected by the range check below
        if not 1 <= quorum <= len(members):
            raise ValueError(f"{JUDGE_PANEL_QUORUM_ENV} must be an integer from 1 to {len(members)}")
    threshold = DEFAULT_JUDGE_PANEL_DISAGREEMENT
    if threshold_text := knobs[JUDGE_PANEL_DISAGREEMENT_ENV]:
        try:
            threshold = float(threshold_text)
        except ValueError:
            threshold = math.nan
        # The chained comparison is False for NaN, so nan and inf fail with out-of-range values.
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{JUDGE_PANEL_DISAGREEMENT_ENV} must be a number from 0 to 1")
    return JudgePanelSettings(tuple(members), aggregation, quorum, threshold)


def _model_family_by_prefix(name: str) -> str | None:
    if _MODEL_FAMILY_O_SERIES_RE.match(name):
        return "openai"
    for family, prefixes in _MODEL_FAMILY_PREFIXES.items():
        for prefix in prefixes:
            # The prefix must end at a non-letter: phi-4 is Microsoft, philosopher-7b is not.
            if name.startswith(prefix) and not name[len(prefix) : len(prefix) + 1].isalpha():
                return family
    return None


def _model_family(provider: str | None, model: str | None) -> str:
    """Infer a model family from the model id, falling back to the provider only when the id is unknown.

    Hosted catalogs mix vendors, so the id decides: nv_build serves
    ``nvidia/llama-3.1-nemotron-70b-instruct``, a Meta model. ``"unknown"``
    never counts as the same family as anything.
    """
    segments = [segment for segment in str(model or "").strip().casefold().split("/") if segment]
    leaf = segments[-1].removeprefix("bedrock-") if segments else ""
    if vendor := _MODEL_FAMILY_VENDOR_RE.match(leaf):
        return vendor.group(1)
    family = _model_family_by_prefix(leaf)
    if family is None and (unlisted_vendor := _MODEL_FAMILY_ANY_VENDOR_RE.match(leaf)):
        family = _model_family_by_prefix(leaf[unlisted_vendor.end() :])
    if family is not None:
        return family
    for segment in reversed(segments[:-1]):
        if org_family := _MODEL_FAMILY_ORGS.get(segment):
            return org_family
    return _MODEL_FAMILY_PROVIDERS.get(str(provider or "").strip().casefold(), "unknown")


def _judge_panel_text(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if len(text) > _JUDGE_PANEL_TEXT_LIMIT:
        text = text[: _JUDGE_PANEL_TEXT_LIMIT - 3] + "..."
    return text


def _judge_panel_behavior_results(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    results = result.get("results")
    if isinstance(results, list) and all(
        isinstance(item, dict) and isinstance(item.get("passed"), bool) for item in results
    ):
        return results
    return None


def _judge_panel_vote(ballot: list[bool]) -> tuple[bool | None, float, float]:
    """Return the majority verdict, its score value, and the share of judges agreeing with it.

    A tie is undecided: ``None``, worth 0.5, which favors neither side.
    """
    yes = sum(ballot)
    share = max(yes, len(ballot) - yes) / len(ballot)
    if yes * 2 == len(ballot):
        return None, 0.5, share
    verdict = yes * 2 > len(ballot)
    return verdict, float(verdict), share


def _judge_panel_member(
    metric: str,
    provider: str,
    model: str,
    result: Any,
    aggregation: str,
    expected_count: int | None,
) -> dict[str, Any]:
    """Build one member's panel entry; a result unusable for this aggregation becomes an error entry."""
    identity = {"provider": provider, "model": model, "family": _model_family(provider, model)}

    def failed(reason: Any) -> dict[str, Any]:
        return {**identity, "status": "error", "reason": _judge_panel_text(reason) or "LLM judge failed"}

    if not isinstance(result, dict):
        return failed("Judge returned an invalid result")
    if str(result.get("status", "")).casefold() == "error":
        return failed(result.get("reason"))
    score = result.get("score")
    # The range check is False for NaN, so it also rejects non-finite scores.
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
        return failed("Judge returned no finite score between 0 and 1")
    entry = {
        **identity,
        "status": "ok",
        "score": round(float(score), 4),
        "reason": _judge_panel_text(result.get("reason")),
    }
    if metric == "accuracy":
        criteria = result.get("criteria")
        complete = (
            isinstance(criteria, dict)
            and set(criteria) == set(_JUDGE_PANEL_CRITERIA)
            and all(isinstance(criteria[key], bool) for key in _JUDGE_PANEL_CRITERIA)
        )
        if aggregation == "vote" and not complete:
            return failed("Judge returned incomplete accuracy criteria")
        entry["criteria"] = {key: criteria[key] for key in _JUDGE_PANEL_CRITERIA} if complete else {}
    elif metric == "behavior_check":
        results = _judge_panel_behavior_results(result)
        if results is None:
            return failed("Judge returned malformed behavior results")
        if expected_count is not None and len(results) != expected_count:
            return failed(f"behavior result count {len(results)} does not match expected {expected_count}")
        entry["results"] = [
            {"step": index + 1, "passed": item["passed"], "reason": _judge_panel_text(item.get("reason"))}
            for index, item in enumerate(results)
        ]
    else:
        achieved = result.get("achieved")
        if aggregation == "vote" and not isinstance(achieved, bool):
            return failed("Judge returned no boolean achieved verdict")
        entry["achieved"] = achieved if isinstance(achieved, bool) else None
        entry["method"] = _judge_panel_text(result.get("method")) or "custom"
    return entry


def aggregate_panel(
    metric: str,
    member_results: list[tuple[str, str, Any]],
    *,
    aggregation: str = "vote",
    quorum: int | None = None,
    disagreement_threshold: float = DEFAULT_JUDGE_PANEL_DISAGREEMENT,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate one LLM metric's per-member results into a single result with a ``panel`` block.

    ``member_results`` holds ``(provider, model, result)`` in panel order, where
    ``result`` is a normalized judge result or a stored panel member entry.
    ``vote`` takes the majority per accuracy criterion, per behavior, or on goal
    ``achieved``; ``median`` and ``mean`` combine member scores. Fewer than
    ``quorum`` usable members yields ``status="error"`` with ``score=None``,
    never a zero score.
    """
    if metric not in _JUDGE_PANEL_METRICS:
        raise ValueError(f"Unsupported judge panel metric: {metric!r}")
    if aggregation not in JUDGE_PANEL_AGGREGATIONS:
        raise ValueError(f"Unsupported judge panel aggregation: {aggregation!r}")
    rows = list(member_results)
    quorum = len(rows) // 2 + 1 if quorum is None else quorum

    def entries(count: int | None) -> list[dict[str, Any]]:
        return [
            _judge_panel_member(metric, provider, model, result, aggregation, count) for provider, model, result in rows
        ]

    members = entries(expected_count)
    if metric == "behavior_check" and expected_count is None:
        # Stored entries may be re-aggregated without the behavior count; the most common length wins.
        lengths = [len(member["results"]) for member in members if member["status"] == "ok"]
        if lengths:
            expected_count = max(lengths, key=lengths.count)
            members = entries(expected_count)

    ok = [member for member in members if member["status"] == "ok"]
    scores = [member["score"] for member in ok]
    # Structured verdicts are voted in every mode; median and mean report them for explainability only.
    if metric == "accuracy":
        voters = [member["criteria"] for member in ok if member["criteria"]]
        ballots = [[voter[key] for voter in voters] for key in _JUDGE_PANEL_CRITERIA] if voters else []
    elif metric == "behavior_check":
        count = expected_count if ok and expected_count else 0
        ballots = [[member["results"][index]["passed"] for member in ok] for index in range(count)]
    else:
        ballot = [member["achieved"] for member in ok if member["achieved"] is not None]
        ballots = [ballot] if ballot else []
    tallies = [_judge_panel_vote(ballot) for ballot in ballots]
    voted_score = statistics.mean(value for _verdict, value, _share in tallies) if tallies else None
    if metric == "accuracy":
        fields: dict[str, Any] = {
            "criteria": {key: tallies[index][0] for index, key in enumerate(_JUDGE_PANEL_CRITERIA)} if tallies else {}
        }
    elif metric == "behavior_check":
        fields = {
            "results": [
                {
                    "step": index + 1,
                    "passed": verdict,
                    "reason": f"{sum(ballots[index])}/{len(ballots[index])} judges observed this behavior",
                }
                for index, (verdict, _value, _share) in enumerate(tallies)
            ]
        }
    else:
        achieved = tallies[0][0] if tallies else None
        fields = {"achieved": achieved, "method": "custom"}
        if achieved is not None:
            # The score follows the judges who carried the vote.
            voted_score = statistics.median(member["score"] for member in ok if member["achieved"] is achieved)

    failed_count = len(rows) - len(ok)
    spread = round(max(scores) - min(scores), 4) if scores else None
    shares = [share for _verdict, _value, share in tallies]
    panel = {
        "aggregation": aggregation,
        "quorum": quorum,
        "members": members,
        "spread": spread,
        "agreement": round(statistics.mean(shares), 4) if shares else None,
        "disagreement": spread is not None and spread >= disagreement_threshold - 1e-9,
        "disagreement_threshold": disagreement_threshold,
        "failed_members": failed_count,
    }
    if len(ok) < max(quorum, 1):
        error: dict[str, Any] = {
            "score": None,
            "status": "error",
            "reason": (
                f"Judge panel quorum not met for {metric}: {len(ok)}/{len(rows)} judges succeeded (quorum {quorum})"
            ),
        }
        if metric == "behavior_check":
            error["results"] = []
        elif metric == "goal_accuracy":
            error["method"] = "custom"
        return {**error, "panel": panel}

    if aggregation == "mean":
        score = statistics.mean(scores)
    elif aggregation == "median" or voted_score is None:
        # A behavior check without behaviors has nothing to vote on.
        score = statistics.median(scores)
    else:
        score = voted_score
    failures = f"; {failed_count} failed" if failed_count else ""
    return {
        "score": round(score, 4),
        "reason": f"panel {aggregation} ({len(ok)}/{len(rows)} judges{failures})",
        **fields,
        "panel": panel,
    }


# --- END SHARED JUDGE PANEL HELPERS ---


# ---------------------------------------------------------------------------
# Public provider HTTP caller
# ---------------------------------------------------------------------------


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        model = str(model or "").strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _fallback_models(primary_model: str) -> list[str]:
    env_fallbacks = [
        item.strip() for item in os.environ.get("LLM_JUDGE_FALLBACK_MODELS", "").split(",") if item.strip()
    ]
    return _dedupe_models([primary_model, *env_fallbacks])


def _provider() -> str:
    configured = os.environ.get("SKILL_EVAL_LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("NVIDIA_API_KEY"):
        return "nv_build"
    return ""


def _resolve_url(provider: str) -> str:
    if provider == "nv_build":
        return os.environ.get("SKILL_EVAL_LLM_BASE_URL") or NVIDIA_BUILD_CHAT_URL
    base_url = os.environ.get("SKILL_EVAL_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    return base_url.rstrip("/") + "/chat/completions" if base_url else OPENAI_CHAT_URL


def _is_native_openai_chat_url(provider: str, request_url: str) -> bool:
    if str(provider or "").strip().casefold() != "openai":
        return False

    raw_url = str(request_url or "")
    if raw_url != raw_url.strip() or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw_url):
        return False
    try:
        parsed = urlparse(raw_url)
        port = parsed.port
    except ValueError:
        return False

    return (
        parsed.scheme.casefold() == "https"
        and parsed.hostname is not None
        and parsed.hostname.casefold() == "api.openai.com"
        and parsed.netloc.casefold() in {"api.openai.com", "api.openai.com:443"}
        and port in {None, 443}
        and parsed.path in {"/v1/chat/completions", "/v1/chat/completions/"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and ";" not in raw_url
        and "?" not in raw_url
        and "#" not in raw_url
    )


def _configured_secret_values(extra_secret_values: tuple[str | None, ...] = ()) -> list[str]:
    values = {
        value
        for name in _CREDENTIAL_ENV_VARS
        if (value := os.environ.get(name, "")) and len(value) >= _MIN_EXACT_SECRET_LENGTH
    }
    for value in extra_secret_values:
        text = str(value) if value else ""
        if len(text) >= _MIN_EXACT_SECRET_LENGTH:
            values.add(text)
    return sorted(values, key=len, reverse=True)


def _redact_configured_credentials(text: str, extra_secret_values: tuple[str | None, ...] = ()) -> str:
    redacted = str(text)
    for secret in _configured_secret_values(extra_secret_values):
        redacted = redacted.replace(secret, _ERROR_REDACTION_MARKER)
    return redacted


def _judge_error(error_reason: str, **metadata: Any) -> dict[str, Any]:
    """Return a bounded, redacted result that cannot be mistaken for a judged zero."""
    safe_reason = _redact_configured_credentials(error_reason).strip() or "LLM judge failed"
    if len(safe_reason) > _JUDGE_ERROR_REASON_LIMIT:
        safe_reason = safe_reason[: _JUDGE_ERROR_REASON_LIMIT - 3] + "..."
    return {**metadata, "score": None, "status": "error", "reason": safe_reason}


def _bounded_judge_text(value: Any) -> str:
    """Normalize trusted-shape model text before it reaches artifacts and reports."""
    text = _redact_configured_credentials(value).strip() if isinstance(value, str) else ""
    if len(text) > _JUDGE_TEXT_LIMIT:
        text = text[: _JUDGE_TEXT_LIMIT - 3] + "..."
    return text


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, int):
        if value <= 0:
            return 0.0
        return 1.0
    score = float(value)
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _format_http_error_with_fallback(error: urllib.error.HTTPError) -> tuple[str, bool]:
    try:
        body = error.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    raw_detail = f"HTTP {error.code}: {error.reason}"
    safe_detail = raw_detail
    if body:
        raw_detail = f"{raw_detail} - {body}"
        safe_detail = f"{safe_detail} - {_redact_configured_credentials(body)[:500]}"
    return _redact_configured_credentials(safe_detail), _should_try_fallback(raw_detail)


def _format_http_error(error: urllib.error.HTTPError) -> str:
    return _format_http_error_with_fallback(error)[0]


def _should_try_fallback(error: str) -> bool:
    text = error.lower()
    return (
        "key_model_access_denied" in text
        or "not allowed to access model" in text
        or "invalid model" in text
        or "model not found" in text
    )


def _chat_completion_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    provider: str | None = None,
    request_url: str | None = None,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "judge_response",
) -> dict[str, Any]:
    resolved_provider = _provider() if provider is None else provider
    resolved_request_url = _resolve_url(resolved_provider) if request_url is None else request_url
    token_key = (
        "max_completion_tokens"
        if _model_leaf(model).startswith("gpt-5")
        and _is_native_openai_chat_url(resolved_provider, resolved_request_url)
        else "max_tokens"
    )
    payload: dict[str, Any] = {
        "model": model,
        token_key: max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if temperature is not None and _supports_custom_temperature(model):
        payload["temperature"] = temperature
    if response_schema is not None:
        from skillevaluator.inference.client import _build_openai_response_format

        payload["response_format"] = _build_openai_response_format(response_schema, schema_name)
    return payload


_ANTHROPIC_API_ROOT = "https://api.anthropic.com"
# A panel member reads only its own provider's key; there is no cross-provider fallback.
_JUDGE_TARGET_KEY_ENV = {
    "nv_build": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "SKILL_EVAL_LLM_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _judge_target_provider_config(target: JudgeTarget) -> tuple[ProviderConfig | None, str | None]:
    """Resolve a panel member's endpoint and key from its own provider variables only.

    Only the ``openai-compatible`` member reads ``SKILL_EVAL_LLM_BASE_URL``, so
    a native provider key is not sent to that gateway URL. The host already
    delivers the primary provider's endpoint in its native ``*_BASE_URL``.
    """
    provider, model = target.provider, target.model
    if provider == "bedrock":
        config = ProviderConfig(
            provider=provider,
            model=model,
            api_key=None,
            base_url=None,
            litellm_model=f"bedrock/{model}",
            region=os.environ.get("AWS_REGION") or "us-west-2",
        )
        return config, None
    key_env = _JUDGE_TARGET_KEY_ENV.get(provider)
    if key_env is None:
        return None, f"Unsupported judge panel provider: {provider}"
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        return None, f"No API key configured for {provider} judge panel member ({key_env})"

    base_url_env: str | None = None
    if provider == "nv_build":
        base_url = PUBLIC_NVIDIA_BUILD_BASE_URL
    elif provider == "anthropic":
        base_url_env = "ANTHROPIC_BASE_URL"
        # Name the official root explicitly so the SDK never consults ambient endpoint settings.
        base_url = _ANTHROPIC_API_ROOT
        if configured := os.environ.get(base_url_env, "").strip():
            try:
                base_url = _normalize_anthropic_base_url(configured, variable=base_url_env)
            except ProviderConfigurationError as exc:
                return None, str(exc)
    else:
        base_url_env = "OPENAI_BASE_URL" if provider == "openai" else "SKILL_EVAL_LLM_BASE_URL"
        configured = os.environ.get(base_url_env, "").strip().rstrip("/")
        if provider == "openai-compatible" and not configured:
            return None, f"No base URL configured for openai-compatible judge panel member ({base_url_env})"
        base_url = configured or OPENAI_BASE_URL
        try:
            parsed = urlparse(base_url)
        except ValueError:
            parsed = None
        if parsed is None or parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return None, f"{base_url_env} must be an absolute HTTP or HTTPS URL for the {provider} judge panel member"

    config = ProviderConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        litellm_model=f"anthropic/{model}" if provider == "anthropic" else f"openai/{model}",
        credential_env=key_env,
        base_url_env=base_url_env,
    )
    return config, None


def call_public_llm(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: int = 60,
    allow_model_fallback: bool = True,
    response_schema: dict[str, Any] | None = None,
    schema_name: str = "judge_response",
    target: JudgeTarget | None = None,
) -> tuple[str | None, str | None]:
    """Call the configured public provider through the shared client.

    The shared client is responsible for OpenAI-compatible endpoints, Anthropic,
    and Bedrock. Keeping this judge on that path prevents provider behavior from
    drifting between dataset generation, Tier 1, and Tier 3.

    A judge panel ``target`` (passed here or set in ``_ACTIVE_JUDGE_TARGET``)
    fixes the provider and model and uses only that member's own credentials;
    ``model``, ``api_key``, and model overrides do not apply to it. Without a
    target the configured provider is used exactly as before.
    """
    _ = timeout, allow_model_fallback
    target = target if target is not None else _ACTIVE_JUDGE_TARGET.get()
    member_api_key = None
    try:
        from skillevaluator.inference.client import LLMClient

        if target is None:
            client = LLMClient(
                model=model,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            provider_config, config_error = _judge_target_provider_config(target)
            if provider_config is None:
                return None, config_error
            member_api_key = provider_config.api_key
            client = LLMClient(max_tokens=max_tokens, temperature=temperature, provider_config=provider_config)
        return (
            client.completions(
                "You are a precise evaluation judge.",
                prompt,
                response_schema=response_schema,
                schema_name=schema_name,
            ),
            None,
        )
    except EmptyLLMResponseError:
        return "", None
    except Exception as exc:
        detail = f"Public provider call failed: {exc}"
        return None, _redact_configured_credentials(detail, (api_key, member_api_key))


_JSON_WHITESPACE = " \t\r\n"
_MAX_JSON_TEXT_CHARS = 100_000
_MAX_JSON_NESTING = 128
_JSON_NUMBER_PREFIX_RE = re.compile(r"-?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]*)?|(?:0|[1-9][0-9]*)\.)?")


def _balanced_json_container_end(text: str, start: int) -> int | None:
    """Return the exclusive end of one bounded structural container."""
    if start >= len(text) or text[start] not in "{[":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
            if len(stack) > _MAX_JSON_NESTING:
                return None
        elif ch in "}]":
            expected = "{" if ch == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                return i + 1
    return None


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting ambiguous duplicate members."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object member")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError("Non-standard JSON constant")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number overflowed to a non-finite value")
    return parsed


def _json_nesting_within_limit(text: str) -> bool:
    """Bound structural nesting without recursively parsing partial JSON."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                return False
        elif character in "}]" and depth:
            depth -= 1
    return True


def _extract_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a JSON payload from LLM response text.

    Tolerates markdown fences and prose around exactly one valid bounded JSON
    container. Multiple complete documents are ambiguous, and an unfinished
    earlier structural segment blocks promotion of a nested object. Top-level
    arrays parse through unchanged (``harbor.report`` relies on that); judge
    callers must dict-check the result themselves.
    """
    text = (text or "").strip()
    if not text or len(text) > _MAX_JSON_TEXT_CHARS:
        return None

    documents: list[dict[str, Any] | list[Any]] = []
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        end = _balanced_json_container_end(text, index)
        if end is None:
            return documents[0] if documents else None
        candidate = text[index:end]
        try:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_reject_duplicate_object_pairs,
                parse_constant=_reject_nonstandard_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError):
            index = end
            continue
        if isinstance(parsed, (dict, list)):
            documents.append(parsed)
            if len(documents) > 1:
                return None
        index = end
    return documents[0] if documents else None


def _is_json_string_prefix(text: str) -> bool:
    """Return whether an unfinished bounded string can be completed as JSON."""
    if not text.startswith('"'):
        return False
    index = 1
    while index < len(text):
        character = text[index]
        if ord(character) < 0x20 or character == '"':
            return False
        if character != "\\":
            index += 1
            continue
        index += 1
        if index >= len(text):
            return True
        escape = text[index]
        if escape == "u":
            for offset in range(1, 5):
                if index + offset >= len(text):
                    return True
                if text[index + offset] not in "0123456789abcdefABCDEF":
                    return False
            index += 5
        elif escape in '"\\/bfnrt':
            index += 1
        else:
            return False
    return True


def _is_json_scalar_prefix(text: str) -> bool:
    if not text:
        return True
    if text.startswith('"'):
        return _is_json_string_prefix(text)
    literals = {"t": "true", "f": "false", "n": "null"}
    if text[0] in literals:
        return literals[text[0]].startswith(text)
    if text[0] == "-" or text[0] in "0123456789":
        return _JSON_NUMBER_PREFIX_RE.fullmatch(text) is not None
    return False


def _is_append_only_json_object_prefix(fragment: str) -> bool:
    """Validate an unfinished flat result entry using bounded decoder steps."""
    if not fragment or len(fragment) > _MAX_JSON_TEXT_CHARS:
        return False

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index: int) -> int:
        while index < len(fragment) and fragment[index] in _JSON_WHITESPACE:
            index += 1
        return index

    index = _skip_whitespace(0)
    if index >= len(fragment) or fragment[index] != "{":
        return False
    index += 1
    keys: set[str] = set()
    while True:
        index = _skip_whitespace(index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        try:
            key, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_string_prefix(fragment[index:])
        if not isinstance(key, str) or key in keys:
            return False
        keys.add(key)
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] != ":":
            return False
        index = _skip_whitespace(index + 1)
        if index >= len(fragment):
            return True
        value_start = index
        try:
            value, next_index = decoder.raw_decode(fragment, index)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _is_json_scalar_prefix(fragment[value_start:])
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and next_index < len(fragment)
            and fragment[next_index] in ".eE"
        ):
            return _JSON_NUMBER_PREFIX_RE.fullmatch(fragment[value_start:]) is not None
        index = _skip_whitespace(next_index)
        if index >= len(fragment):
            return True
        if fragment[index] == "}":
            return False
        if fragment[index] != ",":
            return False
        index += 1


def _salvage_behavior_results(text: str) -> list[dict[str, Any]]:
    """Recover complete per-behavior entries from a truncated ``results`` array.

    Reasoning judges that hit the output-token cap emit ``{"results": [...`` and
    stop mid-entry (``finish_reason="length"``); every fully-formed ``{...}``
    entry before the cut is still valid JSON and can be scored.
    """
    text = text or ""
    if len(text) > _MAX_JSON_TEXT_CHARS or not _json_nesting_within_limit(text):
        return []
    object_start = text.find("{")
    if object_start == -1:
        return []
    if any(character in "[]{}" for character in text[:object_start]):
        return []

    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_parse_finite_json_float,
    )

    def _skip_whitespace(index: int) -> int:
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    # Parse only complete top-level fields preceding ``results``. This rejects
    # nested/unrelated arrays and lets us validate a score emitted before the
    # array without requiring the outer object itself to be complete.
    i = object_start + 1
    array_start = None
    seen_keys: set[str] = set()
    while i < len(text):
        i = _skip_whitespace(i)
        try:
            key, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if not isinstance(key, str):
            return []
        if key in seen_keys:
            return []
        seen_keys.add(key)
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ":":
            return []
        i = _skip_whitespace(i + 1)
        if key == "results":
            if i >= len(text) or text[i] != "[":
                return []
            array_start = i
            break
        try:
            value, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return []
        if key == "score" and _finite_score(value) is None:
            return []
        i = _skip_whitespace(i)
        if i >= len(text) or text[i] != ",":
            return []
        i += 1

    if array_start is None:
        return []

    results: list[dict[str, Any]] = []
    i = array_start + 1
    while True:
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        if text[i] != "{":
            return []
        try:
            entry, i = decoder.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return results if _is_append_only_json_object_prefix(text[i:]) else []
        if not isinstance(entry, dict):
            return []
        results.append(entry)
        i = _skip_whitespace(i)
        if i >= len(text):
            return results
        # Salvage is only for an array truncated before its closing bracket.
        # A closed results array with a malformed outer object is not partial
        # per-entry output and must take the structured-error path.
        if text[i] == "]":
            return []
        if text[i] != ",":
            return []
        i = _skip_whitespace(i + 1)
        if i >= len(text):
            return results
        if text[i] == "]":
            return []


STRUCTURED_JUDGE_MAX_TOKENS = 4096

_JUDGE_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed or validated. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and keep explanations brief."
)


def _call_validated_json_judge(
    prompt: str,
    validate: Any,
    call: Any,
    extract: Any,
    **call_kwargs: Any,
) -> tuple[Any, str | None, dict[str, Any]]:
    """Invoke a JSON judge with one format-correction retry when payload validation fails."""
    call_kwargs.setdefault("max_tokens", STRUCTURED_JUDGE_MAX_TOKENS)

    def invoke(call_prompt: str) -> tuple[Any, str | None, dict[str, Any], str | None]:
        content, error, *metadata = call(call_prompt, **call_kwargs)
        provenance = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        parsed = extract(content) if content else None
        validation_error = validate(parsed) if not error else None
        return parsed, error, provenance, validation_error

    parsed, error, provenance, validation_error = invoke(prompt)
    if error:
        return None, f"LLM judge error: {error}", provenance
    if validation_error is None:
        return parsed, None, provenance

    parsed, error, provenance, validation_error = invoke(prompt + _JUDGE_RETRY_REMINDER)
    if error:
        return None, f"LLM judge retry error: {error}", provenance
    if validation_error is not None:
        return None, f"{validation_error} after retry", provenance
    return parsed, None, provenance


# ---------------------------------------------------------------------------
# Accuracy judge (5-criterion)
# ---------------------------------------------------------------------------

ACCURACY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "object",
            "properties": {
                "SKILL_IDENTIFIED": {"type": "boolean"},
                "ACTION_CORRECT": {"type": "boolean"},
                "FACTUALLY_ACCURATE": {"type": "boolean"},
                "TASK_ADDRESSED": {"type": "boolean"},
                "ACTIONABLE": {"type": "boolean"},
            },
            "required": [
                "SKILL_IDENTIFIED",
                "ACTION_CORRECT",
                "FACTUALLY_ACCURATE",
                "TASK_ADDRESSED",
                "ACTIONABLE",
            ],
            "additionalProperties": False,
        },
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["criteria", "score", "reason"],
    "additionalProperties": False,
}

GOAL_ACCURACY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_goal": {"type": "string"},
        "end_state": {"type": "string"},
        "achieved": {"type": "boolean"},
        "score": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["user_goal", "end_state", "achieved", "score", "reason"],
    "additionalProperties": False,
}

BEHAVIOR_CHECK_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["step", "passed", "reason"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["results", "score", "summary"],
    "additionalProperties": False,
}

ACCURACY_PROMPT = """You are an expert evaluator for AI agent responses. Evaluate by checking \
each criterion below against the expected answer. For each criterion, determine true (satisfied) or false (not satisfied).

1. SKILL_IDENTIFIED: Does the response reference or use the correct skill for the task?
2. ACTION_CORRECT: Does the response describe or execute the correct actions/scripts?
3. FACTUALLY_ACCURATE: Are the factual claims consistent with the expected answer?
4. TASK_ADDRESSED: Does the response directly address the user's request?
5. ACTIONABLE: Does the response provide actionable information (not just acknowledgment)?

Compute score = count(true) / 5.
Be lenient on exact wording but strict on factual correctness.

Respond with ONLY a JSON object:
{{"criteria": {{"SKILL_IDENTIFIED": true/false, "ACTION_CORRECT": true/false, \
"FACTUALLY_ACCURATE": true/false, "TASK_ADDRESSED": true/false, "ACTIONABLE": true/false}}, \
"score": <float>, "reason": "<brief summary>"}}

USER QUESTION:
{question}

EXPECTED ANSWER:
{ground_truth}

SELECTED EVIDENCE (final response + produced artifacts; low-relevance steps may be omitted):
{agent_text}"""

_ACCURACY_CRITERIA_KEYS = frozenset(
    {
        "SKILL_IDENTIFIED",
        "ACTION_CORRECT",
        "FACTUALLY_ACCURATE",
        "TASK_ADDRESSED",
        "ACTIONABLE",
    }
)


def _valid_accuracy_criteria(value: Any) -> bool:
    """Return True when value is a complete 5-criterion boolean mapping."""
    return (
        isinstance(value, dict)
        and value.keys() == _ACCURACY_CRITERIA_KEYS
        and all(isinstance(item, bool) for item in value.values())
    )


def _accuracy_payload_error(parsed: Any) -> str | None:
    """Validate a parsed accuracy judge payload and return an error message if malformed."""
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    if "reason" in parsed and not isinstance(parsed["reason"], str):
        return "Judge response contained an invalid accuracy reason"
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)
    if "criteria" in parsed and not criteria_valid:
        return "Judge response contained invalid accuracy criteria"
    if _finite_score(parsed.get("score")) is None and not criteria_valid:
        return "Judge response contained no valid accuracy score or complete criteria"
    return None


def judge_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the 5-criterion accuracy judge. Returns ``{"score": float, "reason": str, ...}``."""
    if not ground_truth:
        return {"score": 1.0, "reason": "No ground_truth -- skipped"}

    prompt = ACCURACY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        agent_text=agent_text,
    )
    kwargs.setdefault("response_schema", ACCURACY_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "accuracy_judgment")

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _accuracy_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    assert isinstance(parsed, dict)
    criteria = parsed.get("criteria")
    criteria_valid = _valid_accuracy_criteria(criteria)

    score = _finite_score(parsed.get("score"))
    if score is None:
        assert criteria_valid
        yes_count = sum(1 for v in criteria.values() if v is True)
        score = yes_count / 5.0

    return {
        "score": round(score, 4),
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "criteria": criteria if criteria_valid else {},
    }


# ---------------------------------------------------------------------------
# Goal accuracy judge
# ---------------------------------------------------------------------------

GOAL_ACCURACY_PROMPT = """You are an evaluation judge. Determine whether an AI agent achieved \
the expected goal by analyzing the full conversation.

Step 1: What was the user's goal?
Step 2: What end state did the agent reach?
Step 3: Compare the end state to the expected outcome.

USER REQUEST:
{question}

EXPECTED OUTCOME (ground truth):
{ground_truth}

AGENT'S TOOL CALLS:
{tool_summary}

END-STATE EVIDENCE:
{agent_text}

Did the agent achieve the expected goal?

Respond with ONLY a JSON object:
{{"user_goal": "<inferred goal>", "end_state": "<what agent achieved>", \
"achieved": true/false, "score": 1.0 or 0.0, "reason": "<brief explanation>"}}"""


def _goal_payload_error(parsed: Any) -> str | None:
    """Validate a parsed goal-accuracy judge payload and return an error message if malformed."""
    if not isinstance(parsed, dict):
        return "Judge response was not a valid JSON object"
    for field in ("reason", "user_goal", "end_state"):
        if field in parsed and not isinstance(parsed[field], str):
            return f"Judge response contained an invalid {field} value"
    if not isinstance(parsed.get("achieved"), bool):
        return "Judge response contained an invalid achieved value"
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return "Judge response contained an invalid goal score"
    return None


def judge_goal_accuracy(
    question: str,
    ground_truth: str,
    agent_text: str,
    tool_summary: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the goal accuracy judge (two-step: infer goal, compare outcome)."""
    if not ground_truth:
        return {"score": 1.0, "reason": "No ground_truth -- skipped"}

    prompt = GOAL_ACCURACY_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        tool_summary=tool_summary,
        agent_text=agent_text,
    )
    kwargs.setdefault("response_schema", GOAL_ACCURACY_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "goal_accuracy_judgment")

    parsed, error, _provenance = _call_validated_json_judge(
        prompt,
        _goal_payload_error,
        call_public_llm,
        _extract_json,
        **kwargs,
    )
    if error:
        return _judge_error(error)

    assert isinstance(parsed, dict)
    achieved = parsed.get("achieved")
    assert isinstance(achieved, bool)

    score = 1.0 if achieved else 0.0
    if "score" in parsed:
        score = _finite_score(parsed["score"])
        assert score is not None

    result = {
        "score": score,
        "reason": _bounded_judge_text(parsed.get("reason", "")),
        "user_goal": _bounded_judge_text(parsed.get("user_goal", "")),
        "end_state": _bounded_judge_text(parsed.get("end_state", "")),
    }
    if kwargs.get("target") is not None or _ACTIVE_JUDGE_TARGET.get() is not None:
        # Panel members vote on ``achieved``; single-judge results keep their shape.
        result["achieved"] = achieved
    return result


# ---------------------------------------------------------------------------
# Behavior check judge
# ---------------------------------------------------------------------------

BEHAVIOR_CHECK_PROMPT = """You are evaluating whether an AI agent followed expected behaviors \
during a task. Analyze the full conversation and determine if each expected behavior was \
observed.

CONVERSATION:
{conversation}

EXPECTED BEHAVIORS:
{behaviors}

For each behavior, set "passed" to true (observed) or false (not observed) with a brief reason.

Respond with ONLY a JSON object:
{{"results": [{{"step": 1, "passed": true/false, "reason": "..."}}, ...], \
"score": <float between 0.0 and 1.0>, "summary": "<brief summary>"}}"""


def _slice_with_middle_marker(
    text: str,
    budget: int,
    marker: str,
    *,
    max_head: int | None = None,
    fallback_tail: bool = False,
) -> str:
    """Compact *text* to fit *budget* by replacing the middle with *marker*."""
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text
    if budget <= len(marker) + 2:
        return text[-budget:] if fallback_tail else text[:budget]
    avail = budget - len(marker)
    head = max(1, avail // 2 if max_head is None else min(max_head, avail // 2))
    tail = max(1, avail - head)
    return f"{text[:head]}{marker}{text[-tail:]}"


def _compact_behavior_conversation(conversation_text: str, limit: int | None = None) -> str:
    """Keep both setup context and late outcome evidence in behavior prompts."""
    if limit is None:
        limit = _behavior_check_budget()
    if len(conversation_text) <= limit:
        return conversation_text

    marker = "\n...[middle truncated for behavior check]...\n"
    if limit <= len(marker) + 1:
        return conversation_text[:limit]

    final_limit = _behavior_final_response_limit()
    final_header = f"{_SECTION_FINAL_RESPONSE}\n"
    final_idx = -1
    if conversation_text.startswith(final_header):
        final_idx = 0
    else:
        pos = conversation_text.find(f"\n\n{final_header}")
        if pos != -1:
            final_idx = pos + 2

    if final_idx != -1:
        final_end = len(conversation_text)
        for next_hdr in (f"\n\n{_SECTION_USER_REQUEST}\n", f"\n\n{_SECTION_COMPACT_TOOL_HISTORY}\n"):
            pos = conversation_text.find(next_hdr, final_idx)
            if pos != -1 and pos < final_end:
                final_end = pos
        prefix = conversation_text[:final_idx]
        final_sec = conversation_text[final_idx:final_end]
        suffix = conversation_text[final_end:]

        reserved_other = min(1600, max(0, limit - final_limit), limit // 2)
        max_final = min(final_limit, max(1, limit - reserved_other))
        if len(final_sec) > max_final:
            final_body = final_sec[len(final_header) :]
            body_limit = max(1, max_final - len(final_header))
            final_sec = f"{final_header}{_truncate_for_behavior(final_body, body_limit)}"[:max_final]

        rem = limit - len(final_sec)
        if rem <= 0:
            return final_sec[:limit]
        if len(prefix) + len(suffix) <= rem:
            return f"{prefix}{final_sec}{suffix}"
        if not suffix:
            pre_comp = _slice_with_middle_marker(prefix, rem, marker)
            return f"{pre_comp}{final_sec}"[:limit]
        if len(prefix) <= rem // 2:
            suf_comp = _slice_with_middle_marker(suffix, rem - len(prefix), marker, max_head=800, fallback_tail=True)
            return f"{prefix}{final_sec}{suf_comp}"[:limit]
        pre_budget = max(1, min(len(prefix), rem // 2))
        suf_budget = max(0, rem - pre_budget)
        pre_comp = _slice_with_middle_marker(prefix, pre_budget, marker)
        suf_comp = _slice_with_middle_marker(suffix, suf_budget, marker, max_head=800, fallback_tail=True)
        return f"{pre_comp}{final_sec}{suf_comp}"[:limit]

    available = limit - len(marker)
    reserved_head = min(1600, available // 2)
    tail = max(1, available // 3, min(final_limit, max(1, available - max(1, reserved_head))))
    head = max(1, available - tail)
    return f"{conversation_text[:head]}{marker}{conversation_text[-tail:]}"


# Reasoning judges (e.g. openai/openai/gpt-5*) spend completion budget on hidden
# reasoning tokens before emitting the per-behavior results array; the old 1024
# cap was observed live to truncate behavior_check output to EMPTY content
# (finish_reason="length", reasoning_tokens=1024).
BEHAVIOR_JUDGE_MAX_TOKENS = STRUCTURED_JUDGE_MAX_TOKENS

_BEHAVIOR_RETRY_REMINDER = (
    "\n\nIMPORTANT: Your previous reply could not be parsed. Respond with ONLY the "
    "minified JSON object on a single line -- no markdown fences, no prose, and "
    'keep every "reason" under 15 words.'
)


def judge_behavior_check(
    conversation_text: str,
    expected_behaviors: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the behavior check LLM judge."""
    if not expected_behaviors:
        return {"score": 1.0, "reason": "No expected_behavior defined", "results": []}

    behaviors_text = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(expected_behaviors))

    prompt = BEHAVIOR_CHECK_PROMPT.format(
        conversation=_compact_behavior_conversation(conversation_text),
        behaviors=behaviors_text,
    )
    kwargs.setdefault("max_tokens", BEHAVIOR_JUDGE_MAX_TOKENS)
    kwargs.setdefault("response_schema", BEHAVIOR_CHECK_JSON_SCHEMA)
    kwargs.setdefault("schema_name", "behavior_check_judgment")

    content, error = call_public_llm(prompt, **kwargs)
    if error:
        return _judge_error(f"LLM judge error: {error}", results=[])

    def _parse_judge_object(text: str | None) -> dict[str, Any] | list[Any] | None:
        """Parse a JSON object or list from judge response text."""
        return _extract_json(text) if text else None

    parsed = _parse_judge_object(content)
    score = _behavior_payload_score(parsed, len(expected_behaviors))
    attempts = [(content or "", parsed)]
    retry_error = None
    if score is None:
        # One retry max, with an explicit machine-readable-output reminder.
        retry_content, retry_error = call_public_llm(prompt + _BEHAVIOR_RETRY_REMINDER, **kwargs)
        if not retry_error:
            parsed = _parse_judge_object(retry_content)
            attempts.append((retry_content or "", parsed))
            score = _behavior_payload_score(parsed, len(expected_behaviors))

    if score is None:
        # Salvage complete entries from a truncated results array (newest first).
        for text, extracted in reversed(attempts):
            if extracted is not None:
                continue
            salvaged = _salvage_behavior_results(text)
            if salvaged:
                candidate = {
                    "results": salvaged,
                    "summary": (
                        f"Salvaged {len(salvaged)}/{len(expected_behaviors)} behavior "
                        "results from truncated judge response"
                    ),
                }
                candidate_score = _behavior_payload_score(
                    candidate,
                    len(expected_behaviors),
                    allow_partial=True,
                )
                if candidate_score is not None:
                    parsed = candidate
                    score = candidate_score
                    break

    if score is None:
        if retry_error:
            return _judge_error(f"LLM judge retry error: {retry_error}", results=[])
        return _judge_error("Judge response was unparseable or invalid after retry", results=[])

    assert isinstance(parsed, dict)
    results = parsed["results"]

    return {
        "score": round(score, 4),
        "reason": parsed.get("summary", ""),
        "results": results,
    }


def _behavior_payload_score(
    parsed: dict[str, Any] | list[Any] | None,
    expected_count: int,
    *,
    allow_partial: bool = False,
) -> float | None:
    if not isinstance(parsed, dict):
        return None
    results = parsed.get("results")
    if not isinstance(results, list):
        return None
    if any(not isinstance(result, dict) or not isinstance(result.get("passed"), bool) for result in results):
        return None
    if allow_partial:
        if not results or len(results) > expected_count:
            return None
    elif len(results) != expected_count:
        return None
    if "score" in parsed and _finite_score(parsed["score"]) is None:
        return None
    denominator = expected_count if allow_partial else len(results)
    if denominator <= 0:
        return None
    return sum(1 for result in results if result["passed"]) / denominator
