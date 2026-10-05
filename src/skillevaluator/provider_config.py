# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LLM and embedding provider configuration."""

from __future__ import annotations

import ipaddress
import math
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote_to_bytes, urlsplit, urlunsplit

import idna

PUBLIC_NVIDIA_BUILD_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
_PROVIDER_SETUP_URL = "https://docs.nvidia.com/skills/skillevaluator/configuration"

# Pinned frontier chat defaults (not floating aliases like ``gpt-5`` / ``claude-opus-latest``).
# Harbor ``templates/eval.py`` cannot import this module — keep its local
# ``DEFAULT_JUDGE_MODEL`` in sync via the drift test in
# ``tests/tier3/test_judge_parse_robustness.py``.
CHAT_DEFAULT_OPENAI = "gpt-5.6-sol"
CHAT_DEFAULT_ANTHROPIC = "claude-opus-5"
CHAT_DEFAULT_BEDROCK = "us.anthropic.claude-opus-5"
CHAT_DEFAULT_NVIDIA = "nvidia/nemotron-3-super-120b-a12b"
# Gateway catalog IDs are independent of native-provider model names. Operators
# can override these defaults without changing the configured endpoint or key.
CHAT_DEFAULT_GATEWAY = "nvidia/nvidia/nemotron-3-super-120b-long-ctx"
# Lower-cost OpenAI alternative for ``SKILL_EVAL_LLM_MODEL`` overrides.
CHAT_CHEAP_OPENAI = "gpt-5.4-mini"

CHAT_DEFAULT_MODELS = {
    "openai": CHAT_DEFAULT_OPENAI,
    "anthropic": CHAT_DEFAULT_ANTHROPIC,
    "nv_build": CHAT_DEFAULT_NVIDIA,
    "bedrock": CHAT_DEFAULT_BEDROCK,
    "openai-compatible": CHAT_DEFAULT_GATEWAY,
}
# Agent harnesses have separate model defaults with their required capabilities.
GATEWAY_AGENT_DEFAULT_MODELS = {
    "codex": "openai/openai/gpt-5.6-sol",
    "claude-code": "aws/anthropic/bedrock-claude-opus-5",
    "opencode": CHAT_DEFAULT_GATEWAY,
}
EMBEDDING_DEFAULT_NVIDIA = "nvidia/nemotron-3-embed-1b"
EMBEDDING_DEFAULT_GATEWAY = "nvidia/nvidia/nemotron-3-embed-1b"
_EMBEDDING_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "nv_build": EMBEDDING_DEFAULT_NVIDIA,
    "openai-compatible": EMBEDDING_DEFAULT_GATEWAY,
}
_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "nv_build", "bedrock", "openai-compatible"})
_LITELLM_PREFIXES = {
    "openai": "openai",
    "anthropic": "anthropic",
    "nv_build": "openai",
    "bedrock": "bedrock",
    "openai-compatible": "openai",
}

# Cross-model judge panel for the three Tier 3 LLM metrics. The panel is read
# only from the operator's host environment (or ``--judge-panel``), never from
# skill-owned configuration, because a skill must not pick its own judges.
# ``harbor/templates/eval.py`` parses the normalized values with the same rules.
JUDGE_PANEL_ENV = "SKILL_EVAL_JUDGE_PANEL"
JUDGE_PANEL_AGGREGATION_ENV = "SKILL_EVAL_JUDGE_PANEL_AGGREGATION"
JUDGE_PANEL_QUORUM_ENV = "SKILL_EVAL_JUDGE_PANEL_QUORUM"
JUDGE_PANEL_DISAGREEMENT_ENV = "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT"
JUDGE_PANEL_ENV_VARS = frozenset(
    {JUDGE_PANEL_ENV, JUDGE_PANEL_AGGREGATION_ENV, JUDGE_PANEL_QUORUM_ENV, JUDGE_PANEL_DISAGREEMENT_ENV}
)
JUDGE_PANEL_AGGREGATIONS = ("vote", "median", "mean")
JUDGE_PANEL_MAX_MEMBERS = 5
DEFAULT_JUDGE_PANEL_DISAGREEMENT = 0.4
_JUDGE_PANEL_KNOB_ENV_VARS = (JUDGE_PANEL_AGGREGATION_ENV, JUDGE_PANEL_QUORUM_ENV, JUDGE_PANEL_DISAGREEMENT_ENV)
_JUDGE_MODEL_OVERRIDE_ENV_VARS = ("LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL")
_JUDGE_PANEL_EXAMPLE = "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b"
_ANTHROPIC_DNS_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_ANTHROPIC_INTERNAL_LABEL_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$")
_ANTHROPIC_IPV6_ZONE_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_ANTHROPIC_PATH_SAFE = "/:@!$&'()*+,;=-._~%"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_HEX_DIGIT_BYTES = frozenset(b"0123456789abcdefABCDEF")
_UNRESERVED_BYTES = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_NO_CUSTOM_TEMPERATURE_MODEL_IDS = frozenset({"claude-mythos-preview"})
_ANTHROPIC_BEDROCK_PREFIX_RE = re.compile(r"^(?:(?:[a-z]{2}|global)\.)?anthropic\.")
_VERSIONED_CLAUDE_MODEL_RE = re.compile(
    r"^claude-[a-z][a-z-]*-(?P<major>\d+)"
    r"(?:-(?P<minor>\d{1,2}))?"
    r"(?:-(?:\d{8}|latest))?"
    r"(?:-v\d+)?(?::\d+)?$"
)


def _model_leaf(model: str) -> str:
    """Return a normalized model ID for capability checks only."""
    leaf = str(model or "").strip().casefold().rsplit("/", 1)[-1]
    return _ANTHROPIC_BEDROCK_PREFIX_RE.sub("", leaf, count=1)


def _supports_custom_temperature(model: str) -> bool:
    """Return whether ``model`` accepts a non-default temperature value."""
    leaf = _model_leaf(model)
    if leaf.startswith("gpt-5") or leaf in _NO_CUSTOM_TEMPERATURE_MODEL_IDS:
        return False

    match = _VERSIONED_CLAUDE_MODEL_RE.fullmatch(leaf)
    if match is None:
        return True
    version = (int(match.group("major")), int(match.group("minor") or 0))
    return version < (4, 7)


class ProviderConfigurationError(ValueError):
    """Raised when a selected public provider is not fully configured."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider values safe to pass to the relevant SDK."""

    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    litellm_model: str
    region: str | None = None
    credential_env: str | None = None
    base_url_env: str | None = None

    def child_environment(self) -> dict[str, str]:
        """Return this provider's public credential settings for a child process."""
        environment: dict[str, str] = {}
        if self.credential_env and self.api_key:
            environment[self.credential_env] = self.api_key

        if self.base_url_env and self.base_url:
            environment[self.base_url_env] = self.base_url
        elif self.provider == "openai" and self.base_url:
            environment["OPENAI_BASE_URL"] = self.base_url
        elif self.provider == "anthropic" and self.base_url:
            environment["ANTHROPIC_BASE_URL"] = self.base_url
        elif self.provider == "openai-compatible" and self.base_url:
            environment["SKILL_EVAL_LLM_BASE_URL"] = self.base_url
        elif self.provider == "bedrock" and self.region:
            environment["AWS_REGION"] = self.region

        return environment


@dataclass(frozen=True)
class JudgeTarget:
    """One resolved judge-panel member with its own credential and endpoint."""

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    credential_env: str | None = None
    base_url_env: str | None = None
    region: str | None = None

    @property
    def label(self) -> str:
        """Return the normalized ``provider:model`` identity used in config and reports."""
        return f"{self.provider}:{self.model}"

    def provider_config(self) -> ProviderConfig:
        """Return the equivalent provider config, for example for model-catalog probing."""
        return ProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            litellm_model=f"{_LITELLM_PREFIXES[self.provider]}/{self.model}",
            region=self.region,
            credential_env=self.credential_env,
            base_url_env=self.base_url_env,
        )

    def verifier_environment(self) -> dict[str, str]:
        """Return the canonical in-verifier variables this member's judge reads."""
        if self.provider == "openai":
            # Always pin the resolved endpoint, including the official default.
            environment = {"OPENAI_API_KEY": self.api_key, "OPENAI_BASE_URL": self.base_url}
        elif self.provider == "anthropic":
            environment = {"ANTHROPIC_API_KEY": self.api_key, "ANTHROPIC_BASE_URL": self.base_url}
        elif self.provider == "nv_build":
            environment = {"NVIDIA_API_KEY": self.api_key}
        elif self.provider == "bedrock":
            environment = {"AWS_REGION": self.region}
        else:
            environment = {"SKILL_EVAL_LLM_API_KEY": self.api_key, "SKILL_EVAL_LLM_BASE_URL": self.base_url}
        return {name: value for name, value in environment.items() if value}


@dataclass(frozen=True)
class JudgePanelConfig:
    """Validated cross-model judge panel for the three Tier 3 LLM metrics."""

    members: tuple[JudgeTarget, ...]
    aggregation: str
    quorum: int
    disagreement_threshold: float
    warnings: tuple[str, ...] = ()

    def env_value(self) -> str:
        """Return the normalized ``provider:model,...`` panel the verifier parses."""
        return ",".join(member.label for member in self.members)

    def verifier_settings(self) -> dict[str, str]:
        """Return all four panel settings as normalized strings for the verifier."""
        return {
            JUDGE_PANEL_ENV: self.env_value(),
            JUDGE_PANEL_AGGREGATION_ENV: self.aggregation,
            JUDGE_PANEL_QUORUM_ENV: str(self.quorum),
            JUDGE_PANEL_DISAGREEMENT_ENV: str(self.disagreement_threshold),
        }

    def redacted(self) -> dict[str, Any]:
        """Return a JSON-safe description without credentials or endpoints."""
        return {
            "panel": [
                {"provider": member.provider, "model": member.model, "label": member.label} for member in self.members
            ],
            "aggregation": self.aggregation,
            "quorum": self.quorum,
            "disagreement_threshold": self.disagreement_threshold,
            "warnings": list(self.warnings),
        }


def resolve_llm_provider(environ: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve the public provider used for LLM-backed checks and judging."""
    env = _environment(environ)
    provider = _selected_provider(env, "SKILL_EVAL_LLM_PROVIDER")
    _validate_provider(provider, variable="SKILL_EVAL_LLM_PROVIDER")
    configured_model = env.get("SKILL_EVAL_LLM_MODEL")
    if configured_model is None:
        model = CHAT_DEFAULT_MODELS[provider]
    else:
        model = configured_model.strip()
        if not model:
            raise ProviderConfigurationError("SKILL_EVAL_LLM_MODEL must be a non-empty string when set.")

    if provider == "openai":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "OPENAI_API_KEY"),
            base_url=(env.get("SKILL_EVAL_LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/"),
            litellm_model=f"openai/{model}",
            credential_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )
    if provider == "anthropic":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "ANTHROPIC_API_KEY"),
            base_url=_anthropic_base_url(env),
            litellm_model=f"anthropic/{model}",
            credential_env="ANTHROPIC_API_KEY",
            base_url_env="ANTHROPIC_BASE_URL",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
            litellm_model=f"openai/{model}",
            credential_env="NVIDIA_API_KEY",
        )
    if provider == "bedrock":
        return ProviderConfig(
            provider=provider,
            model=model,
            api_key=None,
            base_url=None,
            litellm_model=f"bedrock/{model}",
            region=env.get("AWS_REGION") or "us-west-2",
        )

    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=_required(env, "SKILL_EVAL_LLM_API_KEY"),
        base_url=_required(env, "SKILL_EVAL_LLM_BASE_URL").rstrip("/"),
        litellm_model=f"openai/{model}",
        credential_env="SKILL_EVAL_LLM_API_KEY",
        base_url_env="SKILL_EVAL_LLM_BASE_URL",
    )


def resolve_embedding_provider(environ: Mapping[str, str] | None = None) -> ProviderConfig:
    """Resolve the embedding provider used by Tier 2 semantic overlap checks."""
    env = _environment(environ)
    provider = (
        env.get("SKILL_EVAL_EMBEDDING_PROVIDER")
        or env.get("SKILL_EVAL_LLM_PROVIDER")
        or _selected_provider(env, "SKILL_EVAL_EMBEDDING_PROVIDER")
    ).lower()
    if provider in {"anthropic", "bedrock"}:
        raise ProviderConfigurationError(
            f"SKILL_EVAL_EMBEDDING_PROVIDER is required because {provider} does not provide embeddings. "
            "Set SKILL_EVAL_EMBEDDING_PROVIDER=nv_build|openai|openai-compatible (NVIDIA_API_KEY or "
            "OPENAI_API_KEY supply the first two)."
        )
    _validate_provider(provider, variable="SKILL_EVAL_EMBEDDING_PROVIDER")

    if provider == "openai":
        return ProviderConfig(
            provider=provider,
            model=env.get("SKILL_EVAL_EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS[provider],
            api_key=_required(env, "OPENAI_API_KEY"),
            base_url=(env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or env.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip(
                "/"
            ),
            litellm_model=f"openai/{env.get('SKILL_EVAL_EMBEDDING_MODEL') or _EMBEDDING_DEFAULT_MODELS[provider]}",
            credential_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )
    if provider == "nv_build":
        return ProviderConfig(
            provider=provider,
            model=env.get("SKILL_EVAL_EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS[provider],
            api_key=_required(env, "NVIDIA_API_KEY"),
            base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
            litellm_model=f"openai/{env.get('SKILL_EVAL_EMBEDDING_MODEL') or _EMBEDDING_DEFAULT_MODELS[provider]}",
            credential_env="NVIDIA_API_KEY",
        )

    model = env.get("SKILL_EVAL_EMBEDDING_MODEL", _EMBEDDING_DEFAULT_MODELS[provider]).strip()
    if not model:
        raise ProviderConfigurationError("SKILL_EVAL_EMBEDDING_MODEL must be a non-empty string when set.")
    return ProviderConfig(
        provider=provider,
        model=model,
        api_key=env.get("SKILL_EVAL_EMBEDDING_API_KEY") or _required(env, "SKILL_EVAL_LLM_API_KEY"),
        base_url=(env.get("SKILL_EVAL_EMBEDDING_BASE_URL") or _required(env, "SKILL_EVAL_LLM_BASE_URL")).rstrip("/"),
        litellm_model=f"openai/{model}",
        credential_env=(
            "SKILL_EVAL_EMBEDDING_API_KEY"
            if env.get("SKILL_EVAL_EMBEDDING_API_KEY", "").strip()
            else "SKILL_EVAL_LLM_API_KEY"
        ),
        base_url_env="SKILL_EVAL_EMBEDDING_BASE_URL",
    )


def resolve_judge_panel_config(
    environ: Mapping[str, str] | None = None,
    *,
    source: str = JUDGE_PANEL_ENV,
) -> JudgePanelConfig | None:
    """Resolve and validate the host-configured cross-model judge panel.

    Returns ``None`` when ``SKILL_EVAL_JUDGE_PANEL`` is unset or blank. Every
    member must have its own credential; a misconfigured panel is an error and
    never falls back to the single standard judge. ``source`` names the input
    that supplied the panel string (``--judge-panel`` when the CLI value
    replaced the environment variable) in panel errors; errors about the three
    tuning knobs keep their environment variable names.
    """
    env = _environment(environ)
    raw_panel = env.get(JUDGE_PANEL_ENV, "").strip()
    if not raw_panel:
        knobs = [name for name in _JUDGE_PANEL_KNOB_ENV_VARS if env.get(name, "").strip()]
        if knobs:
            raise ProviderConfigurationError(
                f"{', '.join(knobs)} configure(s) a judge panel, but {JUDGE_PANEL_ENV} is not set.\n\n"
                f"Set the panel (or pass --judge-panel), for example:\n  export {JUDGE_PANEL_ENV}='{_JUDGE_PANEL_EXAMPLE}'\n\n"
                f"or unset {', '.join(knobs)}; panel settings never fall back to a single judge."
            )
        return None

    entries = _parse_judge_panel(raw_panel, source=source)
    aggregation = env.get(JUDGE_PANEL_AGGREGATION_ENV, "").strip().lower() or "vote"
    if aggregation not in JUDGE_PANEL_AGGREGATIONS:
        raise ProviderConfigurationError(
            f"{JUDGE_PANEL_AGGREGATION_ENV} must be one of: {', '.join(JUDGE_PANEL_AGGREGATIONS)}."
        )
    quorum = _judge_panel_quorum(env.get(JUDGE_PANEL_QUORUM_ENV, "").strip(), member_count=len(entries))
    disagreement_threshold = _judge_panel_disagreement(env.get(JUDGE_PANEL_DISAGREEMENT_ENV, "").strip())
    overrides = [name for name in _JUDGE_MODEL_OVERRIDE_ENV_VARS if env.get(name, "").strip()]
    if overrides:
        raise ProviderConfigurationError(
            f"{source} cannot be combined with {', '.join(overrides)}: the panel names each judge's model "
            "explicitly; unset LLM_JUDGE_MODEL/SKILL_EVAL_JUDGE_MODEL or remove the panel."
        )

    warnings: list[str] = []
    if aggregation == "vote" and len(entries) % 2 == 0:
        warnings.append(
            f"Judge panel has an even number of members ({len(entries)}) with vote aggregation; tied criteria "
            "count as 0.5. Use an odd number of judges for decisive majority votes."
        )

    primary = _primary_llm_provider(env)
    if primary in {"openai", "anthropic"} and any(provider == "openai-compatible" for provider, _model in entries):
        # Checked before any member is resolved or probed: a probe would already
        # send the native key to the gateway.
        raise ProviderConfigurationError(
            f"{source} cannot include an openai-compatible member while the {primary} primary is selected: "
            f"SKILL_EVAL_LLM_BASE_URL would be both the gateway member's endpoint and the {primary} primary's "
            f"endpoint override, so the {primary} key would be sent to the gateway.\n\n"
            "Select the gateway as the primary (SKILL_EVAL_LLM_PROVIDER=openai-compatible) and list "
            f"{primary}:MODEL as a member, or remove the openai-compatible member."
        )
    members = tuple(
        _resolve_judge_target(env, provider, model, primary=primary, source=source) for provider, model in entries
    )
    return JudgePanelConfig(
        members=members,
        aggregation=aggregation,
        quorum=quorum,
        disagreement_threshold=disagreement_threshold,
        warnings=tuple(warnings),
    )


def resolve_judge_panel(environ: Mapping[str, str] | None = None) -> list[JudgeTarget] | None:
    """Return the resolved judge-panel members, or ``None`` when no panel is configured."""
    config = resolve_judge_panel_config(environ)
    return None if config is None else list(config.members)


def _parse_judge_panel(raw_panel: str, *, source: str = JUDGE_PANEL_ENV) -> list[tuple[str, str]]:
    """Split ``provider:model,...`` entries, splitting each on its first colon only."""
    entries: list[tuple[str, str]] = []
    for position, raw_entry in enumerate(raw_panel.split(","), start=1):
        entry = raw_entry.strip()
        if not entry:
            raise ProviderConfigurationError(
                f"{source} entry {position} is empty; use comma-separated provider:model entries, "
                f"for example {_JUDGE_PANEL_EXAMPLE}."
            )
        raw_provider, separator, raw_model = entry.partition(":")
        provider = raw_provider.strip().lower()
        model = raw_model.strip()
        if not separator or not provider or not model:
            raise ProviderConfigurationError(
                f"{source} entry {entry!r} must use provider:model form, for example openai:gpt-5.6-sol."
            )
        _validate_provider(provider, variable=f"{source} provider {provider!r}")
        if any(character.isspace() or unicodedata.category(character).startswith("C") for character in model):
            raise ProviderConfigurationError(
                f"{source} model {model!r} must not contain whitespace or control characters."
            )
        entries.append((provider, model))

    if len(entries) > JUDGE_PANEL_MAX_MEMBERS:
        raise ProviderConfigurationError(
            f"{source} supports at most {JUDGE_PANEL_MAX_MEMBERS} judges to keep cost bounded; got {len(entries)}."
        )
    seen: set[tuple[str, str]] = set()
    for provider, model in entries:
        if (provider, model) in seen:
            raise ProviderConfigurationError(f"{source} lists {provider}:{model} more than once.")
        seen.add((provider, model))
    if sum(provider == "openai-compatible" for provider, _model in entries) > 1:
        raise ProviderConfigurationError(
            f"{source} supports at most one openai-compatible member because SKILL_EVAL_LLM_BASE_URL "
            "and SKILL_EVAL_LLM_API_KEY configure a single gateway."
        )
    return entries


def _judge_panel_quorum(raw_quorum: str, *, member_count: int) -> int:
    if not raw_quorum:
        return member_count // 2 + 1
    error = f"{JUDGE_PANEL_QUORUM_ENV} must be an integer between 1 and {member_count} (the number of judges)."
    try:
        quorum = int(raw_quorum)
    except ValueError:
        raise ProviderConfigurationError(error) from None
    if not 1 <= quorum <= member_count:
        raise ProviderConfigurationError(error)
    return quorum


def _judge_panel_disagreement(raw_threshold: str) -> float:
    if not raw_threshold:
        return DEFAULT_JUDGE_PANEL_DISAGREEMENT
    error = f"{JUDGE_PANEL_DISAGREEMENT_ENV} must be a number between 0 and 1."
    try:
        threshold = float(raw_threshold)
    except ValueError:
        raise ProviderConfigurationError(error) from None
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ProviderConfigurationError(error)
    return threshold


def _primary_llm_provider(environ: Mapping[str, str]) -> str | None:
    """Return the provider ``resolve_llm_provider`` selects, or ``None`` when it is ambiguous."""
    try:
        return _selected_provider(environ, "SKILL_EVAL_LLM_PROVIDER")
    except ProviderConfigurationError:
        return None


def _resolve_judge_target(
    environ: Mapping[str, str],
    provider: str,
    model: str,
    *,
    primary: str | None,
    source: str = JUDGE_PANEL_ENV,
) -> JudgeTarget:
    """Resolve one member's own credential and endpoint.

    ``SKILL_EVAL_LLM_BASE_URL`` overrides the primary provider's endpoint only;
    a panel that would also need it for an ``openai-compatible`` member is
    rejected before members are resolved.
    """
    label = f"{provider}:{model}"
    if provider == "openai":
        primary_base_url = environ.get("SKILL_EVAL_LLM_BASE_URL") if primary == "openai" else None
        return JudgeTarget(
            provider=provider,
            model=model,
            api_key=_required_judge_credential(environ, "OPENAI_API_KEY", label, source=source),
            base_url=(primary_base_url or environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/"),
            credential_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
        )
    if provider == "anthropic":
        if primary == "anthropic":
            base_url = _anthropic_base_url(environ)
        elif configured_base_url := environ.get("ANTHROPIC_BASE_URL"):
            base_url = _normalize_anthropic_base_url(configured_base_url, variable="ANTHROPIC_BASE_URL")
        else:
            base_url = None
        return JudgeTarget(
            provider=provider,
            model=model,
            api_key=_required_judge_credential(environ, "ANTHROPIC_API_KEY", label, source=source),
            base_url=base_url,
            credential_env="ANTHROPIC_API_KEY",
            base_url_env="ANTHROPIC_BASE_URL",
        )
    if provider == "nv_build":
        return JudgeTarget(
            provider=provider,
            model=model,
            api_key=_required_judge_credential(environ, "NVIDIA_API_KEY", label, source=source),
            base_url=PUBLIC_NVIDIA_BUILD_BASE_URL,
            credential_env="NVIDIA_API_KEY",
        )
    if provider == "bedrock":
        return JudgeTarget(provider=provider, model=model, region=environ.get("AWS_REGION") or "us-west-2")
    return JudgeTarget(
        provider=provider,
        model=model,
        api_key=_required_judge_credential(environ, "SKILL_EVAL_LLM_API_KEY", label, source=source),
        base_url=_required_judge_credential(environ, "SKILL_EVAL_LLM_BASE_URL", label, source=source).rstrip("/"),
        credential_env="SKILL_EVAL_LLM_API_KEY",
        base_url_env="SKILL_EVAL_LLM_BASE_URL",
    )


def _required_judge_credential(
    environ: Mapping[str, str],
    variable: str,
    label: str,
    *,
    source: str = JUDGE_PANEL_ENV,
) -> str:
    value = environ.get(variable, "").strip()
    if not value:
        panel_input = f"{JUDGE_PANEL_ENV} (or --judge-panel)" if source == JUDGE_PANEL_ENV else source
        raise ProviderConfigurationError(
            f"Judge panel member {label} requires {variable}.\n\n"
            f"Set it in your shell:\n  export {variable}='...'\n\n"
            f"or remove {label} from {panel_input}.\n\n"
            f"Provider setup:\n  {_PROVIDER_SETUP_URL}"
        )
    return value


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _required(environ: Mapping[str, str], variable: str) -> str:
    value = environ.get(variable, "").strip()
    if not value:
        message = f"{variable} is required for the selected provider."
        if variable in {"NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SKILL_EVAL_LLM_API_KEY"}:
            message += (
                "\n\nSet it in your shell:\n"
                f"  export {variable}='your-api-key'\n\n"
                f"Provider setup:\n  {_PROVIDER_SETUP_URL}"
            )
        raise ProviderConfigurationError(message)
    return value


def _anthropic_base_url(environ: Mapping[str, str]) -> str | None:
    for variable in ("SKILL_EVAL_LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        if value := environ.get(variable):
            return _normalize_anthropic_base_url(value, variable=variable)
    return None


def _normalize_anthropic_base_url(value: str, *, variable: str) -> str:
    error = (
        f"{variable} must be an absolute HTTP or HTTPS URL representing an API root without credentials, query, fragment, "
        "whitespace, control characters, backslashes, an invalid authority, or a /v1/messages endpoint."
    )
    if "\\" in value or any(
        character.isspace() or unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        raise ProviderConfigurationError(error)
    if "?" in value or "#" in value:
        raise ProviderConfigurationError(error)

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise ProviderConfigurationError(error) from None

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.netloc.endswith(":")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ProviderConfigurationError(error)

    authority = _canonical_anthropic_authority(parsed.netloc, hostname)
    path = _canonical_anthropic_path(parsed.path)
    if authority is None or path is None:
        raise ProviderConfigurationError(error)

    path = path.rstrip("/")
    if path.endswith("/v1/messages"):
        raise ProviderConfigurationError(error)
    if path.endswith("/v1"):
        path = path.removesuffix("/v1")
    return urlunsplit((parsed.scheme, authority, path, "", ""))


def _canonical_anthropic_authority(netloc: str, hostname: str) -> str | None:
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            return None
        literal = netloc[1:closing_bracket]
        suffix = netloc[closing_bracket + 1 :]
        if literal.casefold() != hostname.casefold() or (
            suffix and (not suffix.startswith(":") or not suffix[1:].isascii() or not suffix[1:].isdigit())
        ):
            return None

        address = literal
        zone = ""
        if "%" in literal:
            address, separator, zone = literal.partition("%25")
            if not separator or "%" in address or "%" in zone or not _ANTHROPIC_IPV6_ZONE_RE.fullmatch(zone):
                return None
        try:
            ipaddress.IPv6Address(address)
        except ValueError:
            return None
        return f"[{address}{'%25' + zone if zone else ''}]{suffix}"

    if "%" in netloc or "[" in netloc or "]" in netloc:
        return None
    host = netloc
    suffix = ""
    if ":" in netloc:
        host, port = netloc.rsplit(":", maxsplit=1)
        if ":" in host or not port.isascii() or not port.isdigit():
            return None
        suffix = f":{port}"
    if host.casefold() != hostname.casefold():
        return None

    if "." in hostname and all(character in "0123456789." for character in hostname):
        try:
            ipaddress.IPv4Address(hostname)
        except ValueError:
            return None
        return f"{host}{suffix}"

    trailing_dot = host.endswith(".")
    dns_name = host.removesuffix(".")
    if not dns_name:
        return None
    if dns_name.isascii() and "_" in dns_name:
        canonical_name = dns_name.lower()
        label_pattern = _ANTHROPIC_INTERNAL_LABEL_RE
    else:
        try:
            canonical_name = idna.encode(dns_name.lower()).decode("ascii")
        except idna.IDNAError:
            return None
        label_pattern = _ANTHROPIC_DNS_LABEL_RE
    if len(canonical_name) > 253 or not all(label_pattern.fullmatch(label) for label in canonical_name.split(".")):
        return None
    return f"{canonical_name}{'.' if trailing_dot else ''}{suffix}"


def _canonical_anthropic_path(path: str) -> str | None:
    canonical: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            canonical.append(character)
            index += 1
            continue

        if index + 2 >= len(path) or path[index + 1] not in _HEX_DIGITS or path[index + 2] not in _HEX_DIGITS:
            return None
        octet = int(path[index + 1 : index + 3], 16)
        if octet in {0x2F, 0x5C, 0x7F} or octet < 0x20:
            return None
        if octet in _UNRESERVED_BYTES:
            canonical.append(chr(octet))
        else:
            canonical.append(f"%{octet:02X}")
        index += 3

    canonical_path = "".join(canonical)
    if "//" in canonical_path.rstrip("/"):
        return None
    decoded_octets = unquote_to_bytes(canonical_path)
    # A decoded percent is safe as data unless it opens a second escape layer.
    if any(
        decoded_octets[index] == 0x25
        and index + 2 < len(decoded_octets)
        and decoded_octets[index + 1] in _HEX_DIGIT_BYTES
        and decoded_octets[index + 2] in _HEX_DIGIT_BYTES
        for index in range(len(decoded_octets))
    ):
        return None
    decoded_path = decoded_octets.decode("utf-8", errors="replace")
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in decoded_path):
        return None
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        return None
    return quote(canonical_path, safe=_ANTHROPIC_PATH_SAFE)


def _selected_provider(environ: Mapping[str, str], variable: str) -> str:
    configured = environ.get(variable, "").strip().lower()
    if configured:
        return configured
    available = [
        provider
        for provider, credential in (
            ("nv_build", "NVIDIA_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
        )
        if environ.get(credential, "").strip()
    ]
    if len(available) > 1:
        choices = _SUPPORTED_PROVIDERS
        if "EMBEDDING" in variable:
            choices = choices - {"anthropic", "bedrock"}
        example = next(provider for provider in available if provider in choices)
        raise ProviderConfigurationError(
            f"{variable} is required.\n"
            "Multiple public provider credentials are configured.\n\n"
            f"Accepted values:\n  {', '.join(sorted(choices))}\n\n"
            f"Choose one, for example:\n  export {variable}={example}"
        )
    if available:
        return available[0]
    alternatives = "  openai     -> OPENAI_API_KEY\n"
    # Anthropic/Bedrock have no embedding models; do not suggest them here.
    if "EMBEDDING" not in variable:
        alternatives += "  anthropic  -> ANTHROPIC_API_KEY\n"
    raise ProviderConfigurationError(
        "No provider is configured.\n\n"
        "For NVIDIA Build, set:\n"
        f"  export {variable}=nv_build\n"
        "  export NVIDIA_API_KEY='your-api-key'\n"
        "  Get a key: https://build.nvidia.com\n\n"
        f"Other {variable} values and their keys:\n"
        f"{alternatives}\n"
        f"More providers and setup options:\n  {_PROVIDER_SETUP_URL}"
    )


def _validate_provider(provider: str, *, variable: str) -> None:
    if provider not in _SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ProviderConfigurationError(f"{variable} must be one of: {choices}.")
