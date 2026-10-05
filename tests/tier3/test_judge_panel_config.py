# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side judge panel configuration: parsing, validation, credentials, and staging guards."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from skillevaluator.provider_config import (
    DEFAULT_JUDGE_PANEL_DISAGREEMENT,
    JUDGE_PANEL_AGGREGATION_ENV,
    JUDGE_PANEL_AGGREGATIONS,
    JUDGE_PANEL_DISAGREEMENT_ENV,
    JUDGE_PANEL_ENV,
    JUDGE_PANEL_ENV_VARS,
    JUDGE_PANEL_MAX_MEMBERS,
    JUDGE_PANEL_QUORUM_ENV,
    OPENAI_BASE_URL,
    PUBLIC_NVIDIA_BUILD_BASE_URL,
    JudgePanelConfig,
    JudgeTarget,
    ProviderConfig,
    ProviderConfigurationError,
    resolve_judge_panel,
    resolve_judge_panel_config,
    resolve_llm_provider,
)
from skillevaluator.tier3.harbor import DEFAULT_LLM_VERIFIER_TIMEOUT_SEC, runner
from skillevaluator.tier3.harbor.adapter import (
    _VERIFIER_PROVIDER_ENV_VARS,
    _runtime_env_toml_block,
    _write_task_toml,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)

OPENAI_KEY = "sk-openai-member-secret"
ANTHROPIC_KEY = "sk-ant-member-secret"
NVIDIA_KEY = "nvapi-member-secret"
GATEWAY_KEY = "gateway-member-secret"
GATEWAY_URL = "https://gateway.example/v1"
PANEL_3 = "openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b"


GATEWAY = {"SKILL_EVAL_LLM_API_KEY": GATEWAY_KEY, "SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL}


def _credentials(**overrides: str) -> dict[str, str]:
    """Return a host environment with every native provider credential configured."""
    environment = {
        "SKILL_EVAL_LLM_PROVIDER": "nv_build",
        "NVIDIA_API_KEY": NVIDIA_KEY,
        "OPENAI_API_KEY": OPENAI_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
    }
    environment.update(overrides)
    return environment


def _panel(panel: str, **environment: str) -> JudgePanelConfig:
    config = resolve_judge_panel_config(_credentials(**{JUDGE_PANEL_ENV: panel, **environment}))
    assert config is not None
    return config


def _member(entry: str, **environment: str) -> JudgeTarget:
    (member,) = _panel(entry, **environment).members
    return member


def _error(environment: dict[str, str]) -> str:
    with pytest.raises(ProviderConfigurationError) as excinfo:
        resolve_judge_panel_config(environment)
    return str(excinfo.value)


def test_judge_panel_constants_match_the_shared_contract() -> None:
    assert JUDGE_PANEL_ENV == "SKILL_EVAL_JUDGE_PANEL"
    assert JUDGE_PANEL_AGGREGATION_ENV == "SKILL_EVAL_JUDGE_PANEL_AGGREGATION"
    assert JUDGE_PANEL_QUORUM_ENV == "SKILL_EVAL_JUDGE_PANEL_QUORUM"
    assert JUDGE_PANEL_DISAGREEMENT_ENV == "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT"
    assert (
        frozenset({JUDGE_PANEL_ENV, JUDGE_PANEL_AGGREGATION_ENV, JUDGE_PANEL_QUORUM_ENV, JUDGE_PANEL_DISAGREEMENT_ENV})
        == JUDGE_PANEL_ENV_VARS
    )
    assert JUDGE_PANEL_AGGREGATIONS == ("vote", "median", "mean")
    assert JUDGE_PANEL_MAX_MEMBERS == 5
    assert DEFAULT_JUDGE_PANEL_DISAGREEMENT == 0.4


# --- Parsing ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw_panel", [None, "", "   ", "\t\n"])
def test_unset_or_blank_panel_means_no_panel(raw_panel: str | None) -> None:
    environment = _credentials()
    if raw_panel is not None:
        environment[JUDGE_PANEL_ENV] = raw_panel

    assert resolve_judge_panel_config(environment) is None
    assert resolve_judge_panel(environment) is None


def test_blank_panel_knobs_without_a_panel_are_not_configuration() -> None:
    environment = {JUDGE_PANEL_AGGREGATION_ENV: "", JUDGE_PANEL_QUORUM_ENV: "  ", JUDGE_PANEL_DISAGREEMENT_ENV: "\t"}

    assert resolve_judge_panel_config(environment) is None


def test_panel_parses_normalizes_and_applies_defaults() -> None:
    config = _panel(" OpenAI : gpt-5.6-sol , anthropic:claude-opus-5,NV_BUILD: nvidia/nemotron-3-super-120b-a12b ")

    assert [(member.provider, member.model) for member in config.members] == [
        ("openai", "gpt-5.6-sol"),
        ("anthropic", "claude-opus-5"),
        ("nv_build", "nvidia/nemotron-3-super-120b-a12b"),
    ]
    assert [member.label for member in config.members] == PANEL_3.split(",")
    assert config.aggregation == "vote"
    assert config.quorum == 2
    assert config.disagreement_threshold == 0.4
    assert config.warnings == ()
    assert config.env_value() == PANEL_3
    assert resolve_judge_panel(_credentials(**{JUDGE_PANEL_ENV: PANEL_3})) == list(config.members)


def test_model_ids_split_on_the_first_colon_only() -> None:
    member = _member("bedrock:us.anthropic.claude-opus-5-v1:0")

    assert member.provider == "bedrock"
    assert member.model == "us.anthropic.claude-opus-5-v1:0"
    assert member.label == "bedrock:us.anthropic.claude-opus-5-v1:0"


@pytest.mark.parametrize(
    ("member_count", "expected_quorum"),
    [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)],
)
def test_default_quorum_is_a_strict_majority(member_count: int, expected_quorum: int) -> None:
    panel = ",".join(f"openai:judge-{index}" for index in range(member_count))

    assert _panel(panel).quorum == expected_quorum


# --- Validation ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_panel", "expected"),
    [
        ("openai:gpt-5.6-sol,,anthropic:claude-opus-5", "empty"),
        ("openai:gpt-5.6-sol,", "empty"),
        (",openai:gpt-5.6-sol", "empty"),
        ("openai", "provider:model"),
        ("openai gpt-5.6-sol", "provider:model"),
        (":gpt-5.6-sol", "provider:model"),
        ("openai:", "provider:model"),
        ("openai:   ", "provider:model"),
        ("gemini:gemini-pro", "gemini"),
        ("openai:gpt 5", "whitespace"),
        ("openai:gpt-5\x07", "whitespace"),
    ],
)
def test_malformed_panel_entries_are_rejected(raw_panel: str, expected: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: raw_panel}))

    assert JUDGE_PANEL_ENV in message
    assert expected in message


def test_unsupported_provider_error_lists_the_supported_providers() -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: "gemini:gemini-pro"}))

    for provider in ("anthropic", "bedrock", "nv_build", "openai", "openai-compatible"):
        assert provider in message


def test_panel_size_is_bounded_to_keep_cost_predictable() -> None:
    six_members = ",".join(f"openai:judge-{index}" for index in range(6))

    message = _error(_credentials(**{JUDGE_PANEL_ENV: six_members}))

    assert "at most 5" in message
    assert "6" in message


@pytest.mark.parametrize(
    "raw_panel",
    [
        "openai:gpt-5.6-sol,anthropic:claude-opus-5,openai:gpt-5.6-sol",
        "OpenAI:gpt-5.6-sol, openai : gpt-5.6-sol",
    ],
)
def test_duplicate_provider_model_pairs_are_rejected(raw_panel: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: raw_panel}))

    assert "openai:gpt-5.6-sol" in message
    assert "more than once" in message


def test_model_ids_remain_case_sensitive_for_duplicate_detection() -> None:
    config = _panel("openai:GPT-X,openai:gpt-x")

    assert [member.model for member in config.members] == ["GPT-X", "gpt-x"]


def test_only_one_openai_compatible_gateway_member_is_supported() -> None:
    message = _error(
        _credentials(**GATEWAY, **{JUDGE_PANEL_ENV: "openai-compatible:judge-a,openai-compatible:judge-b"})
    )

    assert "openai-compatible" in message
    assert "at most one" in message


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("vote", "vote"), ("MEDIAN", "median"), (" Mean ", "mean"), ("", "vote")],
)
def test_aggregation_is_case_insensitive_with_a_vote_default(configured: str, expected: str) -> None:
    assert _panel(PANEL_3, **{JUDGE_PANEL_AGGREGATION_ENV: configured}).aggregation == expected


@pytest.mark.parametrize("configured", ["average", "majority", "max"])
def test_unknown_aggregation_is_rejected(configured: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: PANEL_3, JUDGE_PANEL_AGGREGATION_ENV: configured}))

    assert JUDGE_PANEL_AGGREGATION_ENV in message
    assert "vote, median, mean" in message


@pytest.mark.parametrize(("configured", "expected"), [("1", 1), (" 3 ", 3), ("2", 2)])
def test_quorum_accepts_integers_between_one_and_panel_size(configured: str, expected: int) -> None:
    assert _panel(PANEL_3, **{JUDGE_PANEL_QUORUM_ENV: configured}).quorum == expected


@pytest.mark.parametrize("configured", ["0", "4", "-1", "two", "1.5", "nan"])
def test_quorum_outside_one_to_panel_size_is_rejected(configured: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: PANEL_3, JUDGE_PANEL_QUORUM_ENV: configured}))

    assert JUDGE_PANEL_QUORUM_ENV in message
    assert "between 1 and 3" in message


@pytest.mark.parametrize(("configured", "expected"), [("0", 0.0), ("1", 1.0), ("0.25", 0.25), (" 0.5 ", 0.5)])
def test_disagreement_threshold_accepts_finite_values_in_unit_interval(configured: str, expected: float) -> None:
    assert _panel(PANEL_3, **{JUDGE_PANEL_DISAGREEMENT_ENV: configured}).disagreement_threshold == expected


@pytest.mark.parametrize("configured", ["abc", "nan", "inf", "-inf", "-0.1", "1.5"])
def test_disagreement_threshold_outside_unit_interval_is_rejected(configured: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: PANEL_3, JUDGE_PANEL_DISAGREEMENT_ENV: configured}))

    assert JUDGE_PANEL_DISAGREEMENT_ENV in message
    assert "between 0 and 1" in message


@pytest.mark.parametrize(
    "knob",
    [JUDGE_PANEL_AGGREGATION_ENV, JUDGE_PANEL_QUORUM_ENV, JUDGE_PANEL_DISAGREEMENT_ENV],
)
@pytest.mark.parametrize("raw_panel", [None, "  "])
def test_panel_knobs_without_a_panel_fail_instead_of_falling_back(knob: str, raw_panel: str | None) -> None:
    valid_values = {
        JUDGE_PANEL_AGGREGATION_ENV: "median",
        JUDGE_PANEL_QUORUM_ENV: "1",
        JUDGE_PANEL_DISAGREEMENT_ENV: "1",
    }
    environment = _credentials(**{knob: valid_values[knob]})
    if raw_panel is not None:
        environment[JUDGE_PANEL_ENV] = raw_panel

    message = _error(environment)

    assert knob in message
    assert JUDGE_PANEL_ENV in message


@pytest.mark.parametrize("override", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
def test_panel_conflicts_with_single_judge_model_overrides(override: str) -> None:
    message = _error(_credentials(**{JUDGE_PANEL_ENV: PANEL_3, override: "single-judge-model"}))

    assert override in message
    assert (
        "the panel names each judge's model explicitly; unset LLM_JUDGE_MODEL/SKILL_EVAL_JUDGE_MODEL "
        "or remove the panel"
    ) in message


@pytest.mark.parametrize("override", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
def test_blank_single_judge_overrides_do_not_conflict_with_the_panel(override: str) -> None:
    assert _panel(PANEL_3, **{override: "   "}).env_value() == PANEL_3


def test_single_judge_overrides_without_a_panel_are_untouched() -> None:
    environment = _credentials(LLM_JUDGE_MODEL="legacy-judge", SKILL_EVAL_JUDGE_MODEL="judge")

    assert resolve_judge_panel_config(environment) is None


@pytest.mark.parametrize(
    ("raw_panel", "aggregation", "warned"),
    [
        ("openai:a,anthropic:b", "vote", True),
        ("openai:a,anthropic:b,openai:c,anthropic:d", "", True),
        ("openai:a,anthropic:b", "median", False),
        ("openai:a,anthropic:b", "mean", False),
        ("openai:a", "vote", False),
        ("openai:a,anthropic:b,openai:c", "vote", False),
    ],
)
def test_even_vote_panels_warn_without_failing(raw_panel: str, aggregation: str, warned: bool) -> None:
    config = _panel(raw_panel, **{JUDGE_PANEL_AGGREGATION_ENV: aggregation})

    if warned:
        (warning,) = config.warnings
        assert "odd number" in warning
        assert "0.5" in warning
    else:
        assert config.warnings == ()


# --- Member credentials and endpoints --------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "extra", "expected_base_url"),
    [
        ("nv_build", {}, OPENAI_BASE_URL),
        ("nv_build", {"OPENAI_BASE_URL": "https://openai-proxy.example/v1/"}, "https://openai-proxy.example/v1"),
        # The primary's endpoint override applies to an OpenAI member only when OpenAI is the primary.
        (
            "openai",
            {"SKILL_EVAL_LLM_BASE_URL": "https://primary-proxy.example/v1/"},
            "https://primary-proxy.example/v1",
        ),
        ("openai-compatible", GATEWAY, OPENAI_BASE_URL),
        ("anthropic", {"SKILL_EVAL_LLM_BASE_URL": "https://anthropic-gateway.example/v1"}, OPENAI_BASE_URL),
    ],
)
def test_openai_member_resolves_its_own_credential_and_endpoint(
    primary: str,
    extra: dict[str, str],
    expected_base_url: str,
) -> None:
    member = _member("openai:gpt-5.6-sol", SKILL_EVAL_LLM_PROVIDER=primary, **extra)

    assert member.api_key == OPENAI_KEY
    assert member.base_url == expected_base_url
    assert member.credential_env == "OPENAI_API_KEY"
    assert member.base_url_env == "OPENAI_BASE_URL"
    assert member.region is None


def test_openai_member_base_url_matches_the_openai_primary() -> None:
    environment = _credentials(
        SKILL_EVAL_LLM_PROVIDER="openai",
        SKILL_EVAL_LLM_BASE_URL="https://primary-proxy.example/v1",
        **{JUDGE_PANEL_ENV: "openai:gpt-5.4-mini"},
    )
    (member,) = resolve_judge_panel(environment) or []

    assert member.base_url == resolve_llm_provider(environment).base_url


@pytest.mark.parametrize(
    ("primary", "extra", "expected_base_url"),
    [
        ("nv_build", {}, None),
        ("nv_build", {"SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL}, None),
        ("nv_build", {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1/"}, "https://anthropic-proxy.example"),
        (
            "anthropic",
            {"SKILL_EVAL_LLM_BASE_URL": "https://anthropic-gateway.example/v1"},
            "https://anthropic-gateway.example",
        ),
        (
            "anthropic",
            {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/team/v1"},
            "https://anthropic-proxy.example/team",
        ),
        ("openai-compatible", GATEWAY, None),
        ("openai", {"SKILL_EVAL_LLM_BASE_URL": "https://openai-proxy.example/v1"}, None),
    ],
)
def test_anthropic_member_resolves_its_own_credential_and_endpoint(
    primary: str,
    extra: dict[str, str],
    expected_base_url: str | None,
) -> None:
    member = _member("anthropic:claude-opus-5", SKILL_EVAL_LLM_PROVIDER=primary, **extra)

    assert member.api_key == ANTHROPIC_KEY
    assert member.base_url == expected_base_url
    assert member.credential_env == "ANTHROPIC_API_KEY"
    assert member.base_url_env == "ANTHROPIC_BASE_URL"


def test_invalid_anthropic_member_base_url_is_a_configuration_error() -> None:
    message = _error(
        _credentials(
            ANTHROPIC_BASE_URL="https://user:secret@anthropic-proxy.example/v1",
            **{JUDGE_PANEL_ENV: "anthropic:claude-opus-5"},
        )
    )

    assert "ANTHROPIC_BASE_URL" in message
    assert "secret" not in message


@pytest.mark.parametrize("primary", ["nv_build", "openai", "openai-compatible", "anthropic"])
def test_nvidia_build_member_always_uses_the_public_endpoint(primary: str) -> None:
    member = _member("nv_build:nvidia/nemotron-3-super-120b-a12b", SKILL_EVAL_LLM_PROVIDER=primary, **GATEWAY)

    assert member.api_key == NVIDIA_KEY
    assert member.base_url == PUBLIC_NVIDIA_BUILD_BASE_URL
    assert member.credential_env == "NVIDIA_API_KEY"
    assert member.base_url_env is None


@pytest.mark.parametrize(("region", "expected"), [(None, "us-west-2"), ("eu-central-1", "eu-central-1")])
def test_bedrock_member_uses_the_aws_credential_chain(region: str | None, expected: str) -> None:
    extra = {} if region is None else {"AWS_REGION": region}
    member = _member("bedrock:us.anthropic.claude-opus-5", **extra)

    assert member.api_key is None
    assert member.base_url is None
    assert member.region == expected
    assert member.credential_env is None


def test_openai_compatible_member_uses_the_single_gateway_pair() -> None:
    member = _member(
        "openai-compatible:nvidia/nvidia/nemotron-3-super-120b-long-ctx",
        SKILL_EVAL_LLM_API_KEY=GATEWAY_KEY,
        SKILL_EVAL_LLM_BASE_URL=GATEWAY_URL + "/",
    )

    assert member.api_key == GATEWAY_KEY
    assert member.base_url == GATEWAY_URL
    assert member.credential_env == "SKILL_EVAL_LLM_API_KEY"
    assert member.base_url_env == "SKILL_EVAL_LLM_BASE_URL"


@pytest.mark.parametrize(
    ("entry", "variable"),
    [
        ("openai:gpt-5.6-sol", "OPENAI_API_KEY"),
        ("anthropic:claude-opus-5", "ANTHROPIC_API_KEY"),
        ("nv_build:nvidia/nemotron-3-super-120b-a12b", "NVIDIA_API_KEY"),
        ("openai-compatible:gateway-judge", "SKILL_EVAL_LLM_API_KEY"),
        ("openai-compatible:gateway-judge", "SKILL_EVAL_LLM_BASE_URL"),
    ],
)
@pytest.mark.parametrize("configured", [None, "   "])
def test_missing_member_credential_is_a_hard_error_naming_member_and_variable(
    entry: str,
    variable: str,
    configured: str | None,
) -> None:
    environment = _credentials(**GATEWAY, **{JUDGE_PANEL_ENV: f"bedrock:us.anthropic.claude-opus-5,{entry}"})
    environment.pop(variable)
    if configured is not None:
        environment[variable] = configured

    message = _error(environment)

    assert entry in message
    assert variable in message


def test_primary_is_inferred_from_the_single_configured_credential() -> None:
    environment = {
        "OPENAI_API_KEY": OPENAI_KEY,
        "SKILL_EVAL_LLM_BASE_URL": "https://primary-proxy.example/v1",
        JUDGE_PANEL_ENV: "openai:gpt-5.6-sol",
    }
    (member,) = resolve_judge_panel(environment) or []

    assert resolve_llm_provider(environment).provider == "openai"
    assert member.base_url == "https://primary-proxy.example/v1"


def test_ambiguous_primary_does_not_apply_the_primary_endpoint_override() -> None:
    environment = {
        "OPENAI_API_KEY": OPENAI_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "SKILL_EVAL_LLM_BASE_URL": "https://unattributed-proxy.example/v1",
        JUDGE_PANEL_ENV: "openai:gpt-5.6-sol,anthropic:claude-opus-5",
    }
    openai_member, anthropic_member = resolve_judge_panel(environment) or []

    assert openai_member.base_url == OPENAI_BASE_URL
    assert anthropic_member.base_url is None


# --- JudgeTarget / JudgePanelConfig helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "extra"),
    [
        ("openai", {"SKILL_EVAL_LLM_BASE_URL": "https://primary-proxy.example/v1/"}),
        ("anthropic", {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1"}),
        ("anthropic", {}),
        ("nv_build", {}),
        ("bedrock", {"AWS_REGION": "us-east-1"}),
        ("openai-compatible", GATEWAY),
    ],
)
def test_member_provider_config_matches_the_equivalent_primary_provider(provider: str, extra: dict[str, str]) -> None:
    environment = _credentials(
        SKILL_EVAL_LLM_PROVIDER=provider,
        SKILL_EVAL_LLM_MODEL="judge-model",
        **{JUDGE_PANEL_ENV: f"{provider}:judge-model"},
        **extra,
    )
    (member,) = resolve_judge_panel(environment) or []

    provider_config = member.provider_config()

    assert isinstance(provider_config, ProviderConfig)
    assert provider_config == resolve_llm_provider(environment)


@pytest.mark.parametrize(
    ("entry", "extra", "expected"),
    [
        (
            "openai:gpt-5.6-sol",
            {},
            {"OPENAI_API_KEY": OPENAI_KEY, "OPENAI_BASE_URL": "https://api.openai.com/v1"},
        ),
        ("anthropic:claude-opus-5", {}, {"ANTHROPIC_API_KEY": ANTHROPIC_KEY}),
        (
            "anthropic:claude-opus-5",
            {"ANTHROPIC_BASE_URL": "https://anthropic-proxy.example/v1"},
            {"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "ANTHROPIC_BASE_URL": "https://anthropic-proxy.example"},
        ),
        ("nv_build:nvidia/nemotron-3-super-120b-a12b", {}, {"NVIDIA_API_KEY": NVIDIA_KEY}),
        ("bedrock:us.anthropic.claude-opus-5", {}, {"AWS_REGION": "us-west-2"}),
        (
            "openai-compatible:gateway-judge",
            GATEWAY,
            {"SKILL_EVAL_LLM_API_KEY": GATEWAY_KEY, "SKILL_EVAL_LLM_BASE_URL": GATEWAY_URL},
        ),
    ],
)
def test_member_verifier_environment_uses_canonical_in_container_names(
    entry: str,
    extra: dict[str, str],
    expected: dict[str, str],
) -> None:
    assert _member(entry, **extra).verifier_environment() == expected


@pytest.mark.parametrize(
    ("provider", "extra"),
    [
        ("openai", {"SKILL_EVAL_LLM_BASE_URL": "https://primary-proxy.example/v1"}),
        ("openai", {}),
        ("anthropic", {"SKILL_EVAL_LLM_BASE_URL": "https://anthropic-gateway.example/v1"}),
        ("anthropic", {}),
        ("nv_build", {}),
    ],
)
def test_primary_equivalent_member_values_match_the_primary_verifier_environment(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    extra: dict[str, str],
) -> None:
    environment = _credentials(SKILL_EVAL_LLM_PROVIDER=provider, **{JUDGE_PANEL_ENV: f"{provider}:judge-x"}, **extra)
    monkeypatch.setattr(runner.os, "environ", environment)
    (member,) = resolve_judge_panel(environment) or []

    primary_environment = runner._provider_environment(resolve_llm_provider(environment))

    for name, value in member.verifier_environment().items():
        assert primary_environment[name] == value


def test_member_repr_never_contains_the_credential() -> None:
    member = _member("openai:gpt-5.6-sol")

    assert OPENAI_KEY not in repr(member)
    assert OPENAI_KEY not in repr(_panel(PANEL_3))


def test_verifier_settings_are_normalized_strings() -> None:
    config = _panel(
        " anthropic : claude-opus-5 ,OPENAI:gpt-5.6-sol",
        **{
            JUDGE_PANEL_AGGREGATION_ENV: " Median ",
            JUDGE_PANEL_QUORUM_ENV: " 1 ",
            JUDGE_PANEL_DISAGREEMENT_ENV: " 0.25 ",
        },
    )

    assert config.verifier_settings() == {
        "SKILL_EVAL_JUDGE_PANEL": "anthropic:claude-opus-5,openai:gpt-5.6-sol",
        "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "median",
        "SKILL_EVAL_JUDGE_PANEL_QUORUM": "1",
        "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.25",
    }


def test_verifier_settings_always_carry_resolved_defaults() -> None:
    assert _panel(PANEL_3).verifier_settings() == {
        "SKILL_EVAL_JUDGE_PANEL": PANEL_3,
        "SKILL_EVAL_JUDGE_PANEL_AGGREGATION": "vote",
        "SKILL_EVAL_JUDGE_PANEL_QUORUM": "2",
        "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT": "0.4",
    }


@pytest.mark.parametrize(
    ("raw_panel", "knobs"),
    [
        (PANEL_3, {}),
        (
            " anthropic : claude-opus-5 ,OPENAI:gpt-5.6-sol",
            {
                JUDGE_PANEL_AGGREGATION_ENV: " Median ",
                JUDGE_PANEL_QUORUM_ENV: " 1 ",
                JUDGE_PANEL_DISAGREEMENT_ENV: "0.25",
            },
        ),
        (
            "bedrock:us.anthropic.claude-opus-5-v1:0,openai-compatible:nvidia/nvidia/nemotron-3-super-120b-long-ctx,"
            "nv_build:nvidia/nemotron-3-super-120b-a12b,anthropic:claude-opus-5,openai:o4-mini",
            {JUDGE_PANEL_AGGREGATION_ENV: "mean", JUDGE_PANEL_QUORUM_ENV: "5", JUDGE_PANEL_DISAGREEMENT_ENV: "1"},
        ),
    ],
)
def test_normalized_settings_parse_identically_in_the_verifier(raw_panel: str, knobs: dict[str, str]) -> None:
    """Host and verifier must agree: the staged settings round-trip through the shared parser."""
    from skillevaluator.tier3.eval_core.llm_judge import parse_judge_panel_env

    config = _panel(raw_panel, **GATEWAY, **knobs)

    settings = parse_judge_panel_env(config.verifier_settings())

    assert settings is not None
    assert [(member.provider, member.model) for member in settings.members] == [
        (member.provider, member.model) for member in config.members
    ]
    assert settings.aggregation == config.aggregation
    assert settings.quorum == config.quorum
    assert settings.disagreement_threshold == config.disagreement_threshold


def test_redacted_panel_is_json_safe_and_secret_free() -> None:
    config = _panel(
        "openai:gpt-5.6-sol,anthropic:claude-opus-5",
        ANTHROPIC_BASE_URL="https://anthropic-proxy.example/v1",
    )

    redacted = config.redacted()

    assert redacted == {
        "panel": [
            {"provider": "openai", "model": "gpt-5.6-sol", "label": "openai:gpt-5.6-sol"},
            {"provider": "anthropic", "model": "claude-opus-5", "label": "anthropic:claude-opus-5"},
        ],
        "aggregation": "vote",
        "quorum": 2,
        "disagreement_threshold": 0.4,
        "warnings": list(config.warnings),
    }
    serialized = json.dumps(redacted)
    for secret in (OPENAI_KEY, ANTHROPIC_KEY, NVIDIA_KEY, GATEWAY_KEY):
        assert secret not in serialized
    assert "api_key" not in serialized
    assert "anthropic-proxy" not in serialized


# --- Runtime env and verifier allowlists ------------------------------------------------------------


def test_panel_names_are_explicit_host_controls_and_verifier_allowlisted() -> None:
    assert JUDGE_PANEL_ENV_VARS <= runner._RUNTIME_ENV_HOST_CONTROL_NAMES
    assert JUDGE_PANEL_ENV_VARS <= _VERIFIER_PROVIDER_ENV_VARS


@pytest.mark.parametrize(
    "name",
    [
        *sorted(JUDGE_PANEL_ENV_VARS),
        "skill_eval_judge_panel",
        "SKILLEVALUATOR_JUDGE_PANEL__OPENAI_API_KEY",
        "SKILLEVALUATOR_JUDGE_PANEL__ANTHROPIC_BASE_URL",
    ],
)
def test_skill_runtime_env_cannot_set_judge_panel_controls(name: str) -> None:
    resolved, errors = runner._resolve_runtime_env({name: "openai:lenient-judge"})

    assert resolved == {}
    assert len(errors) == 1
    assert name in errors[0]
    assert "host process" in errors[0]


@pytest.mark.parametrize(
    "reference",
    [*sorted(JUDGE_PANEL_ENV_VARS), "SKILLEVALUATOR_JUDGE_PANEL__OPENAI_API_KEY"],
)
@pytest.mark.parametrize("template", ["${{{name}}}", "${{{name}:-fallback}}", "$" + "{name}", "%{name}%"])
def test_skill_runtime_env_cannot_reference_judge_panel_controls(
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    template: str,
) -> None:
    monkeypatch.setenv(reference, "operator-value")

    resolved, errors = runner._resolve_runtime_env({"INNOCENT_NAME": template.format(name=reference)})

    assert resolved == {}
    assert len(errors) == 1
    assert reference in errors[0]
    assert "operator-owned" in errors[0]


# --- Verifier timeouts ------------------------------------------------------------------------------


_GENERATED_TASK_HEAD = (
    'schema_version = "1.3"\n\n[task]\nname = "nvidia/skillevaluator-case-001"\n'
    'description = "Skill evaluation task for demo"\n\n[metadata]\nskill = "demo"\nentry_id = "case-001"\n'
    "has_skill = true\n\n[agent]\ntimeout_sec = 300.0\n\n[verifier]\ntimeout_sec = {timeout}\n\n[verifier.env]\n"
    'NVIDIA_API_KEY = "${{NVIDIA_API_KEY}}"\nSKILL_EVAL_LLM_PROVIDER = "${{SKILL_EVAL_LLM_PROVIDER}}"\n\n'
    '[environment]\ncpus = 2\nmemory_mb = 4096\nstorage_mb = 2048\nnetwork_mode = "public"\n'
    'skills_dir = "/workspace/skills"\n'
)
_STAGED_VERIFIER_ENV = {"SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}", "NVIDIA_API_KEY": "${NVIDIA_API_KEY}"}


def _write_generated_task(task_dir: Path, **kwargs: float | None) -> str:
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_task_toml(
        task_dir,
        {"id": "case-001", "expected_skill": "demo"},
        True,
        runtime_env={},
        verifier_env=dict(_STAGED_VERIFIER_ENV),
        **kwargs,
    )
    return (task_dir / "task.toml").read_text(encoding="utf-8")


def test_generated_task_toml_is_byte_identical_without_a_panel(tmp_path: Path) -> None:
    default = _write_generated_task(tmp_path / "default")
    explicit_none = _write_generated_task(tmp_path / "explicit-none", verifier_timeout_sec=None)

    assert default == _GENERATED_TASK_HEAD.format(timeout="600.0") + _runtime_env_toml_block({})
    assert explicit_none == default
    assert "JUDGE_PANEL" not in default


@pytest.mark.parametrize("members", [1, 3, 5])
def test_generated_task_verifier_timeout_scales_with_panel_size(tmp_path: Path, members: int) -> None:
    scaled = DEFAULT_LLM_VERIFIER_TIMEOUT_SEC * members

    content = _write_generated_task(tmp_path / f"panel-{members}", verifier_timeout_sec=scaled)

    assert tomllib.loads(content)["verifier"]["timeout_sec"] == 600.0 * members
    assert content == _GENERATED_TASK_HEAD.format(timeout=f"{600.0 * members}") + _runtime_env_toml_block({})


@pytest.mark.parametrize("invalid", [0, -600.0, float("nan"), float("inf"), True])
def test_generated_task_rejects_invalid_verifier_timeouts(tmp_path: Path, invalid: float) -> None:
    with pytest.raises(ValueError, match="verifier timeout"):
        _write_generated_task(tmp_path / "invalid", verifier_timeout_sec=invalid)


def _generated_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "generated-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: generated-skill\ndescription: Demo skill.\n---\n# Demo\n",
        encoding="utf-8",
    )
    (skill / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Run the case.", "expected_answer": "ok", "files": []}]),
        encoding="utf-8",
    )
    return skill


def test_generate_harbor_tasks_threads_the_panel_timeout(tmp_path: Path) -> None:
    skill = _generated_skill(tmp_path)

    (baseline,) = generate_harbor_tasks(skill, tmp_path / "baseline", verifier_env=dict(_STAGED_VERIFIER_ENV))
    (panel,) = generate_harbor_tasks(
        skill,
        tmp_path / "panel",
        verifier_env=dict(_STAGED_VERIFIER_ENV),
        verifier_timeout_sec=DEFAULT_LLM_VERIFIER_TIMEOUT_SEC * 3,
    )

    baseline_text = (baseline / "task.toml").read_text(encoding="utf-8")
    panel_text = (panel / "task.toml").read_text(encoding="utf-8")
    assert tomllib.loads(baseline_text)["verifier"]["timeout_sec"] == 600.0
    assert panel_text == baseline_text.replace(
        "[verifier]\ntimeout_sec = 600.0\n",
        "[verifier]\ntimeout_sec = 1800.0\n",
    )


def _native_skill(tmp_path: Path, task_body: str, *, name: str = "native-skill") -> Path:
    skill = tmp_path / name
    evals = skill / "evals"
    task = evals / "harbor" / "case-001"
    task.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Demo skill.\n---\n# Demo\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Run the native case.", "expected_answer": "ok", "files": []}]),
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[metadata]\nentry_id = "case-001"\n\n'
        + task_body,
        encoding="utf-8",
    )
    return skill


def _stage_native(tmp_path: Path, skill: Path, *, verifier_timeout_sec: float | None) -> dict[str, object]:
    output = tmp_path / f"staged-{skill.name}-{verifier_timeout_sec}"
    (task,) = stage_native_harbor_tasks(
        skill,
        output,
        grading_mode="default",
        verifier_env={"SKILL_EVAL_LLM_PROVIDER": "openai", "SKILL_EVAL_LLM_MODEL": "gpt-5.6-sol"},
        verifier_timeout_sec=verifier_timeout_sec,
    )
    return tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("verifier_timeout_sec", [None, 1800.0])
def test_native_author_verifier_timeouts_are_never_modified(tmp_path: Path, verifier_timeout_sec: float | None) -> None:
    skill = _native_skill(tmp_path, "[verifier]\ntimeout_sec = 180.0\n\n[environment]\n")

    staged = _stage_native(tmp_path, skill, verifier_timeout_sec=verifier_timeout_sec)

    assert staged["verifier"]["timeout_sec"] == 180.0  # type: ignore[index]


@pytest.mark.parametrize(("verifier_timeout_sec", "expected"), [(None, 600.0), (1800.0, 1800.0)])
def test_native_task_without_verifier_table_gets_the_scaled_timeout(
    tmp_path: Path,
    verifier_timeout_sec: float | None,
    expected: float,
) -> None:
    skill = _native_skill(tmp_path, "[environment]\n")

    staged = _stage_native(tmp_path, skill, verifier_timeout_sec=verifier_timeout_sec)

    assert staged["verifier"]["timeout_sec"] == expected  # type: ignore[index]
    assert staged["verifier"]["env"]["SKILL_EVAL_LLM_PROVIDER"] == "${SKILL_EVAL_LLM_PROVIDER}"  # type: ignore[index]


@pytest.mark.parametrize(
    "task_body",
    [
        '[verifier.env]\nCUSTOM_VERIFIER_FLAG = "keep-me"\n\n[environment]\n',
        '[verifier]\nuser = "root"\n\n[environment]\n',
        '[verifier] # authored verifier settings\nuser = "root"\n\n[verifier.env]\nCUSTOM_VERIFIER_FLAG = "keep-me"\n',
    ],
)
def test_native_task_verifier_without_timeout_gets_panel_budget_only_in_panel_mode(
    tmp_path: Path,
    task_body: str,
) -> None:
    skill = _native_skill(tmp_path, task_body)

    unchanged = _stage_native(tmp_path, skill, verifier_timeout_sec=None)
    scaled = _stage_native(tmp_path, skill, verifier_timeout_sec=1800.0)

    assert "timeout_sec" not in unchanged["verifier"]  # type: ignore[operator]
    assert scaled["verifier"]["timeout_sec"] == 1800.0  # type: ignore[index]
    assert {key: value for key, value in scaled["verifier"].items() if key != "timeout_sec"} == unchanged["verifier"]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("table", "task_body", "control"),
    [
        (
            "environment.env",
            '[environment]\n\n[environment.env]\nSKILL_EVAL_JUDGE_PANEL = "openai:lenient-judge"\n',
            "SKILL_EVAL_JUDGE_PANEL",
        ),
        (
            "environment.env",
            '[environment]\n\n[environment.env]\nHELPER = "${SKILL_EVAL_JUDGE_PANEL_QUORUM}"\n',
            "SKILL_EVAL_JUDGE_PANEL_QUORUM",
        ),
        (
            "verifier.env",
            '[verifier.env]\nSKILL_EVAL_JUDGE_PANEL_AGGREGATION = "mean"\n\n[environment]\n',
            "SKILL_EVAL_JUDGE_PANEL_AGGREGATION",
        ),
        (
            "verifier.env",
            '[verifier.env]\nskill_eval_judge_panel = "openai:lenient-judge"\n\n[environment]\n',
            "skill_eval_judge_panel",
        ),
        (
            "verifier.env",
            '[verifier.env]\nOPENAI_API_KEY = "${SKILLEVALUATOR_JUDGE_PANEL__OPENAI_API_KEY}"\n\n[environment]\n',
            "SKILLEVALUATOR_JUDGE_PANEL__OPENAI_API_KEY",
        ),
        (
            "verifier.env",
            '[verifier.env]\nSKILLEVALUATOR_JUDGE_PANEL__OPENAI_BASE_URL = "https://attacker.example/v1"\n\n[environment]\n',
            "SKILLEVALUATOR_JUDGE_PANEL__OPENAI_BASE_URL",
        ),
        (
            "verifier.environment.env",
            '[verifier.environment.env]\nPANEL = "%SKILL_EVAL_JUDGE_PANEL%"\n\n[environment]\n',
            "SKILL_EVAL_JUDGE_PANEL",
        ),
        (
            "steps[0].verifier.env",
            '[[steps]]\nname = "step-one"\n\n[steps.verifier.env]\nSKILL_EVAL_JUDGE_PANEL_DISAGREEMENT = "1.0"\n\n'
            "[environment]\n",
            "SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT",
        ),
    ],
)
@pytest.mark.parametrize("verifier_timeout_sec", [None, 1800.0])
def test_native_tasks_cannot_name_or_reference_judge_panel_controls(
    tmp_path: Path,
    table: str,
    task_body: str,
    control: str,
    verifier_timeout_sec: float | None,
) -> None:
    skill = _native_skill(tmp_path, task_body)

    with pytest.raises(ValueError, match="judge panel") as excinfo:
        _stage_native(tmp_path, skill, verifier_timeout_sec=verifier_timeout_sec)

    assert f"[{table}]" in str(excinfo.value)
    assert control in str(excinfo.value)
