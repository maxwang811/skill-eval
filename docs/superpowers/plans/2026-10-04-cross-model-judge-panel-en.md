# Tier 3 Cross-Model Judge Panel Implementation Plan

**Goal:** Let the three Tier 3 LLM metrics (`accuracy`, `goal_accuracy`, `behavior_check`) be scored by several judges from different model families. The system aggregates their verdicts into a more robust score and reports how well the judges agree. It also flags any judge from the same model family as the agent, because that judge may be biased toward its own family.

**Motivation:** Today each LLM metric in a trial calls one judge with one fixed provider and model (`_call_public_llm_with_provenance` in `templates/eval.py`). This causes three problems:

1. **Judge noise is invisible.** The same trajectory can get a very different score from a different judge model. The report shows only one number.
2. **Same-family preference.** Claude judging a claude-code trajectory, or GPT judging a codex trajectory, may score high, and nothing checks for this.
3. **Skill Lift may not be trustworthy.** Lift is the difference between two arm means. If switching judges flips the sign of the lift, the conclusion should not be trusted, and nothing currently detects this.

**What does not change (compatibility boundary):**

- With no panel configured, behavior and artifacts stay **byte-for-byte identical** to today: one judge.
- The top-level numeric keys of `reward.json` stay the same: six metrics plus `overall`. Panel details go only into `details` in `skill_evaluator_reward.json`.
- The deterministic metrics (`security`, `skill_execution`, `skill_efficiency`) are unaffected.
- `custom_only` mode and user-authored graders are unaffected.

---

## 1. Current State (Where the Changes Land)

| Stage | Location | Today |
|---|---|---|
| Host-side provider resolution | `resolve_llm_provider` in `src/skillevaluator/provider_config.py` | Resolves a single `ProviderConfig` |
| Verifier environment injection | `_provider_environment` in `tier3/harbor/runner.py` (~L478) | Injects one provider's key and URL |
| Verifier env allowlist | `_VERIFIER_PROVIDER_ENV_VARS` and `_VERIFIER_JUDGE_MODEL_ENV_VARS` in `tier3/harbor/adapter.py` (~L205–250) | Variables outside the allowlist never reach the verifier |
| Judge model override | `_judge_model_config` and `_job_judge_verifier_env` in `runner.py` (~L894–950) | Only `LLM_JUDGE_MODEL` and `SKILL_EVAL_JUDGE_MODEL`, each naming a single model |
| Pre-run credential probing | `add_probe_target("standard grader", ...)` in `runner.py` (~L2087) | Probes one judge route |
| In-container provider selection | `_public_provider`, `_resolve_url`, and `_call_public_llm_with_provenance` in `templates/eval.py` (~L1381, 1600, 2247) | Reads the **global** `SKILL_EVAL_LLM_PROVIDER` |
| The three judges | `judge_accuracy` (8117), `judge_goal_accuracy` (8173, may use RAGAS on OpenAI), and `judge_behavior_check` (8322) in `eval.py` | Call `call_public_llm` or `_call_public_llm_with_provenance` directly |
| Fail-closed wrapper | `_call_required_judge` in `eval.py` (8463) | Gives each metric a 180 s wall-time budget enforced with SIGALRM, which **works only on the main thread** |
| Verifier timeout | `DEFAULT_LLM_VERIFIER_TIMEOUT_SEC = 600` in `tier3/harbor/__init__.py` | Sized for one judge: 3 metrics × 180 s |
| Shared judge module | `tier3/eval_core/llm_judge.py` | Kept in behavioral parity with the template; also used by `report.py` |
| Custom metric extraction | `extract_custom_metrics` in `tier3/harbor/metrics.py` | **Treats any non-reserved top-level numeric reward key as a custom metric**, so panel fields must not go at the top level |

---

## 2. User Interface

### 2.1 Panel configuration comes from the host only

The panel can be configured **only** through host environment variables or a CLI flag. It is **never** read from `evals/config.yml`.

Reason: the skill author controls `evals/config.yml`. The existing design already blocks that file from rewriting credentials or base URLs through `runtime_env`. Choosing the judges is equally the evaluator's call. If a skill could pick its own judges, it could pick lenient ones.

```bash
# Comma-separated provider:model list. Provider values match SKILL_EVAL_LLM_PROVIDER.
export SKILL_EVAL_JUDGE_PANEL='openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b'

# Optional
export SKILL_EVAL_JUDGE_PANEL_AGGREGATION=vote      # vote (default) | median | mean
export SKILL_EVAL_JUDGE_PANEL_QUORUM=2              # defaults to floor(N/2)+1
export SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT=0.4      # flag a metric when its score range is >= this value
```

The matching CLI flag is `--judge-panel`. It is available on `tier3 evaluate`, on the direct `tier3 <path>` workflow, and in the Tier 3 flag group of `validate`. A CLI value overrides the environment variable.

### 2.2 Credentials

Each panel member uses the existing credential variables of its provider: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NVIDIA_API_KEY`, or `AWS_*`. If any member's credential is missing at startup, the run fails with a hard error. It never quietly falls back to a single judge.

An `openai-compatible` gateway may appear only once in the panel, because its `SKILL_EVAL_LLM_BASE_URL` and `SKILL_EVAL_LLM_API_KEY` each hold a single value. Support for multiple gateways is deferred to phase 2.

### 2.3 Validation rules (host side, before the run)

- Each entry must have the form `provider:model`, and the provider must be in `_SUPPORTED_PROVIDERS`.
- Duplicate `(provider, model)` pairs are rejected.
- The panel must have at least 1 member and at most 5, to keep cost bounded.
- The quorum must satisfy 1 ≤ quorum ≤ N.
- With an even N and `vote` aggregation, warn and recommend an odd number of members. Tie handling is described in §4.2.
- The panel cannot be combined with `LLM_JUDGE_MODEL` or `SKILL_EVAL_JUDGE_MODEL`, because the meanings conflict. Setting both is an error, with a hint explaining why.

---

## 3. Data Flow

```
host: resolve_judge_panel()  ─►  [JudgeTarget(provider, model, credential_env, base_url_env), ...]
        │
        ├─ pre-run: catalog probe for every member (reuses add_probe_target)
        ├─ run_config.judge → run_config.judge_panel (full list, redacted)
        └─ verifier env: each member's credential vars + normalized SKILL_EVAL_JUDGE_PANEL
                │
container: eval.py main()
        for metric in (accuracy, goal_accuracy, behavior_check):
            members = [run_judge(metric, target) for target in panel]   # sequential, each wrapped in _call_required_judge
            details[metric] = aggregate(members)                         # score + panel block
        reward.json: still six numbers + overall (using the aggregated scores)
        skill_evaluator_reward.json: details[metric].panel = {...}
                │
host: collector
        ├─ existing means, dimension scores, and lift stay as they are (using the aggregated scores)
        ├─ new: per-judge arm means and per-judge lift
        ├─ new: inter-judge agreement (Fleiss' κ and Krippendorff's α)
        └─ new: lift with same-family judges excluded
```

---

## 4. In-Container Implementation (`templates/eval.py`)

### 4.1 Pass the provider explicitly instead of reading global state

This step carries the most risk, because it controls where credentials are sent.

- Add `JudgeTarget = NamedTuple(provider, model)` and `_ACTIVE_JUDGE_TARGET: ContextVar[JudgeTarget | None]`.
- Add a `target: JudgeTarget | None = None` parameter to `_call_public_llm_with_provenance`. If no target is passed, read the contextvar. If the contextvar is empty too, use today's global logic. This fallback keeps the single-judge path unchanged.
- `_resolve_url(provider, *, primary)` applies `SKILL_EVAL_LLM_BASE_URL` only when `provider == SKILL_EVAL_LLM_PROVIDER`, that is, for the primary provider. Every other provider uses its official URL or its own `*_BASE_URL`. This rule **keeps an OpenAI key from being sent to an NVIDIA URL**, and the reverse.
- Look up API keys in a per-provider table:
  - `nv_build` uses `NVIDIA_API_KEY`
  - `openai` uses `OPENAI_API_KEY`
  - `openai-compatible` uses `SKILL_EVAL_LLM_API_KEY`
  - `anthropic` uses `ANTHROPIC_API_KEY`
- The three judge functions **keep their signatures**. Their internal `call_public_llm` calls read the contextvar automatically, which keeps the change small.
- In panel mode, disable `_fallback_models` and `LLM_JUDGE_FALLBACK_MODELS` by passing `allow_model_fallback=False`. Otherwise a member could silently swap to a different model and change the panel's composition.
- In panel mode, `_ragas_goal_accuracy_enabled()` always returns False, so every member runs the same custom goal-judge prompt and the results are comparable. Record `method: "custom"` in the provenance.

### 4.2 Aggregation rules

| Metric | Structured output from each member | `vote` (default) | `median` / `mean` |
|---|---|---|---|
| accuracy | 5 boolean criteria | Majority vote on each criterion; score = passed / 5 | Median or mean of member scores |
| behavior_check | One `passed` boolean per behavior | Majority vote on each behavior; score = passed / count | Same |
| goal_accuracy | `achieved` boolean plus a score | Majority vote on `achieved`; score = median of the majority-side member scores | Same |

**Ties** (possible with an even number of members): a tied criterion counts as 0.5. This is conservative and favors neither side.

Criterion-level voting is the default because it resists one judge's systematic bias better than averaging total scores. It also keeps each rubric item's verdict explainable.

### 4.3 Fail-closed semantics

- Each member runs inside its own `_call_required_judge` wrapper, with its own 180 s budget. SIGALRM behavior does not change.
- If at least `quorum` members succeed, aggregate over the successful ones and record the failures, with redacted reasons, in `panel.failed_members`.
- If fewer than `quorum` members succeed, mark the metric `status=error`. The trial then follows the existing path: `evaluation_status=failed`, a nonzero exit, and an unscoreable reward. **It must never be turned into a score of 0.**

### 4.4 Output schema

In `skill_evaluator_reward.json`, the `details` of each LLM metric gain a `panel` block:

```json
"accuracy": {
  "score": 0.8,
  "reason": "panel vote (3/3 judges)",
  "criteria": {"SKILL_IDENTIFIED": true, "...": "..."},
  "panel": {
    "aggregation": "vote",
    "quorum": 2,
    "members": [
      {"provider": "openai", "model": "gpt-5.6-sol", "family": "openai",
       "status": "ok", "score": 0.8, "criteria": {"...": "..."}, "reason": "..."},
      {"provider": "anthropic", "model": "claude-opus-5", "family": "anthropic",
       "status": "ok", "score": 1.0, "criteria": {"...": "..."}, "reason": "..."},
      {"provider": "nv_build", "model": "nvidia/nemotron-3-super-120b-a12b", "family": "nvidia",
       "status": "error", "reason": "[redacted] timeout"}
    ],
    "spread": 0.2,
    "agreement": 0.8,
    "disagreement": false,
    "failed_members": 1
  }
}
```

- `agreement` is the mean criterion-level agreement rate: for each criterion, the share of members that agree with the majority, averaged over all criteria.
- **No new numeric fields go at the top level.** `extract_custom_metrics` would treat them as custom metrics and pollute `custom_lift.json`. A regression test must lock this in.
- `family` comes from `_model_family(provider, model)`. It first matches model-id prefixes (`claude`, `gpt`, `o\d`, `nemotron`, `llama`, `mistral`, and so on) and falls back to the provider only when no prefix matches. nv_build hosts models from many families, so **family must be inferred from the model id, not from the provider alone**.

### 4.5 Timeouts

The panel runs sequentially, so total verifier time grows roughly N times. The changes:

- When the adapter writes `task.toml`, it sets `[verifier] timeout_sec = DEFAULT_LLM_VERIFIER_TIMEOUT_SEC × N`.
- Author-specified timeouts in native Harbor tasks are left as they are, following the existing rule that author timeouts are not modified. The docs will point users to `--timeout-multiplier` for this case.

Phase 1 does not run members in parallel. SIGALRM works only on the main thread, so parallel calls would need a reworked deadline mechanism. The existing `_ACTIVE_JUDGE_DEADLINE` contextvar plus the urllib timeout already covers most of what is needed. This work is deferred to phase 2 (see §8).

---

## 5. Host-Side Implementation

### 5.1 `provider_config.py`

- Add a `JudgeTarget` dataclass with the fields provider, model, api_key, base_url, credential_env, base_url_env, and region.
- Add `resolve_judge_panel(environ) -> list[JudgeTarget] | None`. It reuses `_selected_provider`, `_validate_provider`, and each provider's credential lookup, and returns None when no panel is configured.

### 5.2 `runner.py`

- In panel mode, `_judge_model_config` returns `{"enabled": True, "panel": [...], "aggregation": ..., "quorum": ...}`. This is stored in `run_config.judge` with credentials redacted.
- Credential probing calls `add_probe_target("standard grader: <provider>:<model>", ...)` once per member. Each member's probe result is recorded separately in `credential_validation`, with the same blocking and degraded semantics as today.
- `_provider_environment` merges in each member's credential variables and the normalized `SKILL_EVAL_JUDGE_PANEL*` values, alongside the primary provider's variables.
- Every member key is added to the reporter's redaction set, just like the existing `runtime_secret_values`.

### 5.3 `adapter.py`

- Add `SKILL_EVAL_JUDGE_PANEL`, `SKILL_EVAL_JUDGE_PANEL_AGGREGATION`, `SKILL_EVAL_JUDGE_PANEL_QUORUM`, and `SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT` to the verifier allowlist.
- Add the same four names to the `runtime_env` blocklist. A skill must not be able to inject or override panel settings through `harbor.runtime_env`, consistent with §2.1.
- Compute the verifier `timeout_sec` as described in §4.5.

### 5.4 `cli.py`

- Add `--judge-panel TEXT` to `tier3 evaluate`, the direct tier3 workflow, and `validate`, and forward it to `run_harbor_eval`.
- Make `doctor --verify-models` iterate over every panel member as well.

### 5.5 `collector.py` and the report

The following outputs are **added** next to the existing aggregation. No existing field changes.

- `<agent>/judge_panel.json`:
  - `per_judge`: for each member, both arms' means on the three LLM metrics, plus that judge's lift.
  - `agreement`: for each metric, across all trials, Fleiss' κ on the boolean criteria and Krippendorff's α (interval) on the scores.
  - `lift_sign_consistent`: whether every judge's overall LLM-metric lift has the same sign.
  - `same_family_judges`: judges from the same family as the agent model.
  - `lift_excluding_same_family`: the lift re-aggregated without same-family judges, or null if there are none.
  - `disagreement_cases`: ids of cases flagged `disagreement=true`, with each judge's score.
- `result.json`: a `judge_panel` summary under each agent.
- HTML report: a new "Judge Panel" section with:
  - a small per-judge lift table;
  - the κ and α values, highlighted yellow below 0.4 and red below 0.2;
  - a warning when the lift sign flips between judges;
  - a warning about same-family judges;
  - a list of high-disagreement cases that expands to show each judge's reasoning.
- The `compare` output gains a "judge agreement" column.

---

## 6. Shared-Module Parity (`eval_core/llm_judge.py`)

`report.py` uses the shared module's `call_public_llm` for insights, and many tests cover both implementations through the `judge_module` fixture. Therefore:

- Add the same `target` parameter and per-provider URL and key routing to the shared module's `call_public_llm`.
- Put the aggregation helpers (`aggregate_panel` and `_model_family`) in the shared module, with a verbatim copy in the template. The template runs standalone and cannot import skillevaluator. A parity test checks that the two copies behave the same.
- Insights themselves do **not** use the panel. They keep using the primary provider.

---

## 7. Tasks and Tests

Work test-first: each task starts by writing failing tests.

### Task 1: In-container credential routing (highest priority)
- [ ] Add `tests/tier3/test_judge_panel_routing.py`:
  - Each member's requests go only to that member's URL and carry only that member's key. Assert headers and URL by capturing real `urllib.request.Request` objects.
  - `SKILL_EVAL_LLM_BASE_URL` applies only to the primary provider.
  - `LLM_JUDGE_FALLBACK_MODELS` has no effect in panel mode.
  - With no panel configured, the request sequence is identical to today's (golden assertion).
- [ ] Implement §4.1.

### Task 2: Aggregation and fail-closed behavior
- [ ] Add `tests/tier3/test_judge_panel_aggregation.py`:
  - Cover vote, median, and mean. Ties count as 0.5. A member whose behavior count does not match is treated as invalid.
  - Cover the quorum boundaries: exactly at quorum, one short, and all members failing.
  - Below quorum, the artifacts have the same shape as today's judge-failure artifacts. Extend `test_judge_failure_artifacts.py` to check this.
  - RAGAS is not used in panel mode.
- [ ] Implement §4.2–4.4.

### Task 3: Host configuration, probing, and allowlists
- [ ] Add `tests/tier3/test_judge_panel_config.py` covering parsing, validation errors, missing credentials, the conflict with `LLM_JUDGE_MODEL`, and rejection of `SKILL_EVAL_JUDGE_PANEL` inside `harbor.runtime_env`.
- [ ] Extend `test_tier3_config_key_parity.py` to assert that `grading.judge_panel` in `evals/config.yml` is rejected as an unknown key.
- [ ] Assert that each member gets its own probe result in `credential_validation`.
- [ ] Assert that the verifier `timeout_sec` scales with N.

### Task 4: Collector and report
- [ ] Add `tests/tier3/test_judge_panel_collector.py`:
  - Use hand-built reward fixtures to verify per-judge lift, κ and α, `lift_sign_consistent`, and same-family exclusion.
  - **Regression test:** panel fields never appear in the output of `extract_custom_metrics`, and `custom_lift.json` does not change.
- [ ] Add report rendering tests, following `tests/tier3/test_report_renders_refs.py`.

### Task 5: Documentation
- [ ] `docs/configuration.mdx` and `docs/environment-variables.mdx`: the new variables and the credential requirements.
- [ ] `docs/tier3-live-evaluation.mdx`: a new "Cross-model judging" subsection with the cost formula (LLM judge calls × N).
- [ ] `docs/reports.mdx`: how to read the Judge Panel section, with rule-of-thumb κ thresholds.
- [ ] `docs/cli-reference.mdx` and `CHANGELOG.md`.

### Task 6: Live verification
- [ ] Run `reference_skills/calculator` and `text-analyzer` with a 3-judge panel and `--agents opencode,codex`:
  - Confirm from the provenance that all three providers were actually called.
  - Break one member's credential on purpose. Confirm that the quorum takes effect and that the report records the failure.
  - Set quorum to N and break the same credential. Confirm that the trial fails closed.
- [ ] Run `make lint test`.

---

## 8. Risks and Follow-Ups

| Risk | Mitigation |
|---|---|
| Credentials routed to the wrong endpoint | Implement Task 1 first; tests assert on real Request objects |
| N× cost and duration | Cap at 5 members; document the cost formula; off by default |
| Panel fields polluting custom metrics | Keep them under `details` only, locked by a regression test |
| Wrong family inference, especially for third-party models hosted on nv_build | Infer from model-id prefixes; record `unknown` when no prefix matches, and leave those judges out of same-family exclusion |
| A skill manipulating judge selection | Host-only configuration; the panel variables are blocked in `runtime_env` |

**Possible phase-2 work:**

1. Call panel members in parallel by dropping the SIGALRM dependency in favor of the contextvar deadline plus socket timeouts.
2. A human-labeled judge calibration set. `skillevaluator tier3 judge-calibrate` would compute each judge's κ against human labels, and the results could feed back into panel weights.
3. Multiple openai-compatible gateways, using a form like `gateway@<name>:model`.
4. Disagreement-driven calling: query 2 judges first, stop if they agree, and call a third only when they disagree. This keeps the average cost under about 2×.
