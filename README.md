# SkillEvaluator

![SkillEvaluator wordmark](docs/assets/skillevaluator-wordmark.svg)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/arXiv-2608.20614-b31b1b.svg)](https://arxiv.org/abs/2608.20614)

SkillEvaluator is a multi-tier framework for evaluating AI agent skills. An
agent skill is a folder with a `SKILL.md` (instructions plus optional scripts
and references), as defined by the [Agent Skills specification](https://agentskills.io/).
SkillEvaluator answers four questions about a skill:

- **Is it safe and well-formed?** (Tier 1: static validation)
- **Does it repeat itself or overlap other skills?** (Tier 2: deduplication)
- **Does it actually help an agent?** (Tier 3: live evaluation with and without the skill)
- **Can the verdict be trusted?** (Tier 3 cross-model judge panel)

Tier 3 implements the method from
[*Evaluating Skills, Not Just Agents*](https://arxiv.org/abs/2608.20614): each
task runs twice in a sandbox, once with the skill and once without, and the
difference in scores is the **Skill Lift**.

![SkillEvaluator three-tier pipeline](docs/assets/three-tier-overview.svg)

## Install

Requires Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/maxwang811/skill-eval.git
cd skill-eval
uv sync --python 3.13 --all-extras
source .venv/bin/activate      # or prefix commands with `uv run`
```

Tier 3 also needs Docker running. Full Tier 1 coverage uses the external
scanners Semgrep, SkillSpector, and Gitleaks; without them those checks are
reported as skipped. Check readiness with:

```bash
skillevaluator tier3 doctor --env-mode docker
```

## Configure a provider

The simplest setup is NVIDIA Build, which covers chat, embeddings, Tier 3
agents, and judging with one key ([get a key](https://build.nvidia.com)):

```bash
export SKILL_EVAL_LLM_PROVIDER=nv_build
export NVIDIA_API_KEY='nvapi-...'
```

Other providers:

| Provider | Variables | Default Tier 3 agent |
| --- | --- | --- |
| NVIDIA Build | `SKILL_EVAL_LLM_PROVIDER=nv_build`, `NVIDIA_API_KEY` | OpenCode |
| OpenAI | `SKILL_EVAL_LLM_PROVIDER=openai`, `OPENAI_API_KEY` | Codex |
| Anthropic | `SKILL_EVAL_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY` | Claude Code |
| Bedrock | `SKILL_EVAL_LLM_PROVIDER=bedrock`, AWS credentials, `AWS_REGION` | Claude Code |
| OpenAI-compatible gateway | `SKILL_EVAL_LLM_PROVIDER=openai-compatible`, `SKILL_EVAL_LLM_BASE_URL`, `SKILL_EVAL_LLM_API_KEY` | OpenCode |

When more than one provider key is set, `SKILL_EVAL_LLM_PROVIDER` is required.
Override the evaluator model with `SKILL_EVAL_LLM_MODEL`, the single judge's
model with `LLM_JUDGE_MODEL`, and agents with `--agents` / `--agent-model`.
Anthropic and Bedrock have no embeddings, so Tier 2 needs
`SKILL_EVAL_EMBEDDING_PROVIDER` set to `nv_build`, `openai`, or `openai-compatible`.

## Run it

Evaluate a skill folder (a directory containing `SKILL.md`):

```bash
skillevaluator validate ./my-skill          # Tier 1 + Tier 2 + Tier 3
skillevaluator validate ./my-skill --tiers 1,2   # skip live evaluation
```

Or run one tier at a time:

```bash
skillevaluator tier1 ./my-skill
skillevaluator tier2 ./my-skill [--catalog ./skill-catalog.json]
skillevaluator tier3 ./my-skill [--agents opencode,codex]
```

`validate` writes HTML, JSON, and `BENCHMARK.md` reports to `-o/--output-dir`.
Tier 3 results live under `my-skill/evals/results/<run>/` (`result.json`,
`report.html`, per-agent scores). Live model calls and sandboxes cost money;
start with one agent and a small dataset.

### Tier 1: validation

Static checks for structure, metadata, licensing, scripts, secrets, Unicode
smuggling, and dependency risks, plus an LLM rubric and LLM security analysis
when a provider is configured (`--no-llm` turns those off). Tier 1 always gates
`validate`.

### Tier 2: deduplication

Embeds the skill's sections and flags repeated guidance inside the skill. With
`--catalog`, it also compares the skill against other skills. Tier 2 gates by
default; `--no-block-on-dedup` makes it advisory.

### Tier 3: live agent evaluation

Tier 3 reads `evals/evals.json`, or creates one starter case when no dataset
exists (generate more with `skillevaluator tier3 create-eval-dataset ./my-skill --full`).
Each case runs in an isolated [Harbor](https://github.com/harbor-framework/harbor)
container with and without the skill, and a verifier scores six metrics:

| Metric | How it is scored |
| --- | --- |
| `security` | Deterministic trace scan for unsafe actions and secret leakage |
| `skill_execution` | Deterministic: was the skill activated and run correctly? |
| `skill_efficiency` | Deterministic: routing and tool-call productivity |
| `accuracy` | LLM judge, 5-criterion rubric |
| `goal_accuracy` | LLM judge: did the agent reach the user's goal? |
| `behavior_check` | LLM judge: did it follow the expected behaviors? |

The metrics roll up into five dimensions (security, correctness,
discoverability, effectiveness, efficiency) and a with-skill vs. without-skill
lift. Supported agents are `claude-code`, `codex`, and `opencode`; Docker is the
default sandbox (`--env-mode local` runs agents on the host and is experimental).
Tier 3 is advisory in `validate` unless you pass `--block-on-agent-eval`.

Custom graders (`evals/grader.py`, `grading.mode: default_plus_custom` or
`custom_only` in `evals/config.yml`) and native Harbor tasks (`evals/harbor/`)
are supported for skills whose success needs domain-specific checks.

## Multi-judge evaluation (cross-model judge panel)

By default each LLM metric is scored by one judge model. A single judge has two
problems: its noise is invisible, and a judge from the same model family as the
agent may favor it. The **judge panel** scores the three LLM metrics
(`accuracy`, `goal_accuracy`, `behavior_check`) with up to five judges from
different model families, combines their verdicts, and reports how much they agree.

### Turn it on

```bash
export SKILL_EVAL_JUDGE_PANEL='openai:gpt-5.6-sol,anthropic:claude-opus-5,nv_build:nvidia/nemotron-3-super-120b-a12b'
skillevaluator tier3 ./my-skill --agents opencode
```

or pass it per run with `--judge-panel` (on `tier3`, `tier3 evaluate`, and `validate`):

```bash
skillevaluator validate ./my-skill --judge-panel 'nv_build:openai/gpt-oss-20b,nv_build:mistralai/mistral-large-2-instruct,nv_build:nvidia/nemotron-3-super-120b-a12b'
```

A single NVIDIA key is enough for a cross-family panel, because NVIDIA Build
hosts models from several families.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SKILL_EVAL_JUDGE_PANEL` | Comma-separated `provider:model` judges (1–5) | unset = single judge |
| `SKILL_EVAL_JUDGE_PANEL_AGGREGATION` | `vote`, `median`, or `mean` | `vote` |
| `SKILL_EVAL_JUDGE_PANEL_QUORUM` | Judges that must succeed for a metric to score | `N // 2 + 1` |
| `SKILL_EVAL_JUDGE_PANEL_DISAGREEMENT` | Flag a metric when judge scores spread by at least this much | `0.4` |

`--judge-panel ''` turns a configured panel off for one run.

### How verdicts combine

- **vote** (default): majority vote on each rubric item (each accuracy
  criterion, each expected behavior, and goal `achieved`). A tie counts as 0.5.
  For goal accuracy, the score is the median score of the majority side.
- **median / mean**: the median or mean of the judges' scores.
- **Quorum and fail-closed**: if fewer than `quorum` judges succeed, the metric
  fails and the trial is unscoreable. It is never turned into a score of 0.
  Failed judges are recorded with redacted reasons.
- Judges run one after another, each with its own time budget. Model fallbacks
  and the RAGAS goal scorer are disabled in panel mode so every judge answers
  the same prompt with its own model.

### Cost and time

A panel multiplies LLM judge calls by N: up to 3 metrics × N judges (plus one
format retry each) per trial. The managed verifier timeout scales to
600 s × N. Native Harbor tasks keep their own timeouts; the run warns when a
task's verifier budget is below 600 s × N, and `--timeout-multiplier` raises it.

### What you get

- `reward.json` keeps the same six metrics plus `overall`, now using the panel's
  combined scores. Each judge's verdict is stored under
  `details.<metric>.panel` in `skill_evaluator_reward.json`.
- `<run>/<agent>/judge_panel.json`, also summarized in `result.json`:
  - **per-judge scores and lift**, so you can see whether one judge drives the result;
  - **agreement** per metric: Fleiss' κ on the yes/no verdicts and
    Krippendorff's α on the scores (rule of thumb: below 0.2 poor, 0.2–0.4 fair,
    above 0.6 substantial);
  - **lift sign check**: a warning when one judge sees the skill helping and
    another sees it hurting, in which case the LLM-metric lift is inconclusive;
  - **same-family judges**: judges from the agent model's family (inferred from
    the model id, such as `claude`, `gpt`, `nemotron`, `llama`). The report
    compares their lift with the other judges' lift, and shows the lift without
    them next to a reference that drops the same number of other-family
    judges. Compare those two numbers, not the headline lift, because a smaller
    vote panel changes the lift by itself;
  - **disagreement cases**: trial metrics where judges' scores spread by at
    least the threshold, with each judge's score and reasoning.
- The HTML report has a **Judge Panel** section with the same information
  (κ/α below 0.4 in yellow, below 0.2 in red), and `skillevaluator tier3 compare`
  adds a `judges κ/α` column.
- `skillevaluator tier3 doctor --verify-models` probes each judge's model with
  its own key before you spend money.

### Credentials and safety

- Each judge uses its own provider's usual key: `NVIDIA_API_KEY`,
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS credentials for Bedrock, or
  `SKILL_EVAL_LLM_API_KEY` + `SKILL_EVAL_LLM_BASE_URL` for one gateway judge.
  A missing key stops the run before it starts. Each judge only ever sends its
  key to its own provider's endpoint.
- At most one `openai-compatible` judge is allowed, and it cannot be combined
  with an `openai` or `anthropic` primary provider, because
  `SKILL_EVAL_LLM_BASE_URL` would then redirect that provider's key to the
  gateway. Use the gateway as the primary instead.
- The panel is set only by you, through the environment or `--judge-panel`. It
  cannot be set from `evals/config.yml`, `harbor.runtime_env`, native task env
  tables, or a skill's Dockerfile. Setting it together with `LLM_JUDGE_MODEL` is
  an error.
- Judge keys are never added to the agent's launch environment. The verifier
  does run inside the agent's container, though, where code from the skill can
  also run, so treat judge keys as readable by the skill under test and use
  dedicated, spend-capped keys for skills you don't trust. With a panel, a
  custom grader that the skill selects for itself is refused unless you pass
  `--grading-mode default_plus_custom`.

## Develop

```bash
make lint      # ruff check src tests
make test      # pytest
make build
```

## Citation and license

Cite the paper for methodology: [arXiv:2608.20614](https://arxiv.org/abs/2608.20614)
(see [CITATION.cff](CITATION.cff)). Licensed under Apache 2.0; see
[LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This project downloads third-party open-source software; review their licenses before use.
