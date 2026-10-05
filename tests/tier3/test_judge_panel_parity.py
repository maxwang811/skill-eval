# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Harbor verifier carries a verbatim copy of the shared judge panel helpers.

The standalone verifier cannot import skillevaluator, so the panel parsing,
model-family, and aggregation helpers live in a marked block that must stay
byte-for-byte identical to ``eval_core/llm_judge.py``.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_MODULE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "eval_core" / "llm_judge.py"
_EVAL_TEMPLATE = _REPO_ROOT / "src" / "skillevaluator" / "tier3" / "harbor" / "templates" / "eval.py"
_BLOCK_BEGIN = b"# --- BEGIN SHARED JUDGE PANEL HELPERS (verbatim copy in harbor/templates/eval.py) ---"
_BLOCK_END = b"# --- END SHARED JUDGE PANEL HELPERS ---"
_CONTRACT_NAMES = (
    b"class JudgeTarget(NamedTuple):",
    b"class JudgePanelSettings(NamedTuple):",
    b"_ACTIVE_JUDGE_TARGET: ContextVar[JudgeTarget | None] = ContextVar(",
    b"def parse_judge_panel_env(",
    b"def _model_family(",
    b"def aggregate_panel(",
)


def _shared_block(path: Path) -> bytes:
    """Return the marked block, both marker lines included, exactly as stored on disk."""
    lines = path.read_bytes().splitlines(keepends=True)
    begins = [index for index, line in enumerate(lines) if line.rstrip(b"\r\n") == _BLOCK_BEGIN]
    ends = [index for index, line in enumerate(lines) if line.rstrip(b"\r\n") == _BLOCK_END]
    assert len(begins) == 1, f"{path.name} must contain exactly one BEGIN marker line"
    assert len(ends) == 1, f"{path.name} must contain exactly one END marker line"
    assert begins[0] < ends[0], f"{path.name} has its END marker before its BEGIN marker"
    return b"".join(lines[begins[0] : ends[0] + 1])


@pytest.mark.parametrize("path", [_SHARED_MODULE, _EVAL_TEMPLATE], ids=["shared-module", "verifier-template"])
def test_each_copy_marks_one_block_with_the_contract_helpers(path: Path) -> None:
    block = _shared_block(path)

    assert all(name in block for name in _CONTRACT_NAMES)


def test_verifier_template_block_is_a_byte_for_byte_copy_of_the_shared_module() -> None:
    shared = _shared_block(_SHARED_MODULE)
    template = _shared_block(_EVAL_TEMPLATE)

    if template != shared:
        diff = "\n".join(
            difflib.unified_diff(
                shared.decode("utf-8", "replace").splitlines(),
                template.decode("utf-8", "replace").splitlines(),
                fromfile="eval_core/llm_judge.py",
                tofile="harbor/templates/eval.py",
                lineterm="",
            )
        )
        pytest.fail(f"Shared judge panel block drifted; copy it verbatim:\n{diff}")
