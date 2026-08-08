from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aethelis.runtime.single_step import SingleStepResult


def write_debug_trace(result: SingleStepResult, path: Path) -> Path:
    """Write a non-formal debug trace with safe runtime metadata only."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "trace_type": "debug",
        "formal_experiment_result": False,
        **result.safe_summary(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
