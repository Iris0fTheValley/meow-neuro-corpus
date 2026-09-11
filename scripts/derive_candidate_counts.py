from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from manifest_tools import ROOT


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def derive_candidate_counts(path: Path | None = None) -> dict[str, Any]:
    path = path or ROOT / "datasets" / "family_training_candidates_fusion_review.jsonl"
    grades = Counter()
    training_true = 0
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        grades[str(row.get("candidate_grade"))] += 1
        if row.get("training_candidate") is True:
            training_true += 1
    try:
        source = str(path.relative_to(ROOT))
    except ValueError:
        source = str(path)
    return {
        "row_count": rows,
        "grade_counts": dict(grades),
        "s_a_review_candidate_count": grades.get("S", 0) + grades.get("A", 0),
        "training_candidate_count": training_true,
        "source": source,
    }
