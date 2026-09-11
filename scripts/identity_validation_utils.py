from __future__ import annotations

import math
import random
from collections.abc import Iterable


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (None, None)  # type: ignore[return-value]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_interval(values: Iterable[float], seed: int = 0, draws: int = 2000) -> tuple[float, float]:
    values = list(values)
    if not values:
        return (None, None)  # type: ignore[return-value]
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))])


def fail_closed_candidate_count(rows: Iterable[dict]) -> int:
    """Count training candidates only when the row-level field is explicitly true."""
    return sum(1 for row in rows if row.get("training_candidate") is True)


def assert_training_gate(rows: Iterable[dict], prerequisites_pass: bool) -> None:
    """The training gate is closed unless every prerequisite is proven."""
    if not prerequisites_pass and fail_closed_candidate_count(rows) != 0:
        raise AssertionError("training_candidate must remain zero when prerequisites are incomplete")
