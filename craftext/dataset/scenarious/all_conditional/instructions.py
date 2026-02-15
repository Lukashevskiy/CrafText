"""Aggregated conditional scenario subsets."""

from craftext.dataset.scenarious.conditional_achievements.instructions import easy as cond_ach_easy, medium as cond_ach_medium
from craftext.dataset.scenarious.conditional_placing.instructions import easy as cond_place_easy, medium as cond_place_medium


def _merge_with_prefix(prefix: str, data: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_conditional(*parts: tuple[str, dict]) -> dict:
    merged = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


easy = _merge_conditional(
    ("achievements", cond_ach_easy),
    ("placing", cond_place_easy),
)

medium = _merge_conditional(
    ("achievements", cond_ach_medium),
    ("placing", cond_place_medium),
)
