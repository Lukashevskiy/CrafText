"""Aggregated building-only scenario subsets used by legacy config keys."""

from craftext.dataset.scenarious.building.line.instructions import easy as line_easy, medium as line_medium
from craftext.dataset.scenarious.building.square.instructions import easy as square_easy, medium as square_medium
from craftext.dataset.scenarious.building.star.instructions import easy as star_easy, medium as star_medium


def _merge_with_prefix(prefix: str, data: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_building(*parts: tuple[str, dict]) -> dict:
    merged = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


easy = _merge_building(
    ("line", line_easy),
    ("square", square_easy),
    ("star", star_easy),
)

medium = _merge_building(
    ("line", line_medium),
    ("square", square_medium),
    ("star", star_medium),
)
