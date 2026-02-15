"""Aggregated scenario subsets combining building and non-building tasks."""

from craftext.dataset.scenarious.all_building.instructions import easy as building_easy, medium as building_medium
from craftext.dataset.scenarious.conditional_achievements.instructions import easy as cond_ach_easy, medium as cond_ach_medium
from craftext.dataset.scenarious.conditional_placing.instructions import easy as cond_place_easy, medium as cond_place_medium
from craftext.dataset.scenarious.explore.instructions import easy as explore_easy, medium as explore_medium
from craftext.dataset.scenarious.localization_place.instructions import easy as loc_easy, medium as loc_medium


def _merge_with_prefix(prefix: str, data: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_all(*parts: tuple[str, dict]) -> dict:
    merged = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


non_building_easy = _merge_all(
    ("conditional_achievements", cond_ach_easy),
    ("conditional_placing", cond_place_easy),
    ("explore", explore_easy),
    ("localization_place", loc_easy),
)

non_building_medium = _merge_all(
    ("conditional_achievements", cond_ach_medium),
    ("conditional_placing", cond_place_medium),
    ("explore", explore_medium),
    ("localization_place", loc_medium),
)


easy_only_building = dict(building_easy)
easy_without_building = dict(non_building_easy)

# Legacy canonical train splits.
easy = _merge_all(("building", building_easy), ("non_building", non_building_easy))
medium = _merge_all(("building", building_medium), ("non_building", non_building_medium))
