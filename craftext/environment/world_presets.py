"""World preset helpers for CrafText/Craftax runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import pathlib
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from craftax.craftax_env import make_craftax_env_from_name

from craftext.environment.scenarious.loader import resolve_base_environment

WORLD_PRESET_CONFIG_DIR_NAME = "world_presets"


@dataclass(frozen=True)
class InventoryGrantSpec:
    item: str
    value: Any
    probability: float = 1.0


@dataclass(frozen=True)
class WorldPresetSpec:
    """Resolved world preset parameters."""

    name: str
    env_name: str
    seed: int
    map_size: Optional[Tuple[int, int]] = None
    blocked_block: Optional[str] = None
    floor_block: Optional[str] = None
    perimeter_block: Optional[str] = None
    disable_mob_spawns: Optional[bool] = None
    walk_through_perimeter_objects: Optional[bool] = None
    starting_inventory: Tuple[InventoryGrantSpec, ...] = ()
    starting_intrinsics: Tuple[InventoryGrantSpec, ...] = ()
    intrinsic_rates: Tuple[InventoryGrantSpec, ...] = ()
    intrinsic_thresholds: Tuple[InventoryGrantSpec, ...] = ()
    recovery_rules: Tuple[InventoryGrantSpec, ...] = ()
    map_rules: Tuple[InventoryGrantSpec, ...] = ()
    ring_inner_radius: Optional[int] = None
    ring_outer_radius: Optional[int] = None
    box_inner_size: Optional[int] = None
    perimeter_tree_prob: Optional[float] = None

    @property
    def has_ring(self) -> bool:
        return self.ring_outer_radius is not None

    @property
    def has_box(self) -> bool:
        return self.box_inner_size is not None


def get_world_preset_config_dir() -> pathlib.Path:
    """Return the package directory that stores YAML world preset configs."""
    module = importlib.import_module("craftext")
    module_path = pathlib.Path(inspect.getmodule(module).__path__[0])
    return module_path / WORLD_PRESET_CONFIG_DIR_NAME


def _find_world_preset_config_path(preset_name: str) -> pathlib.Path:
    config_dir = get_world_preset_config_dir()
    direct = config_dir / f"{preset_name}.yaml"
    if direct.exists():
        return direct

    slash_path = config_dir / f"{preset_name.replace('.', '/')}.yaml"
    if slash_path.exists():
        return slash_path

    matches = list(config_dir.glob(f"**/{preset_name}.yaml"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Multiple world preset configs named {preset_name}.yaml found: {matches}")

    raise FileNotFoundError(f"World preset config {preset_name} not found under {config_dir}")


def _load_world_preset_config(preset_name: str) -> dict[str, object]:
    config_path = _find_world_preset_config_path(preset_name)
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if config_data is None:
        raise ValueError(f"World preset config {preset_name} is empty: {config_path}")
    if not isinstance(config_data, dict):
        raise TypeError(f"World preset config {preset_name} must be a YAML mapping: {config_path}")

    return dict(config_data)


def looks_like_env_name(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return lowered.startswith("craftax") or lowered in {
        "classic",
        "classic_pixels",
        "classic-symbolic",
        "classic_symbolic",
        "full",
        "full_pixels",
        "pixels",
        "symbolic",
        "full-symbolic",
        "full_symbolic",
    }


def normalize_craftax_env_name(value: str) -> str:
    """Normalize user/config environment string into one Craftax env id."""
    normalized = value.strip()
    lowered = normalized.lower()

    alias_map = {
        "classic": "Craftax-Classic-Pixels-v1",
        "classic_pixels": "Craftax-Classic-Pixels-v1",
        "classic-symbolic": "Craftax-Classic-Symbolic-v1",
        "classic_symbolic": "Craftax-Classic-Symbolic-v1",
        "full": "Craftax-Pixels-v1",
        "pixels": "Craftax-Pixels-v1",
        "full_pixels": "Craftax-Pixels-v1",
        "symbolic": "Craftax-Symbolic-v1",
        "full-symbolic": "Craftax-Symbolic-v1",
        "full_symbolic": "Craftax-Symbolic-v1",
    }
    if lowered in alias_map:
        return alias_map[lowered]

    if normalized.endswith("-Text"):
        normalized = normalized[:-5]

    if normalized == "Classic":
        return "Craftax-Classic-Pixels-v1"

    return normalized


def _normalize_map_size(map_size: Optional[int | Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if map_size is None:
        return None
    if isinstance(map_size, int):
        return (map_size, map_size)
    return (int(map_size[0]), int(map_size[1]))


def _normalize_inventory_grants(starting_inventory: Any) -> Tuple[InventoryGrantSpec, ...]:
    if starting_inventory is None:
        return ()
    if not isinstance(starting_inventory, dict):
        raise TypeError("World preset 'starting_inventory' must be a mapping")

    grants: list[InventoryGrantSpec] = []
    for item_name, raw_value in starting_inventory.items():
        if not isinstance(item_name, str) or not item_name.strip():
            raise TypeError("World preset inventory item names must be non-empty strings")

        probability = 1.0
        value = raw_value
        if isinstance(raw_value, dict):
            if "value" in raw_value:
                value = raw_value["value"]
            else:
                value = raw_value.get("count")
            probability = raw_value.get("probability", 1.0)

        if value is None:
            raise ValueError(f"World preset inventory item {item_name} must define 'value' or 'count'")
        probability = float(probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"World preset inventory item {item_name} probability must be in [0, 1]")

        grants.append(InventoryGrantSpec(item=item_name.strip(), value=value, probability=probability))

    return tuple(grants)


def _normalize_named_grants(raw_mapping: Any, field_name: str) -> Tuple[InventoryGrantSpec, ...]:
    if raw_mapping is None:
        return ()
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"World preset '{field_name}' must be a mapping")
    return _normalize_inventory_grants(raw_mapping)


def _grants_to_mapping(grants: Tuple[InventoryGrantSpec, ...]) -> dict[str, Any]:
    return {grant.item: grant.value for grant in grants}


def build_world_preset_spec(
    *,
    env_name: str,
    preset_name: Optional[str],
    seed: Optional[int],
    map_size: Optional[int | Tuple[int, int]] = None,
    blocked_block: Optional[str] = None,
    floor_block: Optional[str] = None,
    perimeter_block: Optional[str] = None,
    disable_mob_spawns: Optional[bool] = None,
    walk_through_perimeter_objects: Optional[bool] = None,
    starting_inventory: Any = None,
    starting_intrinsics: Any = None,
    intrinsic_rates: Any = None,
    intrinsic_thresholds: Any = None,
    recovery_rules: Any = None,
    map_rules: Any = None,
    ring_inner_radius: Optional[int] = None,
    ring_outer_radius: Optional[int] = None,
    box_inner_size: Optional[int] = None,
    perimeter_tree_prob: Optional[float] = None,
    allow_config_lookup: bool = True,
) -> WorldPresetSpec:
    """Build a concrete world preset spec for the requested env."""
    if allow_config_lookup and preset_name and not looks_like_env_name(preset_name):
        try:
            preset_config = _load_world_preset_config(preset_name)
        except FileNotFoundError:
            preset_config = None
        if preset_config is not None:
            return build_world_preset_spec_from_config(
                preset_name=preset_name,
                config_data=preset_config,
                fallback_env_name=env_name,
                fallback_seed=seed,
                fallback_map_size=map_size,
                fallback_ring_inner_radius=ring_inner_radius,
                fallback_ring_outer_radius=ring_outer_radius,
                fallback_box_inner_size=box_inner_size,
                fallback_perimeter_tree_prob=perimeter_tree_prob,
            )

    normalized_env_name = normalize_craftax_env_name(env_name)
    normalized_preset = (preset_name or "default").strip().lower()
    normalized_map_size = _normalize_map_size(map_size)

    resolved_seed = int(seed) if seed is not None else (
        1 if normalized_preset in {"fixed", "ring_fixed"} else int(np.random.randint(0, 2**31 - 1))
    )

    if normalized_preset in {"ring", "ring_random"}:
        normalized_preset = "ring_random"
    if normalized_preset == "ring_fixed":
        normalized_preset = "ring_fixed"
    if normalized_preset == "box":
        normalized_preset = "box"
    if normalized_preset in {"box3", "box3_random_trees", "boxed_3x3_random_trees"}:
        normalized_preset = "box3_random_trees"
    if normalized_preset not in {
        "default",
        "random",
        "fixed",
        "ring_random",
        "ring_fixed",
        "box",
        "box3_random_trees",
    }:
        normalized_preset = "default"

    if normalized_preset.startswith("ring_"):
        ring_inner_radius = 0 if ring_inner_radius is None else int(ring_inner_radius)
        ring_outer_radius = 12 if ring_outer_radius is None else int(ring_outer_radius)
    else:
        ring_inner_radius = None
        ring_outer_radius = None

    if normalized_preset in {"box", "box3_random_trees"}:
        box_inner_size = 3 if box_inner_size is None else int(box_inner_size)
        perimeter_tree_prob = 0.7 if perimeter_tree_prob is None else float(perimeter_tree_prob)
        if disable_mob_spawns is None:
            disable_mob_spawns = True
        if walk_through_perimeter_objects is None:
            walk_through_perimeter_objects = True
    else:
        box_inner_size = None
        perimeter_tree_prob = None

    return WorldPresetSpec(
        name=normalized_preset,
        env_name=normalized_env_name,
        seed=resolved_seed,
        map_size=normalized_map_size,
        blocked_block=None if blocked_block is None else str(blocked_block),
        floor_block=None if floor_block is None else str(floor_block),
        perimeter_block=None if perimeter_block is None else str(perimeter_block),
        disable_mob_spawns=disable_mob_spawns,
        walk_through_perimeter_objects=walk_through_perimeter_objects,
        starting_inventory=_normalize_inventory_grants(starting_inventory),
        starting_intrinsics=_normalize_named_grants(starting_intrinsics, "starting_intrinsics"),
        intrinsic_rates=_normalize_named_grants(intrinsic_rates, "intrinsic_rates"),
        intrinsic_thresholds=_normalize_named_grants(intrinsic_thresholds, "intrinsic_thresholds"),
        recovery_rules=_normalize_named_grants(recovery_rules, "recovery_rules"),
        map_rules=_normalize_named_grants(map_rules, "map_rules"),
        ring_inner_radius=ring_inner_radius,
        ring_outer_radius=ring_outer_radius,
        box_inner_size=box_inner_size,
        perimeter_tree_prob=perimeter_tree_prob,
    )


def build_world_preset_spec_from_config(
    *,
    preset_name: str,
    config_data: dict[str, object],
    fallback_env_name: str,
    fallback_seed: Optional[int],
    fallback_map_size: Optional[int | Tuple[int, int]],
    fallback_ring_inner_radius: Optional[int],
    fallback_ring_outer_radius: Optional[int],
    fallback_box_inner_size: Optional[int],
    fallback_perimeter_tree_prob: Optional[float],
) -> WorldPresetSpec:
    """Build preset spec from YAML config, with optional inheritance."""
    normalized_fallback_env = normalize_craftax_env_name(fallback_env_name)
    extends_value = config_data.get("extends")
    inherited: Optional[WorldPresetSpec] = None
    if extends_value is not None:
        if not isinstance(extends_value, str) or not extends_value.strip():
            raise TypeError(f"World preset {preset_name}: 'extends' must be a non-empty string")
        inherited = build_world_preset_spec(
            env_name=fallback_env_name,
            preset_name=extends_value,
            seed=fallback_seed,
            map_size=fallback_map_size,
            ring_inner_radius=fallback_ring_inner_radius,
            ring_outer_radius=fallback_ring_outer_radius,
            box_inner_size=fallback_box_inner_size,
            perimeter_tree_prob=fallback_perimeter_tree_prob,
            allow_config_lookup=True,
        )

    preset_kind = config_data.get("preset")
    if preset_kind is None and inherited is not None:
        preset_kind = inherited.name
    if preset_kind is None:
        preset_kind = "default"
    if not isinstance(preset_kind, str):
        raise TypeError(f"World preset {preset_name}: 'preset' must be a string")

    declared_env_value = config_data.get("env_name", config_data.get("base_environment"))
    if declared_env_value is None and inherited is not None:
        declared_env_value = inherited.env_name
    if declared_env_value is not None and not isinstance(declared_env_value, str):
        raise TypeError(f"World preset {preset_name}: 'env_name' must be a string")
    if declared_env_value is not None:
        normalized_declared_env = normalize_craftax_env_name(declared_env_value)
        if normalized_declared_env != normalized_fallback_env:
            raise ValueError(
                f"World preset {preset_name} expects env_name={normalized_declared_env}, "
                f"but the active env is {normalized_fallback_env}. "
                "Presets only overlay the selected base environment and do not override it."
            )
    env_value = normalized_fallback_env

    seed_value = config_data.get("seed", inherited.seed if inherited is not None else fallback_seed)
    if seed_value is not None and not isinstance(seed_value, int):
        raise TypeError(f"World preset {preset_name}: 'seed' must be int")

    map_size_value = config_data.get("map_size", inherited.map_size if inherited is not None else fallback_map_size)
    blocked_block_value = config_data.get("blocked_block", inherited.blocked_block if inherited is not None else None)
    floor_block_value = config_data.get("floor_block", inherited.floor_block if inherited is not None else None)
    perimeter_block_value = config_data.get(
        "perimeter_block",
        inherited.perimeter_block if inherited is not None else None,
    )
    disable_mob_spawns_value = config_data.get(
        "disable_mob_spawns",
        inherited.disable_mob_spawns if inherited is not None else None,
    )
    walk_through_perimeter_objects_value = config_data.get(
        "walk_through_perimeter_objects",
        inherited.walk_through_perimeter_objects if inherited is not None else None,
    )
    starting_inventory_value = config_data.get(
        "starting_inventory",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.starting_inventory}
        if inherited is not None and inherited.starting_inventory
        else None,
    )
    starting_intrinsics_value = config_data.get(
        "starting_intrinsics",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.starting_intrinsics}
        if inherited is not None and inherited.starting_intrinsics
        else None,
    )
    intrinsic_rates_value = config_data.get(
        "intrinsic_rates",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.intrinsic_rates}
        if inherited is not None and inherited.intrinsic_rates
        else None,
    )
    intrinsic_thresholds_value = config_data.get(
        "intrinsic_thresholds",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.intrinsic_thresholds}
        if inherited is not None and inherited.intrinsic_thresholds
        else None,
    )
    recovery_rules_value = config_data.get(
        "recovery_rules",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.recovery_rules}
        if inherited is not None and inherited.recovery_rules
        else None,
    )
    map_rules_value = config_data.get(
        "map_rules",
        {grant.item: {"value": grant.value, "probability": grant.probability} for grant in inherited.map_rules}
        if inherited is not None and inherited.map_rules
        else None,
    )
    ring_inner_value = config_data.get(
        "ring_inner_radius",
        inherited.ring_inner_radius if inherited is not None else fallback_ring_inner_radius,
    )
    ring_outer_value = config_data.get(
        "ring_outer_radius",
        inherited.ring_outer_radius if inherited is not None else fallback_ring_outer_radius,
    )
    box_inner_value = config_data.get(
        "box_inner_size",
        inherited.box_inner_size if inherited is not None else fallback_box_inner_size,
    )
    perimeter_tree_prob_value = config_data.get(
        "perimeter_tree_prob",
        inherited.perimeter_tree_prob if inherited is not None else fallback_perimeter_tree_prob,
    )

    return build_world_preset_spec(
        env_name=env_value,
        preset_name=str(preset_kind),
        seed=seed_value,
        map_size=map_size_value,
        blocked_block=blocked_block_value,
        floor_block=floor_block_value,
        perimeter_block=perimeter_block_value,
        disable_mob_spawns=disable_mob_spawns_value,
        walk_through_perimeter_objects=walk_through_perimeter_objects_value,
        starting_inventory=starting_inventory_value,
        starting_intrinsics=starting_intrinsics_value,
        intrinsic_rates=intrinsic_rates_value,
        intrinsic_thresholds=intrinsic_thresholds_value,
        recovery_rules=recovery_rules_value,
        map_rules=map_rules_value,
        ring_inner_radius=ring_inner_value,
        ring_outer_radius=ring_outer_value,
        box_inner_size=box_inner_value,
        perimeter_tree_prob=perimeter_tree_prob_value,
        allow_config_lookup=False,
    )


def _build_static_env_params(env_name: str, map_size: Optional[Tuple[int, int]]) -> Optional[Any]:
    if map_size is None:
        return None

    resolved_family = resolve_base_environment(env_name).family
    if resolved_family == "classic":
        from craftax.craftax_classic.envs.craftax_state import StaticEnvParams
    else:
        from craftax.craftax.craftax_state import StaticEnvParams

    return StaticEnvParams(map_size=map_size)


def build_env_and_params(spec: WorldPresetSpec, *, auto_reset: bool = False) -> Tuple[Any, Any]:
    """Construct Craftax env instance and env params for a world preset."""
    static_env_params = _build_static_env_params(spec.env_name, spec.map_size)

    if static_env_params is None:
        env = make_craftax_env_from_name(spec.env_name, auto_reset=auto_reset)
    elif spec.env_name == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
            CraftaxClassicPixelsEnvNoAutoReset,
        )
        env = CraftaxClassicPixelsEnv(static_env_params) if auto_reset else CraftaxClassicPixelsEnvNoAutoReset(static_env_params)
    elif spec.env_name == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
            CraftaxClassicSymbolicEnvNoAutoReset,
        )
        env = (
            CraftaxClassicSymbolicEnv(static_env_params)
            if auto_reset
            else CraftaxClassicSymbolicEnvNoAutoReset(static_env_params)
        )
    elif spec.env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv, CraftaxPixelsEnvNoAutoReset
        env = CraftaxPixelsEnv(static_env_params) if auto_reset else CraftaxPixelsEnvNoAutoReset(static_env_params)
    elif spec.env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv, CraftaxSymbolicEnvNoAutoReset
        env = CraftaxSymbolicEnv(static_env_params) if auto_reset else CraftaxSymbolicEnvNoAutoReset(static_env_params)
    else:
        raise ValueError(f"Unsupported Craftax environment for world preset: {spec.env_name}")

    env_params = env.default_params
    if spec.disable_mob_spawns:
        replace_kwargs = {"mob_despawn_distance": 0}
        if hasattr(env_params, "spawn_cow_chance"):
            replace_kwargs["spawn_cow_chance"] = 0.0
        if hasattr(env_params, "spawn_zombie_base_chance"):
            replace_kwargs["spawn_zombie_base_chance"] = 0.0
        if hasattr(env_params, "spawn_zombie_night_chance"):
            replace_kwargs["spawn_zombie_night_chance"] = 0.0
        if hasattr(env_params, "spawn_skeleton_chance"):
            replace_kwargs["spawn_skeleton_chance"] = 0.0
        env_params = env_params.replace(**replace_kwargs)

    if spec.has_ring or spec.has_box:
        env = WorldPresetAdapter(env, spec)
    if (
        spec.starting_inventory
        or spec.starting_intrinsics
        or spec.intrinsic_rates
        or spec.intrinsic_thresholds
    ):
        env = CharacterStateAdapter(env, spec)
    if spec.recovery_rules:
        env = RecoveryStateAdapter(env, spec)
    if spec.map_rules:
        env = MapStateAdapter(env, spec)

    return env, env_params


def _distance_mask(height: int, width: int, center_x: int, center_y: int, inner_radius: int, outer_radius: int) -> jnp.ndarray:
    xs = jnp.arange(height)[:, None]
    ys = jnp.arange(width)[None, :]
    dist2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
    return jnp.logical_and(dist2 >= inner_radius**2, dist2 <= outer_radius**2)


def _apply_ring_to_level(
    level_map: jnp.ndarray,
    *,
    player_position: jnp.ndarray,
    blocked_value: int,
    spawn_value: int,
    inner_radius: int,
    outer_radius: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    height, width = level_map.shape
    center_x = height // 2
    center_y = width // 2
    ring_mask = _distance_mask(height, width, center_x, center_y, inner_radius, outer_radius)

    new_map = jnp.where(ring_mask, level_map, blocked_value)
    spawn_x = center_x
    spawn_y = center_y + min(outer_radius, width // 2 - 2)
    if inner_radius > 0:
        spawn_y = center_y + min(max(inner_radius + 1, 1), width // 2 - 2)
    spawn_position = jnp.array([spawn_x, spawn_y], dtype=jnp.int32)
    new_map = new_map.at[spawn_position[0], spawn_position[1]].set(spawn_value)
    return new_map, spawn_position


def _apply_box_to_level(
    level_map: jnp.ndarray,
    *,
    key: Any,
    blocked_value: int,
    floor_value: int,
    tree_value: int,
    inner_size: int,
    perimeter_tree_prob: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    height, width = level_map.shape
    center_x = height // 2
    center_y = width // 2
    half = inner_size // 2

    inner_x0 = center_x - half
    inner_x1 = inner_x0 + inner_size
    inner_y0 = center_y - half
    inner_y1 = inner_y0 + inner_size

    base_map = jnp.full_like(level_map, blocked_value)
    base_map = base_map.at[inner_x0:inner_x1, inner_y0:inner_y1].set(floor_value)

    inner_height = inner_x1 - inner_x0
    inner_width = inner_y1 - inner_y0
    xs = jnp.arange(inner_height)[:, None]
    ys = jnp.arange(inner_width)[None, :]
    inner_perimeter_mask = jnp.logical_or(
        jnp.logical_or(xs == 0, xs == inner_height - 1),
        jnp.logical_or(ys == 0, ys == inner_width - 1),
    )

    random_values = jax.random.uniform(key, shape=(inner_height, inner_width))
    tree_mask = jnp.logical_and(inner_perimeter_mask, random_values < perimeter_tree_prob)
    inner_slice = base_map[inner_x0:inner_x1, inner_y0:inner_y1]
    inner_slice = jnp.where(tree_mask, tree_value, inner_slice)
    base_map = base_map.at[inner_x0:inner_x1, inner_y0:inner_y1].set(inner_slice)

    spawn_position = jnp.array([center_x, center_y], dtype=jnp.int32)
    base_map = base_map.at[spawn_position[0], spawn_position[1]].set(floor_value)
    return base_map, spawn_position


def _resolve_classic_block_value(block_name: Optional[str], default_value: int) -> int:
    if not block_name:
        return default_value
    from craftax.craftax_classic.constants import BlockType

    normalized = str(block_name).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return BlockType[normalized].value
    except KeyError as exc:
        raise ValueError(f"Unknown classic block name for world preset: {block_name}") from exc


def _resolve_full_block_value(block_name: Optional[str], default_value: int) -> int:
    if not block_name:
        return default_value
    from craftax.craftax.constants import BlockType

    normalized = str(block_name).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return BlockType[normalized].value
    except KeyError as exc:
        raise ValueError(f"Unknown full block name for world preset: {block_name}") from exc


def _box_walkable_mask(level_shape: tuple[int, int], inner_size: int) -> jnp.ndarray:
    height, width = level_shape
    center_x = height // 2
    center_y = width // 2
    half = inner_size // 2
    inner_x0 = center_x - half
    inner_x1 = inner_x0 + inner_size
    inner_y0 = center_y - half
    inner_y1 = inner_y0 + inner_size

    mask = jnp.zeros(level_shape, dtype=bool)
    return mask.at[inner_x0:inner_x1, inner_y0:inner_y1].set(True)


def _box_perimeter_mask(level_shape: tuple[int, int], inner_size: int) -> jnp.ndarray:
    height, width = level_shape
    center_x = height // 2
    center_y = width // 2
    half = inner_size // 2
    inner_x0 = center_x - half
    inner_x1 = inner_x0 + inner_size
    inner_y0 = center_y - half
    inner_y1 = inner_y0 + inner_size

    ring_x0 = max(inner_x0 - 1, 0)
    ring_x1 = min(inner_x1 + 1, height)
    ring_y0 = max(inner_y0 - 1, 0)
    ring_y1 = min(inner_y1 + 1, width)

    mask = jnp.zeros(level_shape, dtype=bool)
    mask = mask.at[ring_x0:ring_x1, ring_y0:ring_y1].set(True)
    mask = mask.at[inner_x0:inner_x1, inner_y0:inner_y1].set(False)
    return mask


def _ring_walkable_mask(level_shape: tuple[int, int], inner_radius: int, outer_radius: int) -> jnp.ndarray:
    height, width = level_shape
    center_x = height // 2
    center_y = width // 2
    return _distance_mask(height, width, center_x, center_y, inner_radius, outer_radius)


def _coerce_record_value(current_value: Any, desired_value: Any):
    if isinstance(current_value, jnp.ndarray):
        if isinstance(desired_value, (list, tuple)):
            array_value = jnp.asarray(desired_value, dtype=current_value.dtype)
        else:
            array_value = jnp.full(current_value.shape, desired_value, dtype=current_value.dtype)
        if array_value.shape != current_value.shape:
            raise ValueError(
                f"World preset array shape mismatch: expected {current_value.shape}, got {array_value.shape}"
            )
        return array_value

    if isinstance(desired_value, (list, tuple, dict)):
        raise TypeError("Scalar fields must use scalar values")
    return type(current_value)(desired_value)


def _apply_grants_to_record(
    *,
    record: Any,
    grants: Tuple[InventoryGrantSpec, ...],
    key: Any,
    context_name: str,
):
    updated_record = record
    rng = key
    for grant in grants:
        if not hasattr(updated_record, grant.item):
            raise ValueError(f"World preset item {grant.item!r} does not exist in {context_name}")

        rng, sample_rng = jax.random.split(rng)
        include_item = jax.random.uniform(sample_rng) < float(grant.probability)
        current_value = getattr(updated_record, grant.item)
        desired_value = _coerce_record_value(current_value=current_value, desired_value=grant.value)
        final_value = jax.tree_util.tree_map(
            lambda desired, current: jax.lax.select(include_item, desired, current),
            desired_value,
            current_value,
        )
        updated_record = updated_record.replace(**{grant.item: final_value})
    return updated_record


class WorldPresetAdapter:
    """Adapter that post-processes the reset state into a custom preset world."""

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        obs, state = self.env.reset(key, params)
        state = self._apply_preset(state, key)
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(state)
        return obs, state

    def step(self, key: Any, state: Any, action: int, params: Any):
        return self.env.step(key, state, action, params)

    def _apply_preset(self, state: Any, key: Any):
        resolved_family = resolve_base_environment(self.spec.env_name).family
        if resolved_family == "classic":
            from craftax.craftax_classic.constants import BlockType
            if self.spec.has_box:
                blocked_value = _resolve_classic_block_value(
                    self.spec.blocked_block,
                    BlockType.OUT_OF_BOUNDS.value,
                )
                floor_value = _resolve_classic_block_value(
                    self.spec.floor_block,
                    BlockType.GRASS.value,
                )
                perimeter_value = _resolve_classic_block_value(
                    self.spec.perimeter_block,
                    BlockType.TREE.value,
                )
                updated_map, spawn_position = _apply_box_to_level(
                    state.map,
                    key=key,
                    blocked_value=blocked_value,
                    floor_value=floor_value,
                    tree_value=perimeter_value,
                    inner_size=int(self.spec.box_inner_size),
                    perimeter_tree_prob=float(self.spec.perimeter_tree_prob or 0.7),
                )
                return state.replace(map=updated_map, player_position=spawn_position)

            blocked_value = _resolve_classic_block_value(
                self.spec.blocked_block,
                BlockType.WATER.value,
            )
            floor_value = _resolve_classic_block_value(
                self.spec.floor_block,
                BlockType.GRASS.value,
            )
            updated_map, spawn_position = _apply_ring_to_level(
                state.map,
                player_position=state.player_position,
                blocked_value=blocked_value,
                spawn_value=floor_value,
                inner_radius=int(self.spec.ring_inner_radius or 0),
                outer_radius=int(self.spec.ring_outer_radius),
            )
            return state.replace(map=updated_map, player_position=spawn_position)

        from craftax.craftax.constants import BlockType, ItemType

        current_level = int(state.player_level)
        current_map = state.map[current_level]
        if self.spec.has_box:
            default_floor_value = BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value
            default_perimeter_value = BlockType.TREE.value if current_level == 0 else BlockType.WALL.value
            blocked_value = _resolve_full_block_value(
                self.spec.blocked_block,
                BlockType.OUT_OF_BOUNDS.value,
            )
            floor_value = _resolve_full_block_value(self.spec.floor_block, default_floor_value)
            perimeter_value = _resolve_full_block_value(self.spec.perimeter_block, default_perimeter_value)
            updated_level_map, spawn_position = _apply_box_to_level(
                current_map,
                key=key,
                blocked_value=blocked_value,
                floor_value=floor_value,
                tree_value=perimeter_value,
                inner_size=int(self.spec.box_inner_size),
                perimeter_tree_prob=float(self.spec.perimeter_tree_prob or 0.7),
            )
            updated_map = state.map.at[current_level].set(updated_level_map)
            updated_item_map = state.item_map.at[current_level].set(jnp.zeros_like(state.item_map[current_level]))
            updated_mob_map = state.mob_map.at[current_level].set(jnp.zeros_like(state.mob_map[current_level]))
            updated_light_map = state.light_map.at[current_level].set(jnp.where(updated_level_map == BlockType.OUT_OF_BOUNDS.value, 0.0, state.light_map[current_level]))
            return state.replace(
                map=updated_map,
                item_map=updated_item_map,
                mob_map=updated_mob_map,
                light_map=updated_light_map,
                player_position=spawn_position,
            )

        blocked_value = _resolve_full_block_value(
            self.spec.blocked_block,
            BlockType.WATER.value if current_level == 0 else BlockType.WALL.value,
        )
        floor_value = _resolve_full_block_value(
            self.spec.floor_block,
            BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value,
        )
        updated_level_map, spawn_position = _apply_ring_to_level(
            current_map,
            player_position=state.player_position,
            blocked_value=blocked_value,
            spawn_value=floor_value,
            inner_radius=int(self.spec.ring_inner_radius or 0),
            outer_radius=int(self.spec.ring_outer_radius),
        )
        ring_mask = _distance_mask(
            current_map.shape[0],
            current_map.shape[1],
            current_map.shape[0] // 2,
            current_map.shape[1] // 2,
            int(self.spec.ring_inner_radius or 0),
            int(self.spec.ring_outer_radius),
        )

        updated_map = state.map.at[current_level].set(updated_level_map)
        updated_item_map = state.item_map.at[current_level].set(jnp.where(ring_mask, state.item_map[current_level], ItemType.NONE.value))
        updated_mob_map = state.mob_map.at[current_level].set(jnp.where(ring_mask, state.mob_map[current_level], 0))
        updated_light_map = state.light_map.at[current_level].set(jnp.where(ring_mask, state.light_map[current_level], 0.0))

        return state.replace(
            map=updated_map,
            item_map=updated_item_map,
            mob_map=updated_mob_map,
            light_map=updated_light_map,
            player_position=spawn_position,
        )


class CharacterStateAdapter:
    """Adapter that patches starting character state and applies custom intrinsic dynamics."""

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec
        self.rate_values = _grants_to_mapping(spec.intrinsic_rates)
        self.threshold_values = _grants_to_mapping(spec.intrinsic_thresholds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        obs, state = self.env.reset(key, params)
        if self.spec.starting_inventory:
            inventory = _apply_grants_to_record(
                record=state.inventory,
                grants=self.spec.starting_inventory,
                key=key,
                context_name=f"inventory for env {self.spec.env_name}",
            )
            state = state.replace(inventory=inventory)
        if self.spec.starting_intrinsics:
            state = self._apply_state_grants(state, self.spec.starting_intrinsics, key, "starting_intrinsics")
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(state)
        return obs, state

    def step(self, key: Any, state: Any, action: int, params: Any):
        obs, new_state, reward, done, info = self.env.step(key, state, action, params)
        new_state = self._apply_custom_intrinsic_dynamics(new_state)
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(new_state)
        return obs, new_state, reward, done, info

    def _apply_state_grants(self, state: Any, grants: Tuple[InventoryGrantSpec, ...], key: Any, label: str):
        rng = key
        updates: dict[str, Any] = {}
        for grant in grants:
            if not hasattr(state, grant.item):
                raise ValueError(
                    f"World preset {label} item {grant.item!r} does not exist for env {self.spec.env_name}"
                )
            rng, sample_rng = jax.random.split(rng)
            include_item = jax.random.uniform(sample_rng) < float(grant.probability)
            current_value = getattr(state, grant.item)
            desired_value = _coerce_record_value(current_value=current_value, desired_value=grant.value)
            final_value = jax.tree_util.tree_map(
                lambda desired, current: jax.lax.select(include_item, desired, current),
                desired_value,
                current_value,
            )
            updates[grant.item] = final_value
        return state.replace(**updates) if updates else state

    def _apply_custom_intrinsic_dynamics(self, state: Any):
        if not self.rate_values and not self.threshold_values:
            return state

        updates: dict[str, Any] = {}

        def _value(name: str, default: float) -> float:
            raw = self.rate_values.get(name, default)
            return float(raw)

        def _threshold(name: str, default: float) -> float:
            raw = self.threshold_values.get(name, default)
            return float(raw)

        def _process_meter(counter_name: str, resource_name: str, rate_name: str, threshold_name: str, default_threshold: float):
            if not hasattr(state, counter_name) or not hasattr(state, resource_name):
                return
            rate = _value(rate_name, 0.0)
            threshold = max(_threshold(threshold_name, default_threshold), 1e-6)
            if rate == 0.0:
                return
            counter = getattr(state, counter_name) + rate
            ticks = jnp.floor(jnp.maximum(counter, 0.0) / threshold).astype(jnp.int32)
            new_counter = counter - ticks.astype(counter.dtype) * threshold
            resource = jnp.maximum(getattr(state, resource_name) - ticks, 0)
            updates[counter_name] = new_counter
            updates[resource_name] = resource

        _process_meter("player_hunger", "player_food", "player_hunger_rate", "player_hunger_threshold", 25.0)
        _process_meter("player_thirst", "player_drink", "player_thirst_rate", "player_thirst_threshold", 20.0)

        if hasattr(state, "player_fatigue") and hasattr(state, "player_energy"):
            awake_rate = _value("player_fatigue_rate", 0.0)
            sleep_rate = _value("player_fatigue_sleep_rate", -abs(awake_rate) if awake_rate != 0.0 else 0.0)
            fatigue_delta = jax.lax.select(getattr(state, "is_sleeping", False), sleep_rate, awake_rate)
            if fatigue_delta != 0.0:
                fatigue = state.player_fatigue + fatigue_delta
                drop_threshold = max(_threshold("player_fatigue_threshold", 30.0), 1e-6)
                recover_threshold = min(_threshold("player_fatigue_recover_threshold", -10.0), -1e-6)
                drain_ticks = jnp.floor(jnp.maximum(fatigue, 0.0) / drop_threshold).astype(jnp.int32)
                fatigue = fatigue - drain_ticks.astype(fatigue.dtype) * drop_threshold
                energy = jnp.maximum(state.player_energy - drain_ticks, 0)
                recover_ticks = jnp.floor(jnp.maximum(-fatigue, 0.0) / abs(recover_threshold)).astype(jnp.int32)
                fatigue = fatigue + recover_ticks.astype(fatigue.dtype) * abs(recover_threshold)
                energy = jnp.minimum(energy + recover_ticks, 9)
                updates["player_fatigue"] = fatigue
                updates["player_energy"] = energy

        if hasattr(state, "player_recover") and hasattr(state, "player_health"):
            positive_rate = _value("player_recover_rate", 0.0)
            negative_rate = _value("player_recover_penalty_rate", -abs(positive_rate) if positive_rate != 0.0 else 0.0)
            has_food = getattr(state, "player_food", 1) > 0
            has_drink = getattr(state, "player_drink", 1) > 0
            has_energy = jnp.logical_or(getattr(state, "player_energy", 1) > 0, getattr(state, "is_sleeping", False))
            all_necessities = jnp.logical_and(jnp.logical_and(has_food, has_drink), has_energy)
            recover_delta = jax.lax.select(all_necessities, positive_rate, negative_rate)
            if recover_delta != 0.0:
                recover = state.player_recover + recover_delta
                pos_threshold = max(_threshold("player_recover_positive_threshold", 25.0), 1e-6)
                neg_threshold = min(_threshold("player_recover_negative_threshold", -15.0), -1e-6)
                heal_ticks = jnp.floor(jnp.maximum(recover, 0.0) / pos_threshold).astype(jnp.int32)
                recover = recover - heal_ticks.astype(recover.dtype) * pos_threshold
                health = jnp.minimum(state.player_health + heal_ticks, 9)
                hurt_ticks = jnp.floor(jnp.maximum(-recover, 0.0) / abs(neg_threshold)).astype(jnp.int32)
                recover = recover + hurt_ticks.astype(recover.dtype) * abs(neg_threshold)
                health = jnp.maximum(health - hurt_ticks, 0)
                updates["player_recover"] = recover
                updates["player_health"] = health

        return state.replace(**updates) if updates else state


class RecoveryStateAdapter:
    """Adapter that applies preset-defined recovery overrides for sleep/rest."""

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec
        self.rules = _grants_to_mapping(spec.recovery_rules)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        return self.env.reset(key, params)

    def step(self, key: Any, state: Any, action: int, params: Any):
        obs, new_state, reward, done, info = self.env.step(key, state, action, params)
        new_state = self._apply_recovery_rules(new_state)
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(new_state)
        return obs, new_state, reward, done, info

    def _apply_recovery_rules(self, state: Any):
        updates: dict[str, Any] = {}
        instant_sleep_enabled = bool(self.rules.get("instant_sleep_recovery", False))
        instant_rest_enabled = bool(self.rules.get("instant_rest_recovery", False))

        if instant_sleep_enabled and getattr(state, "is_sleeping", False):
            self._apply_recovery_mode_updates(
                state=state,
                updates=updates,
                prefix="sleep",
            )
            if bool(self.rules.get("wake_after_sleep_recovery", True)) and hasattr(state, "is_sleeping"):
                updates["is_sleeping"] = False

        if instant_rest_enabled and getattr(state, "is_resting", False):
            self._apply_recovery_mode_updates(
                state=state,
                updates=updates,
                prefix="rest",
            )
            if bool(self.rules.get("stop_rest_after_recovery", True)) and hasattr(state, "is_resting"):
                updates["is_resting"] = False

        return state.replace(**updates) if updates else state

    def _apply_recovery_mode_updates(self, *, state: Any, updates: dict[str, Any], prefix: str):
        if hasattr(state, "player_energy"):
            energy_target = self.rules.get(f"{prefix}_energy_value", 9)
            updates["player_energy"] = jnp.asarray(energy_target, dtype=getattr(state.player_energy, "dtype", None))
        if hasattr(state, "player_fatigue"):
            fatigue_target = self.rules.get(f"{prefix}_fatigue_value", 0.0)
            updates["player_fatigue"] = jnp.asarray(fatigue_target, dtype=getattr(state.player_fatigue, "dtype", None))
        if hasattr(state, "player_mana") and f"{prefix}_mana_value" in self.rules:
            updates["player_mana"] = jnp.asarray(
                self.rules[f"{prefix}_mana_value"],
                dtype=getattr(state.player_mana, "dtype", None),
            )
        if hasattr(state, "player_recover") and f"{prefix}_recover_value" in self.rules:
            updates["player_recover"] = jnp.asarray(
                self.rules[f"{prefix}_recover_value"],
                dtype=getattr(state.player_recover, "dtype", None),
            )
        if hasattr(state, "player_health") and f"{prefix}_health_value" in self.rules:
            updates["player_health"] = jnp.asarray(
                self.rules[f"{prefix}_health_value"],
                dtype=getattr(state.player_health, "dtype", None),
            )


class MapStateAdapter:
    """Adapter that enforces preset-defined solid map blocks without patching Craftax."""

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec
        self.rules = _grants_to_mapping(spec.map_rules)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        return self.env.reset(key, params)

    def step(self, key: Any, state: Any, action: int, params: Any):
        obs, new_state, reward, done, info = self.env.step(key, state, action, params)
        new_state = self._apply_solid_block_rules(previous_state=state, new_state=new_state)
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(new_state)
        return obs, new_state, reward, done, info

    def _apply_solid_block_rules(self, *, previous_state: Any, new_state: Any):
        solid_values = self._resolve_solid_block_values()
        if not solid_values:
            return new_state

        resolved_family = resolve_base_environment(self.spec.env_name).family
        if resolved_family == "classic":
            pos = new_state.player_position
            current_block = int(new_state.map[pos[0], pos[1]])
        else:
            pos = new_state.player_position
            level = int(new_state.player_level)
            current_block = int(new_state.map[level, pos[0], pos[1]])

        if current_block not in solid_values:
            return new_state
        return new_state.replace(player_position=previous_state.player_position)

    def _resolve_solid_block_values(self) -> set[int]:
        resolved_family = resolve_base_environment(self.spec.env_name).family
        solid_values: set[int] = set()
        if bool(self.rules.get("solid_out_of_bounds", False)):
            if resolved_family == "classic":
                from craftax.craftax_classic.constants import BlockType
            else:
                from craftax.craftax.constants import BlockType
            solid_values.add(int(BlockType.OUT_OF_BOUNDS.value))

        extra_blocks = self.rules.get("solid_blocks", [])
        if isinstance(extra_blocks, str):
            extra_blocks = [extra_blocks]
        if extra_blocks:
            if resolved_family == "classic":
                resolver = _resolve_classic_block_value
            else:
                resolver = _resolve_full_block_value
            for block_name in extra_blocks:
                solid_values.add(int(resolver(str(block_name), 0)))

        return solid_values
