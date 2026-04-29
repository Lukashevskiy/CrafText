"""World preset helpers for CrafText/Craftax runtime."""

from dataclasses import dataclass
import importlib
import inspect
import pathlib
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from craftax.craftax_env import make_craftax_env_from_name

from craftext.environment.scenarious.loader import resolve_base_environment

WORLD_PRESET_CONFIG_DIR_NAME = "world_presets"
BUILTIN_PRESET_ALIASES = {
    "ring": "ring_random",
    "ring_random": "ring_random",
    "ring_fixed": "ring_fixed",
    "box": "box",
    "box3": "box3_random_trees",
    "box3_random_trees": "box3_random_trees",
    "boxed_3x3_random_trees": "box3_random_trees",
}
BUILTIN_PRESET_NAMES = {
    "default",
    "random",
    "fixed",
    "ring_random",
    "ring_fixed",
    "box",
    "box3_random_trees",
}


@dataclass(frozen=True)
class InventoryGrantSpec:
    """Probabilistic override for one named field.

    Attributes:
        item: Target field name in inventory, state, or rule mapping.
        value: Value to assign when the override is sampled.
        probability: Probability of applying the override on reset.
    """

    item: str
    value: Any
    probability: float = 1.0


@dataclass(frozen=True)
class GeneratedWorldState:
    """Container for generator-produced world state overlays.

    Attributes:
        map: Generated map tensor or per-level map tensor.
        player_position: Spawn or replacement player position.
        item_map: Optional item-layer replacement for full Craftax.
        mob_map: Optional mob-layer replacement for full Craftax.
        light_map: Optional light-layer replacement for full Craftax.
    """

    map: Any
    player_position: Any
    item_map: Any = None
    mob_map: Any = None
    light_map: Any = None


@dataclass(frozen=True)
class EnvPresetSpec:
    """Environment-level preset configuration.

    Attributes:
        env_name: Concrete Craftax environment id.
        seed: Seed used for preset construction and resets.
        map_size: Optional static square map size override.
        disable_mob_spawns: Whether to zero-out spawn probabilities in env params.
    """

    env_name: str
    seed: int
    map_size: Optional[Tuple[int, int]] = None
    disable_mob_spawns: Optional[bool] = None


@dataclass(frozen=True)
class MapPresetSpec:
    """Map generation and map-behavior configuration.

    Attributes:
        generator: Registered generator name such as ``box`` or ``ring``.
        behaviors: Registered map behavior names to apply after generation.
        blocked_block: Block name used outside generated playable area.
        floor_block: Block name used for playable floor cells.
        perimeter_block: Block name used for box perimeter placement.
        rules: Rule mapping consumed by map behaviors.
        ring_inner_radius: Inner radius for ring generation.
        ring_outer_radius: Outer radius for ring generation.
        box_inner_size: Side length for box generation.
        perimeter_tree_prob: Sampling probability for perimeter decoration.
    """

    generator: Optional[str] = None
    behaviors: Tuple[str, ...] = ()
    blocked_block: Optional[str] = None
    floor_block: Optional[str] = None
    perimeter_block: Optional[str] = None
    rules: Tuple[InventoryGrantSpec, ...] = ()
    ring_inner_radius: Optional[int] = None
    ring_outer_radius: Optional[int] = None
    box_inner_size: Optional[int] = None
    perimeter_tree_prob: Optional[float] = None


@dataclass(frozen=True)
class CharacterPresetSpec:
    """Character reset and dynamics configuration.

    Attributes:
        behaviors: Registered character behavior names.
        starting_inventory: Inventory overrides applied on reset.
        starting_intrinsics: Direct state-field overrides applied on reset.
        intrinsic_rates: Post-step intrinsic delta configuration.
        intrinsic_thresholds: Threshold configuration for intrinsic dynamics.
    """

    behaviors: Tuple[str, ...] = ()
    starting_inventory: Tuple[InventoryGrantSpec, ...] = ()
    starting_intrinsics: Tuple[InventoryGrantSpec, ...] = ()
    intrinsic_rates: Tuple[InventoryGrantSpec, ...] = ()
    intrinsic_thresholds: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class RecoveryPresetSpec:
    """Recovery policy configuration.

    Attributes:
        behaviors: Registered recovery behavior names.
        rules: Rule mapping consumed by recovery behaviors.
    """

    behaviors: Tuple[str, ...] = ()
    rules: Tuple[InventoryGrantSpec, ...] = ()


@dataclass(frozen=True)
class WorldPresetSpec:
    """Resolved top-level world preset configuration.

    Attributes:
        name: User-facing preset name or builtin alias.
        env_name: Compatibility field mirroring ``env.env_name``.
        seed: Compatibility field mirroring ``env.seed``.
        env: Environment-specific preset settings.
        map: Map generation and collision-policy settings.
        character: Character reset and dynamics settings.
        recovery: Recovery behavior settings.
    """

    name: str
    env_name: str
    seed: int
    env: EnvPresetSpec
    map: MapPresetSpec
    character: CharacterPresetSpec
    recovery: RecoveryPresetSpec

    @property
    def has_ring(self) -> bool:
        return self.map.generator == "ring"

    @property
    def has_box(self) -> bool:
        return self.map.generator == "box"

    @property
    def has_map_overlay(self) -> bool:
        return self.map.generator is not None

    @property
    def uses_map_adapter(self) -> bool:
        return self.has_map_overlay or bool(self.map.behaviors)

    @property
    def uses_character_adapter(self) -> bool:
        return bool(self.character.behaviors)

    @property
    def uses_recovery_adapter(self) -> bool:
        return bool(self.recovery.behaviors)

    @property
    def generator(self) -> Optional[str]:
        return self.map.generator

    @property
    def map_behaviors(self) -> Tuple[str, ...]:
        return self.map.behaviors

    @property
    def character_behaviors(self) -> Tuple[str, ...]:
        return self.character.behaviors

    @property
    def recovery_behaviors(self) -> Tuple[str, ...]:
        return self.recovery.behaviors

    @property
    def map_size(self) -> Optional[Tuple[int, int]]:
        return self.env.map_size

    @property
    def disable_mob_spawns(self) -> Optional[bool]:
        return self.env.disable_mob_spawns

    @property
    def blocked_block(self) -> Optional[str]:
        return self.map.blocked_block

    @property
    def floor_block(self) -> Optional[str]:
        return self.map.floor_block

    @property
    def perimeter_block(self) -> Optional[str]:
        return self.map.perimeter_block

    @property
    def map_rules(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.map.rules

    @property
    def ring_inner_radius(self) -> Optional[int]:
        return self.map.ring_inner_radius

    @property
    def ring_outer_radius(self) -> Optional[int]:
        return self.map.ring_outer_radius

    @property
    def box_inner_size(self) -> Optional[int]:
        return self.map.box_inner_size

    @property
    def perimeter_tree_prob(self) -> Optional[float]:
        return self.map.perimeter_tree_prob

    @property
    def starting_inventory(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.character.starting_inventory

    @property
    def starting_intrinsics(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.character.starting_intrinsics

    @property
    def intrinsic_rates(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.character.intrinsic_rates

    @property
    def intrinsic_thresholds(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.character.intrinsic_thresholds

    @property
    def recovery_rules(self) -> Tuple[InventoryGrantSpec, ...]:
        return self.recovery.rules


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


def _normalize_builtin_preset_name(preset_name: Optional[str]) -> str:
    normalized_preset = (preset_name or "default").strip().lower()
    normalized_preset = BUILTIN_PRESET_ALIASES.get(normalized_preset, normalized_preset)
    if normalized_preset not in BUILTIN_PRESET_NAMES:
        return "default"
    return normalized_preset


def _normalize_name_list(raw_value: Any, *, field_name: str) -> Tuple[str, ...]:
    if raw_value is None:
        return ()
    values = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"World preset '{field_name}' entries must be non-empty strings")
        normalized.append(value.strip())
    return tuple(normalized)


def _derive_generator_name(
    *,
    explicit_generator: Optional[str],
    normalized_preset: str,
) -> Optional[str]:
    if explicit_generator is not None:
        normalized = explicit_generator.strip().lower()
        if normalized in {"", "none", "default"}:
            return None
        return normalized
    if normalized_preset in {"box", "box3_random_trees"}:
        return "box"
    if normalized_preset.startswith("ring_"):
        return "ring"
    return None


def _derive_map_behavior_names(
    *,
    explicit_names: Optional[Any],
    map_rules: Any,
) -> Tuple[str, ...]:
    if explicit_names is not None:
        return _normalize_name_list(explicit_names, field_name="map_behaviors")
    return ("solid_blocks",) if map_rules else ()


def _derive_character_behavior_names(
    *,
    explicit_names: Optional[Any],
    starting_inventory: Any,
    starting_intrinsics: Any,
    intrinsic_rates: Any,
    intrinsic_thresholds: Any,
) -> Tuple[str, ...]:
    if explicit_names is not None:
        return _normalize_name_list(explicit_names, field_name="character_behaviors")

    names: list[str] = []
    if starting_inventory:
        names.append("starting_inventory")
    if starting_intrinsics:
        names.append("starting_intrinsics")
    if intrinsic_rates or intrinsic_thresholds:
        names.append("intrinsic_dynamics")
    return tuple(names)


def _derive_recovery_behavior_names(
    *,
    explicit_names: Optional[Any],
    recovery_rules: Any,
) -> Tuple[str, ...]:
    if explicit_names is not None:
        return _normalize_name_list(explicit_names, field_name="recovery_behaviors")
    return ("instant_recovery",) if recovery_rules else ()


def _validate_explicit_config_api(
    *,
    preset_name: str,
    generator: Optional[str],
    map_behaviors: Tuple[str, ...],
    character_behaviors: Tuple[str, ...],
    recovery_behaviors: Tuple[str, ...],
    ring_inner_radius: Optional[int],
    ring_outer_radius: Optional[int],
    box_inner_size: Optional[int],
    perimeter_tree_prob: Optional[float],
    starting_inventory: Any,
    starting_intrinsics: Any,
    intrinsic_rates: Any,
    intrinsic_thresholds: Any,
    recovery_rules: Any,
    map_rules: Any,
) -> None:
    if generator is None and any(
        value is not None
        for value in (ring_inner_radius, ring_outer_radius, box_inner_size, perimeter_tree_prob)
    ):
        raise ValueError(
            f"World preset {preset_name}: generator-specific map fields require explicit 'generator'"
        )
    if map_rules and not map_behaviors:
        raise ValueError(
            f"World preset {preset_name}: 'map_rules' requires explicit 'map_behaviors'"
        )
    if starting_inventory and "starting_inventory" not in character_behaviors:
        raise ValueError(
            f"World preset {preset_name}: 'starting_inventory' requires 'starting_inventory' in 'character_behaviors'"
        )
    if starting_intrinsics and "starting_intrinsics" not in character_behaviors:
        raise ValueError(
            f"World preset {preset_name}: 'starting_intrinsics' requires 'starting_intrinsics' in 'character_behaviors'"
        )
    if (intrinsic_rates or intrinsic_thresholds) and "intrinsic_dynamics" not in character_behaviors:
        raise ValueError(
            f"World preset {preset_name}: intrinsic fields require 'intrinsic_dynamics' in 'character_behaviors'"
        )
    if recovery_rules and not recovery_behaviors:
        raise ValueError(
            f"World preset {preset_name}: 'recovery_rules' requires explicit 'recovery_behaviors'"
        )


def build_world_preset_spec(
    *,
    env_name: str,
    preset_name: Optional[str],
    seed: Optional[int],
    env: Optional[dict[str, Any]] = None,
    map: Optional[dict[str, Any]] = None,
    character: Optional[dict[str, Any]] = None,
    recovery: Optional[dict[str, Any]] = None,
    allow_config_lookup: bool = True,
    derive_behavior_defaults: bool = True,
) -> WorldPresetSpec:
    """Build a world preset spec from sectioned configuration inputs.

    Args:
        env_name: Base Craftax environment or alias.
        preset_name: Builtin preset alias or YAML preset name.
        seed: Optional explicit random seed.
        env: Optional environment section override.
        map: Optional map section override.
        character: Optional character section override.
        recovery: Optional recovery section override.
        allow_config_lookup: Whether ``preset_name`` may resolve to YAML.
        derive_behavior_defaults: Whether direct builder calls may derive behavior names.

    Returns:
        A fully normalized ``WorldPresetSpec``.
    """
    env = {} if env is None else dict(env)
    map = {} if map is None else dict(map)
    character = {} if character is None else dict(character)
    recovery = {} if recovery is None else dict(recovery)

    generator = map.get("generator")
    map_behaviors = map.get("behaviors", map.get("map_behaviors"))
    character_behaviors = character.get("behaviors", character.get("character_behaviors"))
    recovery_behaviors = recovery.get("behaviors", recovery.get("recovery_behaviors"))
    map_size = env.get("map_size")
    blocked_block = map.get("blocked_block")
    floor_block = map.get("floor_block")
    perimeter_block = map.get("perimeter_block")
    disable_mob_spawns = env.get("disable_mob_spawns")
    starting_inventory = character.get("starting_inventory")
    starting_intrinsics = character.get("starting_intrinsics")
    intrinsic_rates = character.get("intrinsic_rates")
    intrinsic_thresholds = character.get("intrinsic_thresholds")
    recovery_rules = recovery.get("rules", recovery.get("recovery_rules"))
    map_rules = map.get("rules", map.get("map_rules"))
    ring_inner_radius = map.get("ring_inner_radius")
    ring_outer_radius = map.get("ring_outer_radius")
    box_inner_size = map.get("box_inner_size")
    perimeter_tree_prob = map.get("perimeter_tree_prob")

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
                fallback_generator=generator,
                fallback_map_behaviors=map_behaviors,
                fallback_character_behaviors=character_behaviors,
                fallback_recovery_behaviors=recovery_behaviors,
                fallback_map_size=map_size,
                fallback_ring_inner_radius=ring_inner_radius,
                fallback_ring_outer_radius=ring_outer_radius,
                fallback_box_inner_size=box_inner_size,
                fallback_perimeter_tree_prob=perimeter_tree_prob,
            )

    normalized_env_name = normalize_craftax_env_name(env_name)
    normalized_preset = _normalize_builtin_preset_name(preset_name)
    normalized_generator = (
        _derive_generator_name(
            explicit_generator=generator,
            normalized_preset=normalized_preset,
        )
        if derive_behavior_defaults
        else (None if generator is None else _derive_generator_name(explicit_generator=generator, normalized_preset="default"))
    )
    normalized_map_size = _normalize_map_size(map_size)

    resolved_seed = int(seed) if seed is not None else (
        1 if normalized_preset in {"fixed", "ring_fixed"} else int(np.random.randint(0, 2**31 - 1))
    )

    if normalized_generator == "ring":
        ring_inner_radius = 0 if ring_inner_radius is None else int(ring_inner_radius)
        ring_outer_radius = 12 if ring_outer_radius is None else int(ring_outer_radius)
    else:
        ring_inner_radius = None
        ring_outer_radius = None

    if normalized_generator == "box":
        box_inner_size = 3 if box_inner_size is None else int(box_inner_size)
        perimeter_tree_prob = 0.7 if perimeter_tree_prob is None else float(perimeter_tree_prob)
        if disable_mob_spawns is None:
            disable_mob_spawns = True
    else:
        box_inner_size = None
        perimeter_tree_prob = None

    if derive_behavior_defaults:
        normalized_map_behaviors = _derive_map_behavior_names(
            explicit_names=map_behaviors,
            map_rules=map_rules,
        )
        normalized_character_behaviors = _derive_character_behavior_names(
            explicit_names=character_behaviors,
            starting_inventory=starting_inventory,
            starting_intrinsics=starting_intrinsics,
            intrinsic_rates=intrinsic_rates,
            intrinsic_thresholds=intrinsic_thresholds,
        )
        normalized_recovery_behaviors = _derive_recovery_behavior_names(
            explicit_names=recovery_behaviors,
            recovery_rules=recovery_rules,
        )
    else:
        normalized_map_behaviors = _normalize_name_list(map_behaviors, field_name="map_behaviors")
        normalized_character_behaviors = _normalize_name_list(
            character_behaviors,
            field_name="character_behaviors",
        )
        normalized_recovery_behaviors = _normalize_name_list(
            recovery_behaviors,
            field_name="recovery_behaviors",
        )

    env_spec = EnvPresetSpec(
        env_name=normalized_env_name,
        seed=resolved_seed,
        map_size=normalized_map_size,
        disable_mob_spawns=disable_mob_spawns,
    )
    map_spec = MapPresetSpec(
        generator=normalized_generator,
        behaviors=normalized_map_behaviors,
        blocked_block=None if blocked_block is None else str(blocked_block),
        floor_block=None if floor_block is None else str(floor_block),
        perimeter_block=None if perimeter_block is None else str(perimeter_block),
        rules=_normalize_named_grants(map_rules, "map_rules"),
        ring_inner_radius=ring_inner_radius,
        ring_outer_radius=ring_outer_radius,
        box_inner_size=box_inner_size,
        perimeter_tree_prob=perimeter_tree_prob,
    )
    character_spec = CharacterPresetSpec(
        behaviors=normalized_character_behaviors,
        starting_inventory=_normalize_inventory_grants(starting_inventory),
        starting_intrinsics=_normalize_named_grants(starting_intrinsics, "starting_intrinsics"),
        intrinsic_rates=_normalize_named_grants(intrinsic_rates, "intrinsic_rates"),
        intrinsic_thresholds=_normalize_named_grants(intrinsic_thresholds, "intrinsic_thresholds"),
    )
    recovery_spec = RecoveryPresetSpec(
        behaviors=normalized_recovery_behaviors,
        rules=_normalize_named_grants(recovery_rules, "recovery_rules"),
    )

    return WorldPresetSpec(
        name=normalized_preset,
        env_name=normalized_env_name,
        seed=resolved_seed,
        env=env_spec,
        map=map_spec,
        character=character_spec,
        recovery=recovery_spec,
    )


def build_world_preset_spec_from_config(
    *,
    preset_name: str,
    config_data: dict[str, object],
    fallback_env_name: str,
    fallback_seed: Optional[int],
    fallback_generator: Optional[str],
    fallback_map_behaviors: Any,
    fallback_character_behaviors: Any,
    fallback_recovery_behaviors: Any,
    fallback_map_size: Optional[int | Tuple[int, int]],
    fallback_ring_inner_radius: Optional[int],
    fallback_ring_outer_radius: Optional[int],
    fallback_box_inner_size: Optional[int],
    fallback_perimeter_tree_prob: Optional[float],
) -> WorldPresetSpec:
    """Build a preset spec from YAML configuration with inheritance support.

    Args:
        preset_name: Logical preset name being resolved.
        config_data: Parsed YAML mapping for the preset.
        fallback_env_name: Active environment name from caller context.
        fallback_seed: Seed fallback from caller context.
        fallback_generator: Generator fallback for non-YAML callers.
        fallback_map_behaviors: Map behavior fallback for non-YAML callers.
        fallback_character_behaviors: Character behavior fallback for non-YAML callers.
        fallback_recovery_behaviors: Recovery behavior fallback for non-YAML callers.
        fallback_map_size: Map size fallback from caller context.
        fallback_ring_inner_radius: Ring inner radius fallback from caller context.
        fallback_ring_outer_radius: Ring outer radius fallback from caller context.
        fallback_box_inner_size: Box size fallback from caller context.
        fallback_perimeter_tree_prob: Perimeter probability fallback from caller context.

    Returns:
        A validated and normalized ``WorldPresetSpec``.
    """
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
            env={
                "map_size": fallback_map_size,
            },
            map={
                "generator": fallback_generator,
                "behaviors": fallback_map_behaviors,
                "ring_inner_radius": fallback_ring_inner_radius,
                "ring_outer_radius": fallback_ring_outer_radius,
                "box_inner_size": fallback_box_inner_size,
                "perimeter_tree_prob": fallback_perimeter_tree_prob,
            },
            character={
                "behaviors": fallback_character_behaviors,
            },
            recovery={
                "behaviors": fallback_recovery_behaviors,
            },
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

    generator_value = config_data.get(
        "generator",
        inherited.generator if inherited is not None else None,
    )
    map_behaviors_value = config_data.get(
        "map_behaviors",
        list(inherited.map_behaviors) if inherited is not None and inherited.map_behaviors else None,
    )
    character_behaviors_value = config_data.get(
        "character_behaviors",
        list(inherited.character_behaviors)
        if inherited is not None and inherited.character_behaviors
        else None,
    )
    recovery_behaviors_value = config_data.get(
        "recovery_behaviors",
        list(inherited.recovery_behaviors)
        if inherited is not None and inherited.recovery_behaviors
        else None,
    )
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

    normalized_map_behaviors = _normalize_name_list(map_behaviors_value, field_name="map_behaviors")
    normalized_character_behaviors = _normalize_name_list(
        character_behaviors_value,
        field_name="character_behaviors",
    )
    normalized_recovery_behaviors = _normalize_name_list(
        recovery_behaviors_value,
        field_name="recovery_behaviors",
    )
    normalized_generator = None if generator_value is None else _derive_generator_name(
        explicit_generator=str(generator_value),
        normalized_preset="default",
    )

    _validate_explicit_config_api(
        preset_name=preset_name,
        generator=normalized_generator,
        map_behaviors=normalized_map_behaviors,
        character_behaviors=normalized_character_behaviors,
        recovery_behaviors=normalized_recovery_behaviors,
        ring_inner_radius=ring_inner_value,
        ring_outer_radius=ring_outer_value,
        box_inner_size=box_inner_value,
        perimeter_tree_prob=perimeter_tree_prob_value,
        starting_inventory=starting_inventory_value,
        starting_intrinsics=starting_intrinsics_value,
        intrinsic_rates=intrinsic_rates_value,
        intrinsic_thresholds=intrinsic_thresholds_value,
        recovery_rules=recovery_rules_value,
        map_rules=map_rules_value,
    )

    return build_world_preset_spec(
        env_name=env_value,
        preset_name=str(preset_kind),
        seed=seed_value,
        env={
            "map_size": map_size_value,
            "disable_mob_spawns": disable_mob_spawns_value,
        },
        map={
            "generator": normalized_generator,
            "behaviors": normalized_map_behaviors,
            "blocked_block": blocked_block_value,
            "floor_block": floor_block_value,
            "perimeter_block": perimeter_block_value,
            "rules": map_rules_value,
            "ring_inner_radius": ring_inner_value,
            "ring_outer_radius": ring_outer_value,
            "box_inner_size": box_inner_value,
            "perimeter_tree_prob": perimeter_tree_prob_value,
        },
        character={
            "behaviors": normalized_character_behaviors,
            "starting_inventory": starting_inventory_value,
            "starting_intrinsics": starting_intrinsics_value,
            "intrinsic_rates": intrinsic_rates_value,
            "intrinsic_thresholds": intrinsic_thresholds_value,
        },
        recovery={
            "behaviors": normalized_recovery_behaviors,
            "rules": recovery_rules_value,
        },
        allow_config_lookup=False,
        derive_behavior_defaults=False,
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


def build_env_and_params(
    spec: WorldPresetSpec,
    *,
    auto_reset: bool = False,
    adapter_cls: Optional[type[Any]] = None,
) -> Tuple[Any, Any]:
    """Construct the runtime environment and env params for a preset.

    Args:
        spec: Resolved world preset specification.
        auto_reset: Whether to build an auto-resetting Craftax env variant.
        adapter_cls: Optional runtime adapter override. Defaults to ``CompositePresetAdapter``.

    Returns:
        A tuple of ``(env, env_params)`` ready for reset/step calls.
    """
    static_env_params = _build_static_env_params(spec.env.env_name, spec.env.map_size)

    if static_env_params is None:
        env = make_craftax_env_from_name(spec.env.env_name, auto_reset=auto_reset)
    elif spec.env.env_name == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
            CraftaxClassicPixelsEnvNoAutoReset,
        )
        env = CraftaxClassicPixelsEnv(static_env_params) if auto_reset else CraftaxClassicPixelsEnvNoAutoReset(static_env_params)
    elif spec.env.env_name == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
            CraftaxClassicSymbolicEnvNoAutoReset,
        )
        env = (
            CraftaxClassicSymbolicEnv(static_env_params)
            if auto_reset
            else CraftaxClassicSymbolicEnvNoAutoReset(static_env_params)
        )
    elif spec.env.env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv, CraftaxPixelsEnvNoAutoReset
        env = CraftaxPixelsEnv(static_env_params) if auto_reset else CraftaxPixelsEnvNoAutoReset(static_env_params)
    elif spec.env.env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv, CraftaxSymbolicEnvNoAutoReset
        env = CraftaxSymbolicEnv(static_env_params) if auto_reset else CraftaxSymbolicEnvNoAutoReset(static_env_params)
    else:
        raise ValueError(f"Unsupported Craftax environment for world preset: {spec.env.env_name}")

    env_params = env.default_params
    if spec.env.disable_mob_spawns:
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

    if spec.uses_map_adapter or spec.uses_character_adapter or spec.uses_recovery_adapter:
        runtime_adapter_cls = CompositePresetAdapter if adapter_cls is None else adapter_cls
        env = runtime_adapter_cls(env, spec)

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


def _apply_state_grants_to_state(
    *,
    state: Any,
    grants: Tuple[InventoryGrantSpec, ...],
    key: Any,
    label: str,
    env_name: str,
):
    rng = key
    updates: dict[str, Any] = {}
    for grant in grants:
        if not hasattr(state, grant.item):
            raise ValueError(
                f"World preset {label} item {grant.item!r} does not exist for env {env_name}"
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


class BaseWorldGenerator:
    """Extension point for custom world generation overlays.

    Subclasses translate ``MapPresetSpec`` into a generated world-state overlay
    that is applied during ``reset`` before gameplay begins.
    """

    generator_name = "base"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        self.spec = spec
        self.resolved_family = resolved_family

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return False

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        raise NotImplementedError


class BoxWorldGenerator(BaseWorldGenerator):
    generator_name = "box"

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return spec.has_box

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        if self.resolved_family == "classic":
            from craftax.craftax_classic.constants import BlockType

            blocked_value = _resolve_classic_block_value(self.spec.map.blocked_block, BlockType.OUT_OF_BOUNDS.value)
            floor_value = _resolve_classic_block_value(self.spec.map.floor_block, BlockType.GRASS.value)
            perimeter_value = _resolve_classic_block_value(self.spec.map.perimeter_block, BlockType.TREE.value)
            updated_map, spawn_position = _apply_box_to_level(
                state.map,
                key=key,
                blocked_value=blocked_value,
                floor_value=floor_value,
                tree_value=perimeter_value,
                inner_size=int(self.spec.map.box_inner_size),
                perimeter_tree_prob=float(self.spec.map.perimeter_tree_prob or 0.7),
            )
            return GeneratedWorldState(map=updated_map, player_position=spawn_position)

        from craftax.craftax.constants import BlockType

        current_level = int(state.player_level)
        current_map = state.map[current_level]
        default_floor_value = BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value
        default_perimeter_value = BlockType.TREE.value if current_level == 0 else BlockType.WALL.value
        blocked_value = _resolve_full_block_value(self.spec.map.blocked_block, BlockType.OUT_OF_BOUNDS.value)
        floor_value = _resolve_full_block_value(self.spec.map.floor_block, default_floor_value)
        perimeter_value = _resolve_full_block_value(self.spec.map.perimeter_block, default_perimeter_value)
        updated_level_map, spawn_position = _apply_box_to_level(
            current_map,
            key=key,
            blocked_value=blocked_value,
            floor_value=floor_value,
            tree_value=perimeter_value,
            inner_size=int(self.spec.map.box_inner_size),
            perimeter_tree_prob=float(self.spec.map.perimeter_tree_prob or 0.7),
        )
        updated_light_level = jnp.where(
            updated_level_map == BlockType.OUT_OF_BOUNDS.value,
            0.0,
            state.light_map[current_level],
        )
        return GeneratedWorldState(
            map=state.map.at[current_level].set(updated_level_map),
            player_position=spawn_position,
            item_map=state.item_map.at[current_level].set(jnp.zeros_like(state.item_map[current_level])),
            mob_map=state.mob_map.at[current_level].set(jnp.zeros_like(state.mob_map[current_level])),
            light_map=state.light_map.at[current_level].set(updated_light_level),
        )


class RingWorldGenerator(BaseWorldGenerator):
    generator_name = "ring"

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return spec.has_ring

    def apply(self, state: Any, key: Any) -> GeneratedWorldState:
        if self.resolved_family == "classic":
            from craftax.craftax_classic.constants import BlockType

            blocked_value = _resolve_classic_block_value(self.spec.map.blocked_block, BlockType.WATER.value)
            floor_value = _resolve_classic_block_value(self.spec.map.floor_block, BlockType.GRASS.value)
            updated_map, spawn_position = _apply_ring_to_level(
                state.map,
                player_position=state.player_position,
                blocked_value=blocked_value,
                spawn_value=floor_value,
                inner_radius=int(self.spec.map.ring_inner_radius or 0),
                outer_radius=int(self.spec.map.ring_outer_radius),
            )
            return GeneratedWorldState(map=updated_map, player_position=spawn_position)

        from craftax.craftax.constants import BlockType, ItemType

        current_level = int(state.player_level)
        current_map = state.map[current_level]
        blocked_value = _resolve_full_block_value(
            self.spec.map.blocked_block,
            BlockType.WATER.value if current_level == 0 else BlockType.WALL.value,
        )
        floor_value = _resolve_full_block_value(
            self.spec.map.floor_block,
            BlockType.GRASS.value if current_level == 0 else BlockType.PATH.value,
        )
        updated_level_map, spawn_position = _apply_ring_to_level(
            current_map,
            player_position=state.player_position,
            blocked_value=blocked_value,
            spawn_value=floor_value,
            inner_radius=int(self.spec.map.ring_inner_radius or 0),
            outer_radius=int(self.spec.map.ring_outer_radius),
        )
        ring_mask = _distance_mask(
            current_map.shape[0],
            current_map.shape[1],
            current_map.shape[0] // 2,
            current_map.shape[1] // 2,
            int(self.spec.map.ring_inner_radius or 0),
            int(self.spec.map.ring_outer_radius),
        )
        return GeneratedWorldState(
            map=state.map.at[current_level].set(updated_level_map),
            player_position=spawn_position,
            item_map=state.item_map.at[current_level].set(
                jnp.where(ring_mask, state.item_map[current_level], ItemType.NONE.value)
            ),
            mob_map=state.mob_map.at[current_level].set(jnp.where(ring_mask, state.mob_map[current_level], 0)),
            light_map=state.light_map.at[current_level].set(
                jnp.where(ring_mask, state.light_map[current_level], 0.0)
            ),
        )


WORLD_GENERATOR_REGISTRY: dict[str, type[BaseWorldGenerator]] = {
    BoxWorldGenerator.generator_name: BoxWorldGenerator,
    RingWorldGenerator.generator_name: RingWorldGenerator,
}


class BaseMapBehavior:
    """Extension point for map-related policies such as collision rules.

    Map behaviors run in the preset pipeline after generation and may adjust
    reset state or post-step state while remaining independent from generation.
    """

    behavior_name = "base_map"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        self.spec = spec
        self.resolved_family = resolved_family
        self.rules = _grants_to_mapping(spec.map.rules)

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return False

    def apply_reset(self, state: Any, key: Any) -> Any:
        return state

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        return new_state


class SolidBlocksBehavior(BaseMapBehavior):
    behavior_name = "solid_blocks"

    def __init__(self, spec: WorldPresetSpec, resolved_family: str) -> None:
        super().__init__(spec, resolved_family)
        self.solid_block_values = self._resolve_solid_block_values()

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return bool(spec.map.rules)

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        if not self.solid_block_values:
            return new_state

        if self.resolved_family == "classic":
            pos = new_state.player_position
            current_block = int(new_state.map[pos[0], pos[1]])
        else:
            pos = new_state.player_position
            level = int(new_state.player_level)
            current_block = int(new_state.map[level, pos[0], pos[1]])

        if current_block not in self.solid_block_values:
            return new_state
        return new_state.replace(player_position=previous_state.player_position)

    def _resolve_solid_block_values(self) -> set[int]:
        solid_values: set[int] = set()
        if bool(self.rules.get("solid_out_of_bounds", False)):
            if self.resolved_family == "classic":
                from craftax.craftax_classic.constants import BlockType
            else:
                from craftax.craftax.constants import BlockType
            solid_values.add(int(BlockType.OUT_OF_BOUNDS.value))

        extra_blocks = self.rules.get("solid_blocks", [])
        if isinstance(extra_blocks, str):
            extra_blocks = [extra_blocks]
        if extra_blocks:
            resolver = _resolve_classic_block_value if self.resolved_family == "classic" else _resolve_full_block_value
            for block_name in extra_blocks:
                solid_values.add(int(resolver(str(block_name), 0)))
        return solid_values


MAP_BEHAVIOR_REGISTRY: dict[str, type[BaseMapBehavior]] = {
    SolidBlocksBehavior.behavior_name: SolidBlocksBehavior,
}


class BaseCharacterBehavior:
    """Extension point for character state overrides and dynamics.

    Character behaviors may patch reset state, step-time dynamics, or both.
    """

    behavior_name = "base_character"

    def __init__(self, spec: WorldPresetSpec) -> None:
        self.spec = spec

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return False

    def apply_reset(self, state: Any, key: Any) -> Any:
        return state

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        return new_state


class StartingInventoryBehavior(BaseCharacterBehavior):
    behavior_name = "starting_inventory"

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return bool(spec.character.starting_inventory)

    def apply_reset(self, state: Any, key: Any) -> Any:
        inventory = _apply_grants_to_record(
            record=state.inventory,
            grants=self.spec.character.starting_inventory,
            key=key,
            context_name=f"inventory for env {self.spec.env.env_name}",
        )
        return state.replace(inventory=inventory)


class StartingIntrinsicsBehavior(BaseCharacterBehavior):
    behavior_name = "starting_intrinsics"

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return bool(spec.character.starting_intrinsics)

    def apply_reset(self, state: Any, key: Any) -> Any:
        return _apply_state_grants_to_state(
            state=state,
            grants=self.spec.character.starting_intrinsics,
            key=key,
            label="starting_intrinsics",
            env_name=self.spec.env.env_name,
        )


class IntrinsicDynamicsBehavior(BaseCharacterBehavior):
    behavior_name = "intrinsic_dynamics"

    def __init__(self, spec: WorldPresetSpec) -> None:
        super().__init__(spec)
        self.rate_values = _grants_to_mapping(spec.character.intrinsic_rates)
        self.threshold_values = _grants_to_mapping(spec.character.intrinsic_thresholds)

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return bool(spec.character.intrinsic_rates or spec.character.intrinsic_thresholds)

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        state = new_state
        updates: dict[str, Any] = {}

        def _value(name: str, default: float) -> float:
            return float(self.rate_values.get(name, default))

        def _threshold(name: str, default: float) -> float:
            return float(self.threshold_values.get(name, default))

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


CHARACTER_BEHAVIOR_REGISTRY: dict[str, type[BaseCharacterBehavior]] = {
    StartingInventoryBehavior.behavior_name: StartingInventoryBehavior,
    StartingIntrinsicsBehavior.behavior_name: StartingIntrinsicsBehavior,
    IntrinsicDynamicsBehavior.behavior_name: IntrinsicDynamicsBehavior,
}


class BaseRecoveryBehavior:
    """Extension point for recovery and sleep/rest overrides."""

    behavior_name = "base_recovery"

    def __init__(self, spec: WorldPresetSpec) -> None:
        self.spec = spec
        self.rules = _grants_to_mapping(spec.recovery.rules)

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return False

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        return new_state


class InstantRecoveryBehavior(BaseRecoveryBehavior):
    behavior_name = "instant_recovery"

    @classmethod
    def matches(cls, spec: WorldPresetSpec) -> bool:
        return bool(spec.recovery.rules)

    def apply_step(self, previous_state: Any, new_state: Any) -> Any:
        state = new_state
        updates: dict[str, Any] = {}
        instant_sleep_enabled = bool(self.rules.get("instant_sleep_recovery", False))
        instant_rest_enabled = bool(self.rules.get("instant_rest_recovery", False))

        if instant_sleep_enabled and getattr(state, "is_sleeping", False):
            self._apply_recovery_mode_updates(state=state, updates=updates, prefix="sleep")
            if bool(self.rules.get("wake_after_sleep_recovery", True)) and hasattr(state, "is_sleeping"):
                updates["is_sleeping"] = False

        if instant_rest_enabled and getattr(state, "is_resting", False):
            self._apply_recovery_mode_updates(state=state, updates=updates, prefix="rest")
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


RECOVERY_BEHAVIOR_REGISTRY: dict[str, type[BaseRecoveryBehavior]] = {
    InstantRecoveryBehavior.behavior_name: InstantRecoveryBehavior,
}


class CompositePresetAdapter:
    """Single runtime wrapper that executes all preset pipelines.

    The adapter applies reset-time and step-time transforms in a fixed order and
    recomputes observations once after all transforms complete.

    Args:
        env: Wrapped Craftax environment.
        spec: Resolved world preset specification.
    """

    def __init__(self, env: Any, spec: WorldPresetSpec) -> None:
        self.env = env
        self.spec = spec
        self.resolved_family = resolve_base_environment(spec.env.env_name).family
        self.world_generator = self._build_world_generator()
        self.map_behaviors = self._build_map_behaviors()
        self.character_behaviors = self._build_character_behaviors()
        self.recovery_behaviors = self._build_recovery_behaviors()
        self.reset_transforms = self._build_reset_pipeline()
        self.step_transforms = self._build_step_pipeline()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, key: Any, params: Any):
        obs, state = self.env.reset(key, params)
        for transform in self.reset_transforms:
            state = transform(state, key)
        return self._finalize(obs=obs, state=state)

    def step(self, key: Any, state: Any, action: int, params: Any):
        obs, new_state, reward, done, info = self.env.step(key, state, action, params)
        for transform in self.step_transforms:
            new_state = transform(state, new_state)
        obs, new_state = self._finalize(obs=obs, state=new_state)
        return obs, new_state, reward, done, info

    def _finalize(self, *, obs: Any, state: Any):
        if hasattr(self.env, "get_obs"):
            obs = self.env.get_obs(state)
        return obs, state

    def _build_reset_pipeline(self) -> tuple[Callable[[Any, Any], Any], ...]:
        transforms: list[Callable[[Any, Any], Any]] = []
        if self.spec.uses_map_adapter:
            transforms.append(self._apply_map_overlay)
        for behavior in self.map_behaviors:
            if behavior.apply_reset.__func__ is not BaseMapBehavior.apply_reset:
                transforms.append(behavior.apply_reset)
        for behavior in self.character_behaviors:
            if behavior.apply_reset.__func__ is not BaseCharacterBehavior.apply_reset:
                transforms.append(behavior.apply_reset)
        return tuple(transforms)

    def _build_step_pipeline(self) -> tuple[Callable[[Any, Any], Any], ...]:
        transforms: list[Callable[[Any, Any], Any]] = []
        for behavior in self.map_behaviors:
            if behavior.apply_step.__func__ is not BaseMapBehavior.apply_step:
                transforms.append(behavior.apply_step)
        for behavior in self.character_behaviors:
            if behavior.apply_step.__func__ is not BaseCharacterBehavior.apply_step:
                transforms.append(behavior.apply_step)
        for behavior in self.recovery_behaviors:
            if behavior.apply_step.__func__ is not BaseRecoveryBehavior.apply_step:
                transforms.append(behavior.apply_step)
        return tuple(transforms)

    def _build_map_behaviors(self) -> tuple[BaseMapBehavior, ...]:
        behaviors: list[BaseMapBehavior] = []
        for behavior_name in self.spec.map.behaviors:
            behavior_cls = MAP_BEHAVIOR_REGISTRY.get(behavior_name)
            if behavior_cls is None:
                raise ValueError(f"Unknown map behavior for world preset: {behavior_name}")
            behaviors.append(behavior_cls(self.spec, self.resolved_family))
        return tuple(behaviors)

    def _build_character_behaviors(self) -> tuple[BaseCharacterBehavior, ...]:
        behaviors: list[BaseCharacterBehavior] = []
        for behavior_name in self.spec.character.behaviors:
            behavior_cls = CHARACTER_BEHAVIOR_REGISTRY.get(behavior_name)
            if behavior_cls is None:
                raise ValueError(f"Unknown character behavior for world preset: {behavior_name}")
            behaviors.append(behavior_cls(self.spec))
        return tuple(behaviors)

    def _build_recovery_behaviors(self) -> tuple[BaseRecoveryBehavior, ...]:
        behaviors: list[BaseRecoveryBehavior] = []
        for behavior_name in self.spec.recovery.behaviors:
            behavior_cls = RECOVERY_BEHAVIOR_REGISTRY.get(behavior_name)
            if behavior_cls is None:
                raise ValueError(f"Unknown recovery behavior for world preset: {behavior_name}")
            behaviors.append(behavior_cls(self.spec))
        return tuple(behaviors)

    def _apply_map_overlay(self, state: Any, key: Any):
        if self.world_generator is None:
            return state
        generated = self.world_generator.apply(state, key)
        updates = {
            "map": generated.map,
            "player_position": generated.player_position,
        }
        if generated.item_map is not None:
            updates["item_map"] = generated.item_map
        if generated.mob_map is not None:
            updates["mob_map"] = generated.mob_map
        if generated.light_map is not None:
            updates["light_map"] = generated.light_map
        return state.replace(**updates)

    def _build_world_generator(self) -> Optional[BaseWorldGenerator]:
        if self.spec.map.generator is None:
            return None
        generator_cls = WORLD_GENERATOR_REGISTRY.get(self.spec.map.generator)
        if generator_cls is None:
            raise ValueError(f"Unknown world generator for preset: {self.spec.map.generator}")
        return generator_cls(self.spec, self.resolved_family)
