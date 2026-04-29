"""Interactive local renderer and keyboard control loop for manual CrafText play."""

import argparse
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from pygame.colordict import THECOLORS as colors

from craftext.environment.craftext_wrapper import RawInstructionWrapper, TextEnvState
from craftext.environment.scenarious.instruction_transformers import DefaultInstructionTransformer
from craftext.environment.scenarious.loader import (
    CraftextScenariosConfigLoader,
    ScenariosConfig,
    resolve_base_environment,
)
from craftext.environment.scenarious.manager import DefaultJAXRepresentation, JaxScenarioDataHandler
from craftext.environment.scenarious.processors import RawProcessor
from craftext.environment.world_presets import (
    build_env_and_params,
    build_world_preset_spec,
    looks_like_env_name,
    normalize_craftax_env_name,
)

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ACTION_KEY_SPECS = (
    (pygame.K_q, "NOOP"),
    (pygame.K_w, "UP"),
    (pygame.K_d, "RIGHT"),
    (pygame.K_s, "DOWN"),
    (pygame.K_a, "LEFT"),
    (pygame.K_SPACE, "DO"),
    (pygame.K_t, "PLACE_TABLE"),
    (pygame.K_TAB, "SLEEP"),
    (pygame.K_r, "PLACE_STONE"),
    (pygame.K_f, "PLACE_FURNACE"),
    (pygame.K_p, "PLACE_PLANT"),
    (pygame.K_1, "MAKE_WOOD_PICKAXE"),
    (pygame.K_2, "MAKE_STONE_PICKAXE"),
    (pygame.K_3, "MAKE_IRON_PICKAXE"),
    (pygame.K_4, "MAKE_WOOD_SWORD"),
    (pygame.K_5, "MAKE_STONE_SWORD"),
    (pygame.K_6, "MAKE_IRON_SWORD"),
    (pygame.K_7, "MAKE_DIAMOND_PICKAXE"),
    (pygame.K_8, "MAKE_DIAMOND_SWORD"),
    (pygame.K_e, "REST"),
    (pygame.K_COMMA, "ASCEND"),
    (pygame.K_PERIOD, "DESCEND"),
    (pygame.K_y, "MAKE_IRON_ARMOUR"),
    (pygame.K_u, "MAKE_DIAMOND_ARMOUR"),
    (pygame.K_i, "SHOOT_ARROW"),
    (pygame.K_o, "MAKE_ARROW"),
    (pygame.K_g, "CAST_FIREBALL"),
    (pygame.K_h, "CAST_ICEBALL"),
    (pygame.K_j, "PLACE_TORCH"),
    (pygame.K_z, "DRINK_POTION_RED"),
    (pygame.K_x, "DRINK_POTION_GREEN"),
    (pygame.K_c, "DRINK_POTION_BLUE"),
    (pygame.K_v, "DRINK_POTION_PINK"),
    (pygame.K_b, "DRINK_POTION_CYAN"),
    (pygame.K_n, "DRINK_POTION_YELLOW"),
    (pygame.K_m, "READ_BOOK"),
    (pygame.K_k, "ENCHANT_SWORD"),
    (pygame.K_l, "ENCHANT_ARMOUR"),
    (pygame.K_LEFTBRACKET, "MAKE_TORCH"),
    (pygame.K_RIGHTBRACKET, "LEVEL_UP_DEXTERITY"),
    (pygame.K_MINUS, "LEVEL_UP_STRENGTH"),
    (pygame.K_EQUALS, "LEVEL_UP_INTELLIGENCE"),
    (pygame.K_SEMICOLON, "ENCHANT_BOW"),
)

def resolve_play_env_name(
    config: ScenariosConfig,
    env_override: Optional[str],
    world_preset_override: Optional[str],
) -> str:
    """Resolve the environment name used for manual play."""
    if env_override:
        return normalize_craftax_env_name(env_override)

    preset_value = world_preset_override or config.world_preset
    if preset_value and looks_like_env_name(preset_value):
        return normalize_craftax_env_name(preset_value)

    return normalize_craftax_env_name(config.base_environment)


def load_rendering_runtime(env_name: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Load renderer and enums matching the selected Craftax family."""
    resolved_base_env = resolve_base_environment(env_name)
    if resolved_base_env.family == "classic":
        from craftax.craftax_classic.constants import (
            Action,
            Achievement,
            BLOCK_PIXEL_SIZE_HUMAN,
            INVENTORY_OBS_HEIGHT,
            OBS_DIM,
        )
        from craftax.craftax_classic.renderer import render_craftax_pixels
    else:
        from craftax.craftax.constants import (
            Action,
            Achievement,
            BLOCK_PIXEL_SIZE_HUMAN,
            INVENTORY_OBS_HEIGHT,
            OBS_DIM,
        )
        from craftax.craftax.renderer import render_craftax_pixels

    return OBS_DIM, BLOCK_PIXEL_SIZE_HUMAN, INVENTORY_OBS_HEIGHT, Action, Achievement, render_craftax_pixels


def build_key_mapping(action_enum: Any) -> Dict[int, Any]:
    """Build only the controls supported by the selected Action enum."""
    mapping: Dict[int, Any] = {}
    for key_code, action_name in ACTION_KEY_SPECS:
        if hasattr(action_enum, action_name):
            mapping[key_code] = getattr(action_enum, action_name)
    return mapping


class CrafTextRenderer:
    def __init__(
        self,
        *,
        render_fn: Any,
        obs_dim: Tuple[int, int],
        block_pixel_size_human: int,
        inventory_obs_height: int,
        scenario_handler: JaxScenarioDataHandler,
        pixel_render_size: int = 4,
    ) -> None:
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []
        self.block_pixel_size_human = block_pixel_size_human
        self.scenario_handler = scenario_handler

        self.screen_size = (
            obs_dim[1] * block_pixel_size_human * pixel_render_size,
            (obs_dim[0] + inventory_obs_height) * block_pixel_size_human * pixel_render_size + 150,
        )

        pygame.init()
        pygame.key.set_repeat(250, 75)
        self.font_t = pygame.font.Font(pygame.font.get_default_font(), 11)
        self.screen_surface = pygame.display.set_mode(self.screen_size)
        self._render = jax.jit(render_fn, static_argnums=(1,))

    def update(self) -> None:
        self.pygame_events = list(pygame.event.get())
        pygame.display.flip()

    def render_field(self, env_state: TextEnvState[Any]) -> jax.Array:
        pixels = self._render(env_state.env_state, block_pixel_size=self.block_pixel_size_human)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        return pixels

    def draw_header(self, text: str, subtitle: str) -> None:
        pygame.draw.rect(self.screen_surface, colors["white"], (0, 0, self.screen_size[0], 150))

        text_surface = self.font_t.render(text, True, colors["black"])
        subtitle_surface = self.font_t.render(subtitle, True, colors["black"])

        text_rect = text_surface.get_rect(center=(self.screen_size[0] // 2, 75 // 2))
        self.screen_surface.blit(text_surface, text_rect)

        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_size[0] // 2, (75 // 2) + 75))
        self.screen_surface.blit(subtitle_surface, subtitle_rect)

    def render(self, env_state: TextEnvState[Any], env_name: str) -> None:
        self.screen_surface.fill((0, 0, 0))

        instruction = self.scenario_handler.scenario_data.instructions_list[int(env_state.idx)]
        image = np.array(self.render_field(env_state))
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 75))
        self.draw_header(instruction, env_name)

    def is_quit_requested(self) -> bool:
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state: Any, key_mapping: Dict[int, Any]) -> Optional[int]:
        if getattr(state, "is_sleeping", False) or getattr(state, "is_resting", False):
            noop = key_mapping.get(pygame.K_q)
            return None if noop is None else noop.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN and event.key in key_mapping:
                return key_mapping[event.key].value

        return None


def print_new_achievements(old_achievements: jax.Array, new_achievements: jax.Array, achievement_enum: Any) -> None:
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievement_enum(i).name} ({new_achievements.sum()}/{len(achievement_enum)})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive CrafText manual play.")
    parser.add_argument(
        "--config",
        default="building/easy/line",
        help="Scenario config name under craftext/dataset/configs.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Explicit Craftax env id or alias. Example: Craftax-Pixels-v1, classic, full.",
    )
    parser.add_argument(
        "--world-preset",
        default=None,
        help="World preset name or env id. Supported builtins: default, random, fixed, ring_random, ring_fixed, ring, box, box3_random_trees.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for world generation. Fixed presets derive stable noise angles from it.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=None,
        help="Override square world size for presets that rebuild the underlying env.",
    )
    parser.add_argument(
        "--ring-inner-radius",
        type=int,
        default=None,
        help="Inner radius for ring presets.",
    )
    parser.add_argument(
        "--ring-outer-radius",
        type=int,
        default=None,
        help="Outer radius for ring presets.",
    )
    parser.add_argument(
        "--box-inner-size",
        type=int,
        default=None,
        help="Playable square size for box presets.",
    )
    parser.add_argument(
        "--perimeter-tree-prob",
        type=float,
        default=None,
        help="Tree probability on the one-tile perimeter for box presets.",
    )
    parser.add_argument(
        "--instruction-idx",
        type=int,
        default=-1,
        help="Fixed instruction index. Use -1 to sample randomly.",
    )
    parser.add_argument(
        "--pixel-render-size",
        type=int,
        default=1,
        help="Integer upscaling factor for the rendered game view.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CraftextScenariosConfigLoader.load_config(args.config)
    env_name = resolve_play_env_name(
        config=config,
        env_override=args.env,
        world_preset_override=args.world_preset,
    )
    preset_name = None
    if args.world_preset and not looks_like_env_name(args.world_preset):
        preset_name = args.world_preset

    world_preset_spec = build_world_preset_spec(
        env_name=env_name,
        preset_name=preset_name,
        seed=args.seed,
        env={
            "map_size": args.map_size,
        },
        map={
            "ring_inner_radius": args.ring_inner_radius,
            "ring_outer_radius": args.ring_outer_radius,
            "box_inner_size": args.box_inner_size,
            "perimeter_tree_prob": args.perimeter_tree_prob,
        },
    )

    obs_dim, block_pixel_size_human, inventory_obs_height, action_enum, achievement_enum, render_fn = (
        load_rendering_runtime(env_name)
    )
    key_mapping = build_key_mapping(action_enum)

    env, env_params = build_env_and_params(world_preset_spec, auto_reset=False)
    scenario_handler = JaxScenarioDataHandler(
        scenario_processor=RawProcessor,
        instruction_transformer=DefaultInstructionTransformer,
        config_name=args.config,
        jax_representation_class=DefaultJAXRepresentation,
    )
    wrapper = RawInstructionWrapper(env, scenario_handler=scenario_handler)

    logger.info("Play config: %s", args.config)
    logger.info("Config base_environment: %s", config.base_environment)
    logger.info("Resolved env: %s", env_name)
    logger.info("World preset: %s", world_preset_spec.name)
    logger.info("World seed: %s", world_preset_spec.seed)
    if world_preset_spec.map_size is not None:
        logger.info("World map_size: %s", world_preset_spec.map_size)
    if world_preset_spec.has_ring:
        logger.info(
            "World ring: inner=%s outer=%s",
            world_preset_spec.ring_inner_radius,
            world_preset_spec.ring_outer_radius,
        )
    if world_preset_spec.has_box:
        logger.info(
            "World box: inner_size=%s perimeter_tree_prob=%s",
            world_preset_spec.box_inner_size,
            world_preset_spec.perimeter_tree_prob,
        )

    base_rng = jax.random.PRNGKey(world_preset_spec.seed)
    rng = base_rng

    _, env_state = wrapper.reset(base_rng, env_params, instruction_idx=args.instruction_idx)
    renderer = CrafTextRenderer(
        render_fn=render_fn,
        obs_dim=obs_dim,
        block_pixel_size_human=block_pixel_size_human,
        inventory_obs_height=inventory_obs_height,
        scenario_handler=scenario_handler,
        pixel_render_size=args.pixel_render_size,
    )
    renderer.render(env_state=env_state, env_name=env_name)

    print("Controls")
    for key_code, action in key_mapping.items():
        print(f"{pygame.key.name(key_code)}: {action.name.lower()}")

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state.env_state, key_mapping)
        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.env_state.achievements
            _, env_state, reward, done, _ = wrapper.step(rng, env_state, action, env_params)
            new_achievements = env_state.env_state.achievements
            print_new_achievements(old_achievements, new_achievements, achievement_enum)

            if reward > 0.01 or reward < -0.01:
                print(f"Reward: {reward}\n")

            renderer.render(env_state=env_state, env_name=env_name)

            if done:
                reset_rng = base_rng if world_preset_spec.name in {"fixed", "ring_fixed"} else _rng
                _, env_state = wrapper.reset(reset_rng, env_params, instruction_idx=args.instruction_idx)
                renderer.render(env_state=env_state, env_name=env_name)

        renderer.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
