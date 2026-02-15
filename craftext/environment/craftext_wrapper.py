"""Instruction-aware wrappers and typed state containers for CrafText env steps."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Generic, MutableMapping, Optional, Protocol, Tuple, TypeVar, Union
import jax
import jax.numpy as jnp
from jax import Array
from jax import lax

from flax import struct


from craftext.environment.scenarious.manager import  JaxScenarioDataHandler

from craftext.environment.states.state import GameData
from craftext.environment.states.state_classic import GameDataClassic
from craftext.environment.craftext_constants import Scenarios 
from craftext.environment.scenarious.checkers.target_state  import TargetState
from craftext.environment.scenarious.checkers.registry import CHECKER_FUNCTIONS

logger = logging.getLogger(__name__)

ObsT = TypeVar("ObsT")
EnvStateT = TypeVar("EnvStateT")
InfoT = TypeVar("InfoT", bound=MutableMapping[str, Any])


class JaxEnvProtocol(Protocol[ObsT, EnvStateT, InfoT]):
    def reset(self, key: Array, params: Any) -> Tuple[ObsT, EnvStateT]:
        ...

    def step(
        self,
        key: Array,
        state: EnvStateT,
        action: int,
        params: Any,
    ) -> Tuple[ObsT, EnvStateT, Array, Array, InfoT]:
        ...

@struct.dataclass
class TextEnvState(Generic[EnvStateT]):
    env_state: EnvStateT
    timestep: int
    idx: int
    success_rate: float
    total_success_rate: float
    rng: Array
    instruction_done: bool
    checker_id: int
    target_state: TargetState   


def generic_check(
    game_data: Union[GameData, GameDataClassic],
    target_state: TargetState,
    idx: int,
    fns: Any,
) -> jnp.ndarray:
    return lax.switch(idx, fns, game_data, target_state)


class BaseInstructionWrapper(Generic[ObsT, EnvStateT, InfoT], ABC):
    
    def __init__(self, env: JaxEnvProtocol[ObsT, EnvStateT, InfoT], scenario_handler: JaxScenarioDataHandler) -> None:
        """
        Initializes the InstructionWrapper with the environment, creating EncodeModel and CrafTextScenarios.
        
        Parameters:
        - env: The environment to wrap.
        - scenario_handler: The scenario handler to use.
        """
        self.scenario_handler = scenario_handler

        self.env = env
        self.steps = 0

        # Determine the environment key and state structure
        self.environment_key = self.scenario_handler.environment_key
        self.StateStructure = GameData if self.environment_key == 1 else GameDataClassic

        logger.info("Initialized Instruction Wrapper with environment key: %s", self.environment_key)
        # print(self.StateStructure)
        self.n_instructions = len(self.scenario_handler.scenario_data.instructions_list)

    @abstractmethod
    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        pass

    def reset(self, _rng: Array, env_params: Any, instruction_idx: int = -1) -> Tuple[ObsT, TextEnvState[EnvStateT]]:
        """
        Resets the environment and selects a random instruction embedding or token for the new episode.
        """

        obs, state = self.env.reset(_rng, env_params)
        
        idx = jax.lax.cond(
                instruction_idx == -1, 
                lambda: jax.random.randint(_rng, shape=(), minval=0, maxval=self.n_instructions),
                lambda: instruction_idx
            )
        
        # Initialize the state with the selected instruction embedding/token and set success rates to zero
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            idx=idx,
            success_rate=0.0,
            total_success_rate=0.0,
            rng=_rng,
            instruction_done=False,
            checker_id=self.scenario_handler.scenario_data_jax.scenario_checker[idx],
            target_state=self.scenario_handler.scenario_data_jax.arguments.select(idx)
        )
        return obs, state

    def step(
        self,
        _rng: Array,
        env_state: TextEnvState[EnvStateT],
        action: int,
        env_params: Any,
    ) -> Tuple[ObsT, TextEnvState[EnvStateT], Array, Array, InfoT]:
        """
        Takes a step in the environment, checking if the instruction is done, updating success rate and rewards.
        """
        obs, state, reward, done, info = self.env.step(_rng, env_state.env_state, action, env_params)
        
        # Obtain the game data vector for the current state and check instruction completion
        game_data_vector = self.StateStructure.from_state(env_state.env_state, state, action)
                    
        ts = self.scenario_handler.scenario_data_jax.arguments.select(env_state.idx)

        instruction_done = generic_check(game_data_vector, ts, env_state.checker_id, CHECKER_FUNCTIONS)
        
        # If EXPLORE mode - give craftAx reward
        reward = lax.cond(
                    env_state.checker_id != Scenarios.EXPLORE,
                    lambda: reward / 50,
                    lambda: reward
                )
       # reward = jax.lax.cond(instruction_done, lambda _: reward + 1, lambda _: reward, operand=None)
        done = instruction_done | done
   
        new_episode_sr = env_state.success_rate + jnp.float32(instruction_done)

        # Update state with the new success rates
        state = TextEnvState(
            env_state=state,
            timestep=state.timestep,
            idx=env_state.idx,
            success_rate=new_episode_sr * (1 - done),
            total_success_rate=env_state.total_success_rate * (1 - done) + new_episode_sr * done,
            rng=env_state.rng,
            instruction_done=instruction_done,
            checker_id=env_state.checker_id,
            target_state=env_state.target_state
        )
        
        # Update step information in info dictionary
        info.update({"SR": state.total_success_rate, "steps": self.steps})
        info.update({"Cheker_id": env_state.checker_id})
        self.steps += 1
        return obs, state, reward, done, info

class EncodedInstructionWrapper(BaseInstructionWrapper[ObsT, EnvStateT, Dict[str, Any]]):
    def __init__(self, env: JaxEnvProtocol[ObsT, EnvStateT, Dict[str, Any]], scenario_handler: JaxScenarioDataHandler) -> None:
        # Here we expect a handler that has been created with an EncodedProcessor
        super().__init__(env, scenario_handler)
        if not hasattr(self.scenario_handler.scenario_data_jax, 'embeddings_list'):
            raise ValueError("EncodedInstructionWrapper requires a scenario handler that produces embeddings.")

    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        instructions_emb = self.scenario_handler.scenario_data_jax.embeddings_list[idx]
        instruction_text = self.scenario_handler.scenario_data.instructions_list[idx]
        return instructions_emb, instruction_text

class RawInstructionWrapper(BaseInstructionWrapper[ObsT, EnvStateT, Dict[str, Any]]):
    def _get_instruction(self, idx: int) -> Tuple[Optional[Array], str]:
        instruction_text = self.scenario_handler.scenario_data.instructions_list[idx]
        return None, instruction_text

 
 
