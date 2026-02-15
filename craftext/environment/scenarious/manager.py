"""Scenario handlers and JAX-ready scenario assembly for core CrafText."""

from craftext.environment.scenarious.processors import ScenarioProcessor, EncodedProcessor
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union, Type
from craftext.environment.scenarious.loader import CraftextScenariosConfigLoader, load_scenarios, ScenariosConfig
from craftext.environment.scenarious.base import AbstractScenarioHandler
from craftext.environment.scenarious.instruction_transformers import AbstractInstructionTransformer, DefaultInstructionTransformer, PlansInstructionTransformer
from craftext.environment.craftext_constants import plans_path
from craftext.environment.scenarious.checkers.target_state import TargetState

import logging
from typing import TypeVar, Generic
from jax import numpy as jnp
import jax

from abc import ABC, abstractmethod

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ScenarioFieldType(Enum):
    """Schema behavior for scenario fields during expansion."""

    SINGLE_VALUE = "single_value"  # The base instruction (not copied)
    PARAPHRASE_LIST = "paraphrase_list"  # A list of paraphrases (added to the base instruction)
    REPEAT_WITH_PARAPHRASES = "repeat_with_paraphrases"  # Repeated for each instruction and its paraphrases

SCENARIO_SCHEMA = {
    "instruction": ScenarioFieldType.SINGLE_VALUE,  
    "instruction_paraphrases": ScenarioFieldType.PARAPHRASE_LIST,  
    "scenario_checker": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
    "arguments": ScenarioFieldType.REPEAT_WITH_PARAPHRASES,  
}

@dataclass
class BaseScenarioData:
    """Materialized scenario rows before JAX conversion.

    Attributes:
        instructions_list: Flattened list of instructions (with paraphrases).
        scenario_checker: Checker enum values aligned with instructions.
        arguments: Checker argument objects aligned with instructions.
        scenario_names: Scenario identifiers aligned with instructions.
    """

    instructions_list: List[str]
    scenario_checker: List[int] # Changed from int to List[int]
    arguments: List[TargetState]
    scenario_names: List[str]

@dataclass
class BaseScenarioDataJAX:
    """JAX-serializable scenario tensors used at runtime."""

    scenario_checker: jax.Array
    arguments: TargetState

scenario_data_type = TypeVar('scenario_data_type')

class JAXRepresentation(ABC, Generic[scenario_data_type]):
    """Abstract converter from Python scenario payloads to JAX payloads."""

    @abstractmethod
    def convert(self, scenarios_data: scenario_data_type) -> BaseScenarioDataJAX:
        """Convert scenario data object into JAX-ready representation.

        Args:
            scenarios_data: Scenario data in Python/native structures.

        Returns:
            BaseScenarioDataJAX: JAX-friendly scenario representation.
        """
        ...


class DefaultJAXRepresentation(JAXRepresentation):
    """Default converter for non-encoded scenario data."""

    def convert(self, scenario_data: BaseScenarioData) -> BaseScenarioDataJAX:
        """Convert base scenario data to JAX tensors.

        Args:
            scenario_data: Scenario data without instruction embeddings.

        Returns:
            BaseScenarioDataJAX: JAX-ready checkers and arguments.
        """
        scenario_checker_jax = self._prepare_jax_checkers(scenario_data.scenario_checker)
        
        data = BaseScenarioDataJAX(
            scenario_checker=scenario_checker_jax,
            arguments=TargetState().stack(scenario_data.arguments),
        )
        
        return data

    def _prepare_jax_checkers(self, checkers_list):
        """Convert checker enums into integer JAX array.

        Args:
            checkers_list: Sequence of checker enum values.

        Returns:
            jax.Array: Integer checker ids.
        """
        logger.info("Preparing JAX checkers: %s", len(checkers_list))
        return jnp.array(list(map(lambda x: x.value, checkers_list)))

class BaseScenarioDataHandler(AbstractScenarioHandler):
    """Load, expand, and preprocess raw scenario definitions."""

    def __init__(self, scenario_processor: Type[ScenarioProcessor], instruction_transformer: Type[AbstractInstructionTransformer], config_name: str) -> None:
        """Build scenario data from config for one environment family.

        Args:
            scenario_processor: Processor class for instruction preprocessing.
            instruction_transformer: Transformer class for instruction rewriting.
            config_name: Scenario config name to load.
        """
        super().__init__()
        self.scenario_processor = scenario_processor()
        self.instruction_transformer = instruction_transformer()
        self.config: ScenariosConfig = self._load_config(config_name)
        self.use_paraphrases: bool = self.config.use_parafrases
        self.environment_key: int = 0 if "Classic" in self.config.base_environment else 1
        self.n_instructions: int = 0
        self.instruction_to_update_file: str = plans_path  # This remains as a constant path for the plans
        self._post_config_loaded()
        self.all_scenario: Dict[str, Any] = self._load_scenarios(self.config)
        self.scenario_data: BaseScenarioData = self._prepare_scenarios()

    def _load_config(self, config_name: str) -> ScenariosConfig:
        """Load validated scenario config for current handler.

        Args:
            config_name: Scenario config identifier.

        Returns:
            ScenariosConfig: Parsed config object.
        """
        return CraftextScenariosConfigLoader().load_config(config_name)

    def _post_config_loaded(self) -> None:
        """Hook for subclasses to initialize extra config-derived fields."""
        return

    @property
    def initial_instruction(self) -> List[Any]:
        """Return one default processed instruction sample.

        Returns:
            List[Any]: Single-item processed instruction payload.
        """
        processed_items = self.scenario_processor.process(["None"])
        return processed_items[:1]
    

    def castom_initial_instruction(self, instruction: str) -> List[Any]:
        """Return one processed sample for a custom instruction.

        Args:
            instruction: Raw instruction string.

        Returns:
            List[Any]: Single-item processed instruction payload.
        """
        processed_items = self.scenario_processor.process([instruction])
        return processed_items[:1]

    def _load_scenarios(self, config: ScenariosConfig) -> Dict[str, Any]:
        """Load raw scenarios dictionary for config.

        Args:
            config: Parsed scenarios config.

        Returns:
            Dict[str, Any]: Raw scenario mapping.
        """
        return load_scenarios(config)

    def get_scenarios(self) -> Union[BaseScenarioData, Any]:
        """Return processed scenario data structure.

        Returns:
            Union[BaseScenarioData, Any]: Materialized scenario data.
        """
        
        return self.scenario_data

    
    def _prepare_scenarios(self) -> Union[BaseScenarioData, Any]:
        """Materialize scenario rows and optional embeddings.

        Returns:
            Union[BaseScenarioData, Any]: Processed scenario data.
        """
        instructions_list: List[str] = []
        checker_indecies:  List[int] = []
        arguments:         List[TargetState] = []
        names:             List[str] = []
        keys = list(self.all_scenario.keys())
        
        for name in tqdm(keys):
            entry = self.all_scenario[name]
            current_instr = entry.get("instruction")
            current_checker_index = entry.get("scenario_checker")
            current_paraphrases = entry.get("instruction_paraphrases", [])
            current_arguments = entry.get("arguments")
            instructions_list.append(current_instr)
            checker_indecies.append(current_checker_index)
            arguments.append(current_arguments)
            names.append(name)
            if self.use_paraphrases:
                for para in current_paraphrases:
                    names.append(f"{name}_PARA")
                    instructions_list.append(para)
                    checker_indecies.append(current_checker_index)
                    arguments.append(current_arguments)
                    
        is_encoded: bool = isinstance(self.scenario_processor, EncodedProcessor)

        if is_encoded:
            from craftext.environment.scenarious.encoded_support import EncodedScenarioData

            self.scenario_data = EncodedScenarioData(
                instructions_list=instructions_list,
                scenario_checker=checker_indecies,
                arguments=arguments,
                scenario_names=names,
                embeddings_list=self.scenario_processor.process(instructions_list)
            )
        else:
            self.scenario_data = BaseScenarioData(
                instructions_list=instructions_list,
                scenario_checker=checker_indecies,
                arguments=arguments,
                scenario_names=names,
            )
        logger.info(f"Prepared {len(instructions_list)} instructions (Encoded: {is_encoded})")
        return self.scenario_data

class JaxScenarioDataHandler(BaseScenarioDataHandler):
    """Scenario handler with additional JAX conversion stage."""

    def __init__(self, scenario_processor: Type[ScenarioProcessor], instruction_transformer: Type[AbstractInstructionTransformer], config_name: str, jax_representation_class: Type[JAXRepresentation]) -> None:
        """Create scenario handler with JAX representation stage.

        Args:
            scenario_processor: Processor class for instruction preprocessing.
            instruction_transformer: Transformer class for instruction rewriting.
            config_name: Scenario config name.
            jax_representation_class: Converter class to JAX representation.
        """
        super().__init__(scenario_processor, instruction_transformer, config_name)
        
        self.jax_representation_converter: JAXRepresentation = jax_representation_class()
        self.scenario_data_jax: Union[BaseScenarioDataJAX, Any] = self.scenarios_to_jax()

    def scenarios_to_jax(self) -> Union[BaseScenarioDataJAX, Any]:
        """Convert loaded scenario data to JAX-friendly structures.

        Returns:
            Union[BaseScenarioDataJAX, Any]: JAX scenario payload.
        """
        return self.jax_representation_converter.convert(self.scenario_data)

def create_scenarios_with_dataset(use_plans_gpt: bool) -> Type[JaxScenarioDataHandler]:
    """Factory producing a scenario handler class with selected transformer.

    Args:
        use_plans_gpt: If ``True``, use plans transformer; otherwise default.

    Returns:
        Type[JaxScenarioDataHandler]: Configured handler class.
    """

    class CustomCrafTextScenariosWithPlans(JaxScenarioDataHandler):
        """Concrete scenario handler class with fixed transformer strategy."""

        def __init__(self, scenario_processor: Type[ScenarioProcessor], config_name: str) -> None:
            """Initialize configured inner scenario handler.

            Args:
                scenario_processor: Processor class for instructions.
                config_name: Scenario config name.
            """
            instruction_transformer: Type[AbstractInstructionTransformer] = PlansInstructionTransformer if use_plans_gpt else DefaultInstructionTransformer
            super().__init__(scenario_processor, instruction_transformer, jax_representation_class=DefaultJAXRepresentation, config_name=config_name)
    return CustomCrafTextScenariosWithPlans
