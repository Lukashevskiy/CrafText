"""Scenario instruction processors (raw and encoded) for CrafText."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, List


class ScenarioProcessor(ABC):
    @abstractmethod
    def process(self, instructions: List[str]) -> Tuple[Any, List[str], int]:
        """
        Process a list of instructions.
        Returns a tuple of (processed_items, original_instructions, num_variants).
        - processed_items: Can be embeddings or other processed data. For raw, it will be None.
        - original_instructions: The original list of instructions, maybe modified (e.g. plans).
        - num_variants: Number of variants generated per instruction.
        """
        pass

class EncodedProcessor(ScenarioProcessor):
    def __init__(self, encode_model):
        self.encode_model = encode_model

    def process(self, instructions: List[str]) -> Any:
        encoded_instructions = self.encode_model.encode(instructions)
        # Assuming encode_model might return multiple variants per instruction
        if len(instructions) == 0:
            return []
        num_variants = len(encoded_instructions) // len(instructions)
        assert len(encoded_instructions) == len(instructions) * num_variants, \
            f"Unexpected size of encoded instructions ({len(encoded_instructions)} vs {len(instructions)}). Ensure encode_model is consistent."
        
        return encoded_instructions

class RawProcessor(ScenarioProcessor):
    def process(self, instructions: List[str]) -> Any:
        # For raw scenarios, embeddings are None and there's only one variant.
        return [None] * len(instructions)
