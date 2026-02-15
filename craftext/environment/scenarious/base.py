"""Abstract interfaces for scenario handlers used by CrafText environments."""

from abc import ABC, abstractmethod


class AbstractScenarioHandler(ABC):
    
    @property
    @abstractmethod
    def initial_instruction(self):
        pass

    @abstractmethod
    def castom_initial_instruction(self, instruction):
        pass

    @abstractmethod
    def _load_scenarios(self, config):
        pass

    @abstractmethod
    def get_scenarios(self):
        pass

    @abstractmethod
    def _prepare_scenarios(self):
        pass
        
