"""Shared representation interfaces for scenario data to JAX conversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from craftext.environment.scenarious.checkers.target_state import TargetState
from craftext.environment.scenarious.manager import BaseScenarioData
from craftext.environment.scenarious.encoded_support import EncodedScenarioData
import logging

from typing import Generic, TypeVar

logger = logging.getLogger(__name__)
