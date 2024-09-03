# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import openvino as ov

from openvino_xai.common.utils import IdentityPreprocessFN


class MethodBase(ABC):
    """Base class for XAI methods."""

    def __init__(
        self,
        model: ov.Model = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        device_name: str = "CPU",
    ):
        self._model = model
        self._model_compiled = None
        self.preprocess_fn = preprocess_fn
        self._device_name = device_name
        self.predictions: Dict[int, Prediction] = {}

    @property
    def model_compiled(self) -> ov.CompiledModel | None:
        return self._model_compiled

    @abstractmethod
    def prepare_model(self, load_model: bool = True) -> ov.Model:
        """Model preparation steps."""

    def model_forward(self, x: np.ndarray, preprocess: bool = True) -> Mapping:
        """Forward pass of the compiled model. Applies preprocess_fn."""
        if not self._model_compiled:
            raise RuntimeError("Model is not compiled. Call prepare_model() first.")
        if preprocess:
            x = self.preprocess_fn(x)
        return self._model_compiled(x)

    @abstractmethod
    def generate_saliency_map(self, data: np.ndarray) -> Dict[int, np.ndarray] | np.ndarray:
        """Saliency map generation."""

    def load_model(self) -> None:
        core = ov.Core()
        self._model_compiled = core.compile_model(model=self._model, device_name=self._device_name)


@dataclass
class Prediction:
    label: int | None = None
    score: float | None = None
    bounding_box: List | Tuple | None = None
