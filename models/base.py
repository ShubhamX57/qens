from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Sequence
import numpy as np


@dataclass
class ParameterSpec:
    name: str
    lower: float | None = None
    upper: float | None = None
    default: float | None = None
    prior: str = "uniform"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelSpec:
    name: str
    description: str
    parameters: List[ParameterSpec] = field(default_factory=list)
    evaluator: Callable[[np.ndarray, Dict[str, float]], np.ndarray] | None = None

    def evaluate(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        if self.evaluator is None:
            raise NotImplementedError(f"Model '{self.name}' does not define an evaluator.")
        return self.evaluator(x, params)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("evaluator", None)
        data["parameters"] = [p.to_dict() for p in self.parameters]
        return data


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, model: ModelSpec) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> ModelSpec:
        return self._models[name]

    def names(self) -> list[str]:
        return sorted(self._models.keys())

    def all(self) -> list[ModelSpec]:
        return [self._models[n] for n in self.names()]


__all__ = ["ParameterSpec", "ModelSpec", "ModelRegistry"]
