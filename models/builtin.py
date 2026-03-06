from __future__ import annotations

import numpy as np

from .base import ModelSpec, ParameterSpec, ModelRegistry


def lorentzian(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    amp = params["amplitude"]
    gamma = params["gamma"]
    center = params["center"]
    background = params["background"]
    return amp * (gamma**2 / ((x - center) ** 2 + gamma**2)) + background


def gaussian(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    amp = params["amplitude"]
    sigma = params["sigma"]
    center = params["center"]
    background = params["background"]
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background


def build_builtin_registry() -> ModelRegistry:
    registry = ModelRegistry()

    registry.register(
        ModelSpec(
            name="Lorentzian",
            description="Single Lorentzian peak with constant background.",
            parameters=[
                ParameterSpec("amplitude", 0.0, None, 1.0),
                ParameterSpec("gamma", 1e-6, None, 0.1),
                ParameterSpec("center", None, None, 0.0),
                ParameterSpec("background", None, None, 0.0),
            ],
            evaluator=lorentzian,
        )
    )

    registry.register(
        ModelSpec(
            name="Gaussian",
            description="Single Gaussian peak with constant background.",
            parameters=[
                ParameterSpec("amplitude", 0.0, None, 1.0),
                ParameterSpec("sigma", 1e-6, None, 0.1),
                ParameterSpec("center", None, None, 0.0),
                ParameterSpec("background", None, None, 0.0),
            ],
            evaluator=gaussian,
        )
    )

    return registry
