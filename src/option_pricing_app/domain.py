"""Validated inputs and outputs shared by the standalone application."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ExerciseStyle(str, Enum):
    EUROPEAN = "European"
    AMERICAN = "American"


@dataclass(frozen=True)
class PutContract:
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        _positive("strike", self.strike)
        _positive("maturity", self.maturity)


@dataclass(frozen=True)
class MarketInputs:
    spot: float
    volatility: float
    risk_free_rate: float

    def __post_init__(self) -> None:
        _positive("spot", self.spot)
        _positive("volatility", self.volatility)
        if not np.isfinite(self.risk_free_rate) or not -1.0 < self.risk_free_rate < 1.0:
            raise ValueError("risk_free_rate must be finite and between -1 and 1")


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 5_000
    n_steps: int = 100
    seed: int | None = 42
    max_display_paths: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.n_paths, bool) or not isinstance(self.n_paths, int):
            raise TypeError("n_paths must be an integer")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int):
            raise TypeError("n_steps must be an integer")
        if not 100 <= self.n_paths <= 25_000:
            raise ValueError("n_paths must be between 100 and 25,000")
        if not 1 <= self.n_steps <= 500:
            raise ValueError("n_steps must be between 1 and 500")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if not 1 <= self.max_display_paths <= 100:
            raise ValueError("max_display_paths must be between 1 and 100")


@dataclass(frozen=True)
class PricingResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    time_grid: np.ndarray
    displayed_paths: np.ndarray
    terminal_prices: np.ndarray
    terminal_payoffs: np.ndarray
    exercise_style: ExerciseStyle
    model_name: str
    pricing_method: str


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
