"""Validated inputs and outputs shared by the standalone application."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from option_pricing_app.lsmc_policy import FittedLSMCPolicy


class ExerciseStyle(str, Enum):
    EUROPEAN = "European"
    AMERICAN = "American"


PRICE_BASIS_TERMS = ("1", "S", "S²")
HESTON_BASIS_TERMS = (
    "1",
    "S",
    "S²",
    "v",
    "v²",
    "S·v",
    "S²·v",
    "S·v²",
    "S²·v²",
)


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
class HestonInputs:
    mean_reversion_speed: float = 2.0
    long_run_variance: float = 0.04
    volatility_of_variance: float = 0.30
    correlation: float = -0.70
    initial_variance: float = 0.04

    def __post_init__(self) -> None:
        _positive("mean_reversion_speed", self.mean_reversion_speed)
        _positive("long_run_variance", self.long_run_variance)
        _positive("volatility_of_variance", self.volatility_of_variance)
        _positive("initial_variance", self.initial_variance)
        if not np.isfinite(self.correlation) or not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be finite and between -1 and 1")

    @property
    def satisfies_feller_condition(self) -> bool:
        return (
            2.0 * self.mean_reversion_speed * self.long_run_variance
            >= self.volatility_of_variance**2
        )


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int = 1_000
    n_steps: int = 100
    seed: int | None = 42
    max_display_paths: int = 100
    basis_terms: tuple[str, ...] = PRICE_BASIS_TERMS

    def __post_init__(self) -> None:
        if isinstance(self.n_paths, bool) or not isinstance(self.n_paths, int):
            raise TypeError("n_paths must be an integer")
        if isinstance(self.n_steps, bool) or not isinstance(self.n_steps, int):
            raise TypeError("n_steps must be an integer")
        if not 100 <= self.n_paths <= 25_000:
            raise ValueError("n_paths must be between 100 and 25,000")
        if not 1 <= self.n_steps <= 1_000:
            raise ValueError("n_steps must be between 1 and 1,000")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")
        if not 1 <= self.max_display_paths <= 100:
            raise ValueError("max_display_paths must be between 1 and 100")
        if not self.basis_terms:
            raise ValueError("at least one LSMC basis term must be selected")
        unknown_terms = set(self.basis_terms) - set(HESTON_BASIS_TERMS)
        if unknown_terms:
            raise ValueError(f"unsupported LSMC basis terms: {sorted(unknown_terms)}")


@dataclass(frozen=True)
class LSMCContinuationDiagnostic:
    """Chart-ready slice of one fitted LSMC continuation regression."""

    timestep: int
    time: float
    time_to_maturity: float
    itm_path_count: int
    minimum_required_paths: int
    basis_terms: tuple[str, ...]
    price_grid: np.ndarray
    immediate_payoff: np.ndarray
    continuation_value: np.ndarray
    boundary: float
    representative_variance: float | None


@dataclass(frozen=True)
class PricingResult:
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    time_grid: np.ndarray
    displayed_paths: np.ndarray
    path_quantile_05: np.ndarray
    path_quantile_95: np.ndarray
    risk_neutral_expected_path: np.ndarray
    displayed_variance_paths: np.ndarray | None
    variance_quantile_05: np.ndarray | None
    variance_quantile_95: np.ndarray | None
    expected_variance_path: np.ndarray | None
    long_run_variance: float | None
    terminal_prices: np.ndarray
    terminal_payoffs: np.ndarray
    discounted_realised_cash_flows: np.ndarray
    exercise_style: ExerciseStyle
    model_name: str
    pricing_method: str
    european_mc_price: float
    european_mc_standard_error: float
    european_exact_price: float | None
    convergence_path_counts: np.ndarray
    convergence_estimates: np.ndarray
    convergence_lower: np.ndarray
    convergence_upper: np.ndarray
    exercise_percentages: np.ndarray | None
    continuation_diagnostics: tuple[LSMCContinuationDiagnostic, ...] | None
    fitted_policy: FittedLSMCPolicy | None


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
