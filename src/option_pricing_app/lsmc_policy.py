"""Frozen LSMC stopping-policy fitting and out-of-sample evaluation.

Splits what ``service.price_put`` normally does in one pass -- simulate paths, fit the
LSMC continuation regressions, and price on those same paths -- into a fit step and a
separate evaluation step. This lets a policy fitted on one path set be applied,
unchanged, to an independently simulated path set: the same-sample price is a biased
estimator (the stopping rule was chosen to maximise value on that exact data), while
evaluating the frozen rule on fresh paths gives a valid low estimator of the true price
(Glasserman, *Monte Carlo Methods in Financial Engineering*, 2004, Sec. 8.6).

This module deliberately contains no Streamlit or Plotly imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from option_pricing_app.domain import HestonInputs, MarketInputs, PutContract, SimulationConfig
from option_pricing_app.stats import GBMPriceSimulator, HestonPriceSimulator, LSMCPriceSimulator
from option_pricing_app.stats.option_pricing_simulator import ContinuationRegression


def create_asset_path_simulator(model: str):
    if model == "GBM":
        return GBMPriceSimulator()
    if model == "Heston":
        return HestonPriceSimulator()
    raise ValueError(f"Unsupported asset model: {model}")


def generate_paths(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Seed the RNG from ``config.seed`` and simulate price (and variance) paths."""
    simulator = create_asset_path_simulator(model)
    if config.seed is not None:
        np.random.seed(config.seed)
    if model == "GBM":
        paths = simulator.generate_price_paths(
            market.spot,
            contract.maturity,
            market.risk_free_rate,
            market.volatility,
            config.n_paths,
            config.n_steps,
        )
        return paths, None
    if heston_inputs is None:
        raise ValueError("Heston parameters are required for the Heston model")
    paths, variance_paths = simulator.generate_price_paths(
        market.spot,
        contract.maturity,
        market.risk_free_rate,
        heston_inputs.mean_reversion_speed,
        heston_inputs.long_run_variance,
        heston_inputs.volatility_of_variance,
        heston_inputs.correlation,
        heston_inputs.initial_variance,
        config.n_paths,
        config.n_steps,
    )
    return paths, variance_paths


@dataclass(frozen=True)
class FittedLSMCPolicy:
    """A frozen LSMC stopping policy: continuation regressions plus the fixed rules
    needed to apply them, unchanged, to a different set of simulated paths."""

    continuation_models: MappingProxyType[int, ContinuationRegression]
    strike: float
    maturity: float
    risk_free_rate: float
    dt: float
    n_steps: int
    basis_terms: tuple[str, ...]
    exercise_immediately_at_zero: bool
    same_sample_price: float
    same_sample_standard_error: float


def wrap_fitted_policy(
    lsmc: LSMCPriceSimulator,
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    dt: float,
    price: float,
    standard_error: float,
) -> FittedLSMCPolicy:
    """Freeze an already-run ``backward_induction`` fit into a ``FittedLSMCPolicy``.

    ``ContinuationRegression`` instances are already frozen dataclasses holding copied
    coefficient and column-scale arrays; wrapping the dict in a read-only view is
    enough to stop the caller's mutable ``lsmc`` instance from being aliased here.
    """
    immediate_exercise_at_zero = max(contract.strike - market.spot, 0.0)
    exercise_immediately_at_zero = (
        immediate_exercise_at_zero > 0 and immediate_exercise_at_zero >= price
    )
    return FittedLSMCPolicy(
        continuation_models=MappingProxyType(dict(lsmc.continuation_models)),
        strike=contract.strike,
        maturity=contract.maturity,
        risk_free_rate=market.risk_free_rate,
        dt=dt,
        n_steps=config.n_steps,
        basis_terms=tuple(config.basis_terms),
        exercise_immediately_at_zero=exercise_immediately_at_zero,
        same_sample_price=price,
        same_sample_standard_error=standard_error,
    )


def fit_lsmc_policy(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
) -> FittedLSMCPolicy:
    """Simulate training paths and fit the LSMC stopping policy, then freeze it.

    ``config.seed`` selects the training path set. The returned policy's continuation
    regressions are evaluated, not refit, by ``evaluate_independent_policy`` on a
    separately simulated path set.
    """
    paths, variance_paths = generate_paths(contract, market, config, model, heston_inputs)
    dt = contract.maturity / config.n_steps
    lsmc = LSMCPriceSimulator()
    price, standard_error = lsmc.backward_induction(
        paths, contract.strike, market.risk_free_rate, dt, config.basis_terms, variance_paths
    )
    return wrap_fitted_policy(lsmc, contract, market, config, dt, price, standard_error)


@dataclass(frozen=True)
class IndependentPolicyEvaluation:
    """Result of applying a frozen policy to an independent path set."""

    mean_estimate: float
    standard_error: float
    exercise_steps: np.ndarray
    discounted_payoffs: np.ndarray


def evaluate_independent_policy(
    policy: FittedLSMCPolicy,
    price_paths: np.ndarray,
    variance_paths: np.ndarray | None = None,
) -> IndependentPolicyEvaluation:
    """Apply a frozen LSMC policy to an independent path set without refitting.

    Moves forward through the exercise dates: at each date with a fitted continuation
    regression, surviving in-the-money paths are exercised if their immediate payoff
    exceeds the regression's (unrefit) predicted continuation value. If no regression
    was fitted at a date -- no in-the-money training paths there -- the policy
    continues at that date. Every path is exercised at most once, at its earliest
    selected date; the realised payoff is discounted from that date back to time zero.
    """
    n_time_points, n_paths = price_paths.shape
    if n_time_points != policy.n_steps + 1:
        raise ValueError("price_paths time dimension does not match the fitted policy")

    if policy.exercise_immediately_at_zero:
        payoff0 = max(policy.strike - float(price_paths[0, 0]), 0.0)
        return IndependentPolicyEvaluation(
            mean_estimate=payoff0,
            standard_error=0.0,
            exercise_steps=np.zeros(n_paths, dtype=int),
            discounted_payoffs=np.full(n_paths, payoff0, dtype=float),
        )

    alive = np.ones(n_paths, dtype=bool)
    exercise_steps = np.full(n_paths, -1, dtype=int)
    payoffs_at_exercise = np.zeros(n_paths, dtype=float)

    for t in range(1, n_time_points - 1):
        model = policy.continuation_models.get(t)
        if model is None or not np.any(alive):
            continue
        prices_t = price_paths[t]
        itm = alive & (prices_t < policy.strike)
        if not np.any(itm):
            continue
        variances_t = None if variance_paths is None else variance_paths[t, itm]
        continuation_value = model.predict(prices_t[itm], variances_t)
        immediate_payoff = policy.strike - prices_t[itm]
        exercise_now = immediate_payoff > continuation_value
        if not np.any(exercise_now):
            continue
        itm_indices = np.flatnonzero(itm)
        exercising_indices = itm_indices[exercise_now]
        exercise_steps[exercising_indices] = t
        payoffs_at_exercise[exercising_indices] = immediate_payoff[exercise_now]
        alive[exercising_indices] = False

    maturity_prices = price_paths[-1]
    maturity_itm = alive & (maturity_prices < policy.strike)
    maturity_indices = np.flatnonzero(maturity_itm)
    if maturity_indices.size:
        exercise_steps[maturity_indices] = n_time_points - 1
        payoffs_at_exercise[maturity_indices] = policy.strike - maturity_prices[maturity_indices]

    discounted = payoffs_at_exercise * np.exp(
        -policy.risk_free_rate * policy.dt * np.maximum(exercise_steps, 0)
    )
    return IndependentPolicyEvaluation(
        mean_estimate=float(np.mean(discounted)),
        standard_error=float(np.std(discounted) / np.sqrt(n_paths)),
        exercise_steps=exercise_steps,
        discounted_payoffs=discounted,
    )
