"""Early-exercise policy summaries and Longstaff-Schwartz-style validation tables.

This module deliberately contains no Streamlit or Plotly imports, mirroring the
separation between computation and presentation used in ``numerical_studies``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from option_pricing_app.domain import (
    ExerciseStyle,
    MarketInputs,
    PricingResult,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.numerical_studies import derive_replication_seeds
from option_pricing_app.service import black_scholes_put_price, crr_american_put_price, price_put

# Mirrors Longstaff & Schwartz (2001), Table 1, which fixes the strike and risk-free
# rate and varies spot around the strike (36/38/40/42/44 for K=40, i.e. -10%/-5%/
# ATM/+5%/+10%), volatility (.2 and .4, a 2x pair) and maturity (1 and 2 years, also
# a 2x pair), for 5 x 2 x 2 = 20 rows.
VALIDATION_MONEYNESS_GRID = (0.90, 0.95, 1.00, 1.05, 1.10)
VALIDATION_VOLATILITY_MULTIPLIERS = (1.0, 2.0)
VALIDATION_MATURITY_MULTIPLIERS = (1.0, 2.0)


@dataclass(frozen=True)
class EarlyExercisePolicyRow:
    method: str
    american_value: float
    american_standard_error: float | None
    european_value: float
    early_exercise_value: float


def build_early_exercise_policy_table(
    result: PricingResult,
    contract: PutContract,
    market: MarketInputs,
    model: str,
) -> tuple[EarlyExercisePolicyRow, ...]:
    """Longstaff & Schwartz (2001), Table 1-style summary for the current run.

    Reports the American value, European value and their difference (the early
    exercise value) from an independent reference -- a CRR binomial tree plus the
    closed-form European price, standing in for the paper's finite-difference
    benchmark -- and from the LSMC simulation. The reference row is only included
    under GBM, since CRR and the closed-form price both assume constant-volatility
    lognormal dynamics.
    """
    rows: list[EarlyExercisePolicyRow] = []
    if model == "GBM":
        crr_american = crr_american_put_price(contract, market)
        closed_form_european = black_scholes_put_price(contract, market)
        rows.append(
            EarlyExercisePolicyRow(
                method="CRR binomial / closed-form (reference)",
                american_value=crr_american,
                american_standard_error=None,
                european_value=closed_form_european,
                early_exercise_value=crr_american - closed_form_european,
            )
        )
    rows.append(
        EarlyExercisePolicyRow(
            method="LSMC (simulation)",
            american_value=result.price,
            american_standard_error=result.standard_error,
            european_value=result.european_mc_price,
            early_exercise_value=result.price - result.european_mc_price,
        )
    )
    return tuple(rows)


def early_exercise_policy_to_csv(rows: tuple[EarlyExercisePolicyRow, ...]) -> str:
    """Return the early-exercise policy table as run-level, reproducible CSV."""
    output = io.StringIO()
    fieldnames = [
        "method",
        "american_value",
        "american_standard_error",
        "european_value",
        "early_exercise_value",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "method": row.method,
                "american_value": row.american_value,
                "american_standard_error": (
                    row.american_standard_error
                    if row.american_standard_error is not None
                    else ""
                ),
                "european_value": row.european_value,
                "early_exercise_value": row.early_exercise_value,
            }
        )
    return output.getvalue()


@dataclass(frozen=True)
class ValidationRow:
    spot: float
    volatility: float
    maturity: float
    seed: int
    crr_american: float
    closed_form_european: float
    crr_early_exercise_value: float
    lsmc_american: float
    lsmc_standard_error: float
    lsmc_early_exercise_value: float
    difference_in_early_exercise_value: float


def make_validation_grid(
    strike: float, volatility: float, maturity: float
) -> tuple[tuple[float, float, float], ...]:
    """Return the (spot, volatility, maturity) grid, anchored to the current inputs."""
    return tuple(
        (round(strike * moneyness, 10), volatility * vol_mult, maturity * mat_mult)
        for moneyness in VALIDATION_MONEYNESS_GRID
        for vol_mult in VALIDATION_VOLATILITY_MULTIPLIERS
        for mat_mult in VALIDATION_MATURITY_MULTIPLIERS
    )


def run_numerical_validation_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    base_seed: int,
) -> tuple[ValidationRow, ...]:
    """Longstaff & Schwartz (2001), Table 1-style parameter-sweep validation.

    Compares LSMC American puts against an independent CRR binomial reference (in
    place of the paper's finite-difference benchmark) across a grid of spot prices,
    volatilities and maturities anchored to the current contract and market inputs,
    holding the strike and risk-free rate fixed -- the same design as Table 1. Each
    grid point uses a single simulation with its own derived seed, matching the
    paper's one-run-per-row design; the reported standard error is that single run's
    Monte Carlo standard error, not an empirical replication spread. GBM only.
    """
    grid = make_validation_grid(contract.strike, market.volatility, contract.maturity)
    seeds = derive_replication_seeds(base_seed, len(grid))
    rows = []
    for (spot, volatility, maturity), seed in zip(grid, seeds, strict=True):
        row_contract = PutContract(contract.strike, maturity)
        row_market = MarketInputs(spot, volatility, market.risk_free_rate)
        crr_american = crr_american_put_price(row_contract, row_market)
        closed_form_european = black_scholes_put_price(row_contract, row_market)
        crr_early_exercise_value = crr_american - closed_form_european
        result = price_put(
            row_contract,
            row_market,
            SimulationConfig(config.n_paths, config.n_steps, seed, 1, config.basis_terms),
            ExerciseStyle.AMERICAN,
            "GBM",
        )
        lsmc_early_exercise_value = result.price - closed_form_european
        rows.append(
            ValidationRow(
                spot=spot,
                volatility=volatility,
                maturity=maturity,
                seed=seed,
                crr_american=crr_american,
                closed_form_european=closed_form_european,
                crr_early_exercise_value=crr_early_exercise_value,
                lsmc_american=result.price,
                lsmc_standard_error=result.standard_error,
                lsmc_early_exercise_value=lsmc_early_exercise_value,
                difference_in_early_exercise_value=(
                    crr_early_exercise_value - lsmc_early_exercise_value
                ),
            )
        )
    return tuple(rows)


def validation_study_to_csv(rows: tuple[ValidationRow, ...]) -> str:
    """Return the validation grid as run-level, reproducible CSV."""
    output = io.StringIO()
    fieldnames = [
        "spot",
        "volatility",
        "maturity_years",
        "seed",
        "crr_american",
        "closed_form_european",
        "crr_early_exercise_value",
        "lsmc_american",
        "lsmc_standard_error",
        "lsmc_early_exercise_value",
        "difference_in_early_exercise_value",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "spot": row.spot,
                "volatility": row.volatility,
                "maturity_years": row.maturity,
                "seed": row.seed,
                "crr_american": row.crr_american,
                "closed_form_european": row.closed_form_european,
                "crr_early_exercise_value": row.crr_early_exercise_value,
                "lsmc_american": row.lsmc_american,
                "lsmc_standard_error": row.lsmc_standard_error,
                "lsmc_early_exercise_value": row.lsmc_early_exercise_value,
                "difference_in_early_exercise_value": row.difference_in_early_exercise_value,
            }
        )
    return output.getvalue()
