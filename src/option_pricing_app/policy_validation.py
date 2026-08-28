"""Paired-seed, out-of-sample validation of the LSMC American stopping policy.

Section 4.7-style design: for each replication, fit the LSMC policy on one simulated
path set (the "training" seed) and separately evaluate that frozen policy on an
independently simulated path set (the "valuation" seed). Reports both the same-sample
estimate (fit and evaluated on the same paths -- the estimator ordinarily reported,
known to be biased high because the stopping rule was chosen to fit that exact data)
and the independent estimate (Glasserman, *Monte Carlo Methods in Financial
Engineering*, 2004, Sec. 8.6, describes this as a valid low estimator of the true
price), alongside a CRR binomial reference restricted to the same exercise grid.

The paired difference (same-sample minus independent) is reported as an in-sample
reuse gap, not presented as a formal high-bias estimator -- a single replication pair
is not enough to estimate a bias, only to see whether reuse matters here.

GBM only: the CRR binomial reference assumes constant-volatility lognormal dynamics.
This module deliberately contains no Streamlit or Plotly imports.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from option_pricing_app.domain import MarketInputs, PutContract, SimulationConfig
from option_pricing_app.lsmc_policy import (
    evaluate_independent_policy,
    fit_lsmc_policy,
    generate_paths,
)
from option_pricing_app.service import crr_american_put_price

# A fixed constant, not derived from the live simulation's seed: the point of this
# study is a reproducible table for the dissertation, not a session-varying result.
PAIRED_VALIDATION_BASE_SEED = 123


def matched_grid_crr_price(contract: PutContract, market: MarketInputs, n_steps: int) -> float:
    """CRR value restricted to the same exercise dates as the LSMC policy.

    An apples-to-apples reference for a policy that can only decide at ``n_steps``
    exercise dates, rather than continuously. Equals the ordinary (near-continuous)
    CRR reference exactly when ``n_steps`` equals ``CRR_BINOMIAL_STEPS``, since both
    are then the same binomial-tree computation.
    """
    return crr_american_put_price(contract, market, steps=n_steps)


def derive_paired_seeds(base_seed: int, replications: int) -> tuple[tuple[int, int], ...]:
    """Derive independent (training, valuation) seed pairs for each replication."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not 3 <= replications <= 20:
        raise ValueError("replications must be between 3 and 20")
    root = np.random.SeedSequence(base_seed)
    pairs = []
    for replication_seed in root.spawn(replications):
        training_child, valuation_child = replication_seed.spawn(2)
        pairs.append(
            (
                int(training_child.generate_state(1, dtype=np.uint32)[0]),
                int(valuation_child.generate_state(1, dtype=np.uint32)[0]),
            )
        )
    return tuple(pairs)


@dataclass(frozen=True)
class PairedReplicationResult:
    replication: int
    training_seed: int
    valuation_seed: int
    same_sample_estimate: float
    same_sample_standard_error: float
    independent_estimate: float
    independent_standard_error: float
    paired_difference: float
    error_vs_matched_crr: float


@dataclass(frozen=True)
class PairedValidationStudyResult:
    replications: tuple[PairedReplicationResult, ...]
    mean_same_sample_estimate: float
    same_sample_empirical_sd: float
    mean_independent_estimate: float
    independent_empirical_sd: float
    mean_paired_difference: float
    independent_mean_error_vs_matched_crr: float
    independent_rmse_vs_matched_crr: float
    matched_grid_crr: float
    continuous_crr: float
    contract: PutContract
    market: MarketInputs
    n_paths: int
    n_steps: int
    basis_terms: tuple[str, ...]
    base_seed: int


def run_paired_validation_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    base_seed: int = PAIRED_VALIDATION_BASE_SEED,
    replications: int = 20,
) -> PairedValidationStudyResult:
    """Fit on training paths, evaluate the frozen policy on independent valuation
    paths, and compare both to a matched-exercise-grid CRR reference. GBM only."""
    seed_pairs = derive_paired_seeds(base_seed, replications)
    matched_grid_crr = matched_grid_crr_price(contract, market, config.n_steps)
    continuous_crr = crr_american_put_price(contract, market)

    results = []
    for index, (training_seed, valuation_seed) in enumerate(seed_pairs, start=1):
        training_config = SimulationConfig(
            config.n_paths, config.n_steps, training_seed, 1, config.basis_terms
        )
        policy = fit_lsmc_policy(contract, market, training_config, "GBM", None)
        valuation_config = SimulationConfig(
            config.n_paths, config.n_steps, valuation_seed, 1, config.basis_terms
        )
        valuation_paths, _ = generate_paths(contract, market, valuation_config, "GBM", None)
        evaluation = evaluate_independent_policy(policy, valuation_paths, None)
        results.append(
            PairedReplicationResult(
                replication=index,
                training_seed=training_seed,
                valuation_seed=valuation_seed,
                same_sample_estimate=policy.same_sample_price,
                same_sample_standard_error=policy.same_sample_standard_error,
                independent_estimate=evaluation.mean_estimate,
                independent_standard_error=evaluation.standard_error,
                paired_difference=policy.same_sample_price - evaluation.mean_estimate,
                error_vs_matched_crr=evaluation.mean_estimate - matched_grid_crr,
            )
        )

    same_sample = np.array([r.same_sample_estimate for r in results])
    independent = np.array([r.independent_estimate for r in results])
    paired_difference = np.array([r.paired_difference for r in results])
    errors = independent - matched_grid_crr

    return PairedValidationStudyResult(
        replications=tuple(results),
        mean_same_sample_estimate=float(np.mean(same_sample)),
        same_sample_empirical_sd=float(np.std(same_sample, ddof=1)),
        mean_independent_estimate=float(np.mean(independent)),
        independent_empirical_sd=float(np.std(independent, ddof=1)),
        mean_paired_difference=float(np.mean(paired_difference)),
        independent_mean_error_vs_matched_crr=float(np.mean(errors)),
        independent_rmse_vs_matched_crr=float(np.sqrt(np.mean(errors**2))),
        matched_grid_crr=matched_grid_crr,
        continuous_crr=continuous_crr,
        contract=contract,
        market=market,
        n_paths=config.n_paths,
        n_steps=config.n_steps,
        basis_terms=tuple(config.basis_terms),
        base_seed=base_seed,
    )


def paired_validation_to_csv(study: PairedValidationStudyResult) -> str:
    """Return all paired replications, with both seeds, as reproducible CSV."""
    output = io.StringIO()
    fieldnames = [
        "replication",
        "training_seed",
        "valuation_seed",
        "same_sample_estimate",
        "same_sample_standard_error",
        "independent_estimate",
        "independent_standard_error",
        "paired_difference",
        "error_vs_matched_crr",
        "matched_grid_crr",
        "continuous_crr",
        "spot",
        "strike",
        "maturity_years",
        "volatility",
        "risk_free_rate",
        "path_count",
        "time_step_count",
        "basis_terms",
        "base_seed",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in study.replications:
        writer.writerow(
            {
                "replication": row.replication,
                "training_seed": row.training_seed,
                "valuation_seed": row.valuation_seed,
                "same_sample_estimate": row.same_sample_estimate,
                "same_sample_standard_error": row.same_sample_standard_error,
                "independent_estimate": row.independent_estimate,
                "independent_standard_error": row.independent_standard_error,
                "paired_difference": row.paired_difference,
                "error_vs_matched_crr": row.error_vs_matched_crr,
                "matched_grid_crr": study.matched_grid_crr,
                "continuous_crr": study.continuous_crr,
                "spot": study.market.spot,
                "strike": study.contract.strike,
                "maturity_years": study.contract.maturity,
                "volatility": study.market.volatility,
                "risk_free_rate": study.market.risk_free_rate,
                "path_count": study.n_paths,
                "time_step_count": study.n_steps,
                "basis_terms": ", ".join(study.basis_terms),
                "base_seed": study.base_seed,
            }
        )
    return output.getvalue()
