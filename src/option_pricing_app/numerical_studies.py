"""Repeated-run numerical studies built on the existing pricing service.

This module deliberately contains no Streamlit or Plotly imports.  Each experiment
calls ``price_put`` afresh, so every scalar estimate comes from newly simulated paths
and a newly fitted LSMC stopping policy.  Full path arrays are discarded immediately.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from option_pricing_app.domain import (
    HESTON_BASIS_TERMS,
    ExerciseStyle,
    HestonInputs,
    MarketInputs,
    PricingResult,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.service import black_scholes_put_price, price_put

PATH_COUNT_CANDIDATES = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000)
EXERCISE_GRID_CANDIDATES = (5, 10, 25, 50, 100, 250, 500, 1_000)


@dataclass(frozen=True)
class ExperimentRun:
    seed: int
    estimated_price: float
    standard_error: float


@dataclass(frozen=True)
class ExperimentPoint:
    setting_value: int | str
    setting_label: str
    basis_terms: tuple[str, ...]
    runs: tuple[ExperimentRun, ...]
    mean_estimate: float
    empirical_standard_deviation: float
    lower_empirical_quantile: float
    upper_empirical_quantile: float


@dataclass(frozen=True)
class PathCountStudyResult:
    base_seed: int
    derived_seeds: tuple[int, ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    time_steps: int
    basis_terms: tuple[str, ...]
    heston_inputs: HestonInputs | None
    european_reference: float | None


@dataclass(frozen=True)
class ExerciseGridStudyResult:
    base_seed: int
    derived_seeds: tuple[int, ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    path_count: int
    basis_terms: tuple[str, ...]
    heston_inputs: HestonInputs | None


@dataclass(frozen=True)
class BasisSensitivityStudyResult:
    base_seed: int
    derived_seeds: tuple[int, ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    path_count: int
    time_steps: int
    heston_inputs: HestonInputs | None


@dataclass(frozen=True)
class NumericalStudiesResult:
    path_count: PathCountStudyResult
    exercise_grid: ExerciseGridStudyResult | None
    basis_sensitivity: BasisSensitivityStudyResult


PricingFunction = Callable[
    [PutContract, MarketInputs, SimulationConfig, ExerciseStyle, str, HestonInputs | None],
    PricingResult,
]


def make_path_count_grid(max_paths: int) -> tuple[int, ...]:
    """Return a compact increasing grid that always contains ``max_paths``."""
    if max_paths < 100:
        raise ValueError("max_paths must be at least 100")
    return tuple(sorted({value for value in PATH_COUNT_CANDIDATES if value < max_paths} | {max_paths}))


def make_exercise_grid(max_steps: int) -> tuple[int, ...]:
    """Return a compact increasing exercise grid containing the selected maximum."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    return tuple(
        sorted({value for value in EXERCISE_GRID_CANDIDATES if value < max_steps} | {max_steps})
    )


def derive_replication_seeds(base_seed: int, replications: int) -> tuple[int, ...]:
    """Derive deterministic independent child seeds using NumPy SeedSequence."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not 3 <= replications <= 20:
        raise ValueError("replications must be between 3 and 20")
    root = np.random.SeedSequence(base_seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in root.spawn(replications)
    )


def generate_base_seed() -> int:
    """Generate and return an explicit seed for an initially unseeded study."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def basis_specifications(model: str, selected: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Return a small documented collection of nested LSMC specifications."""
    if model == "GBM":
        candidates = [("1",), ("1", "S"), ("1", "S", "S²")]
    elif model == "Heston":
        candidates = [
            ("1", "S"),
            ("1", "S", "S²"),
            ("1", "S", "S²", "v"),
            ("1", "S", "S²", "v", "S·v"),
            HESTON_BASIS_TERMS,
        ]
    else:
        raise ValueError(f"Unsupported asset model: {model}")
    selected = tuple(selected)
    if selected not in candidates:
        candidates.append(selected)
    return tuple(candidates)


def _aggregate_point(
    setting_value: int | str,
    setting_label: str,
    basis_terms: tuple[str, ...],
    runs: list[ExperimentRun],
) -> ExperimentPoint:
    estimates = np.asarray([run.estimated_price for run in runs], dtype=float)
    return ExperimentPoint(
        setting_value=setting_value,
        setting_label=setting_label,
        basis_terms=basis_terms,
        runs=tuple(runs),
        mean_estimate=float(np.mean(estimates)),
        empirical_standard_deviation=float(np.std(estimates, ddof=1)),
        lower_empirical_quantile=float(np.quantile(estimates, 0.025)),
        upper_empirical_quantile=float(np.quantile(estimates, 0.975)),
    )


def _run_point(
    *,
    setting_value: int | str,
    setting_label: str,
    contract: PutContract,
    market: MarketInputs,
    model: str,
    heston_inputs: HestonInputs | None,
    path_count: int,
    time_steps: int,
    basis_terms: tuple[str, ...],
    seeds: tuple[int, ...],
    pricing_function: PricingFunction,
) -> ExperimentPoint:
    runs = []
    for seed in seeds:
        # price_put performs the complete simulation, continuation regressions,
        # stopping decisions and time-zero averaging for this setting and seed.
        result = pricing_function(
            contract,
            market,
            SimulationConfig(path_count, time_steps, seed, 1, basis_terms),
            ExerciseStyle.AMERICAN,
            model,
            heston_inputs,
        )
        runs.append(ExperimentRun(seed, result.price, result.standard_error))
    return _aggregate_point(setting_value, setting_label, basis_terms, runs)


def run_path_count_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    pricing_function: PricingFunction = price_put,
) -> PathCountStudyResult:
    seeds = derive_replication_seeds(base_seed, replications)
    points = tuple(
        _run_point(
            setting_value=path_count,
            setting_label=f"{path_count:,}",
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            path_count=path_count,
            time_steps=config.n_steps,
            basis_terms=config.basis_terms,
            seeds=seeds,
            pricing_function=pricing_function,
        )
        for path_count in make_path_count_grid(config.n_paths)
    )
    european_reference = black_scholes_put_price(contract, market) if model == "GBM" else None
    return PathCountStudyResult(
        base_seed,
        seeds,
        points,
        contract,
        market,
        model,
        config.n_steps,
        config.basis_terms,
        heston_inputs,
        european_reference,
    )


def run_exercise_grid_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    pricing_function: PricingFunction = price_put,
) -> ExerciseGridStudyResult:
    if model != "GBM":
        raise ValueError("The exercise-grid study is currently limited to GBM")
    seeds = derive_replication_seeds(base_seed, replications)
    points = tuple(
        _run_point(
            setting_value=time_steps,
            setting_label=f"{time_steps:,}",
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            path_count=config.n_paths,
            time_steps=time_steps,
            basis_terms=config.basis_terms,
            seeds=seeds,
            pricing_function=pricing_function,
        )
        for time_steps in make_exercise_grid(config.n_steps)
    )
    return ExerciseGridStudyResult(
        base_seed,
        seeds,
        points,
        contract,
        market,
        model,
        config.n_paths,
        config.basis_terms,
        heston_inputs,
    )


def run_basis_sensitivity_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    pricing_function: PricingFunction = price_put,
) -> BasisSensitivityStudyResult:
    seeds = derive_replication_seeds(base_seed, replications)
    specifications = basis_specifications(model, config.basis_terms)
    points = tuple(
        _run_point(
            setting_value=" + ".join(terms),
            setting_label=" + ".join(terms),
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            path_count=config.n_paths,
            time_steps=config.n_steps,
            basis_terms=terms,
            seeds=seeds,
            pricing_function=pricing_function,
        )
        for terms in specifications
    )
    return BasisSensitivityStudyResult(
        base_seed,
        seeds,
        points,
        contract,
        market,
        model,
        config.n_paths,
        config.n_steps,
        heston_inputs,
    )


def run_numerical_studies(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
) -> NumericalStudiesResult:
    """Run all applicable studies sequentially and retain only scalar summaries."""
    path_count = run_path_count_study(
        contract, market, config, model, heston_inputs, base_seed, replications
    )
    exercise_grid = (
        run_exercise_grid_study(
            contract, market, config, model, heston_inputs, base_seed, replications
        )
        if model == "GBM"
        else None
    )
    basis = run_basis_sensitivity_study(
        contract, market, config, model, heston_inputs, base_seed, replications
    )
    return NumericalStudiesResult(path_count, exercise_grid, basis)


def estimate_workload(config: SimulationConfig, model: str, replications: int) -> tuple[int, int]:
    """Return (pricing runs, simulated path-steps) without executing a study."""
    path_grid = make_path_count_grid(config.n_paths)
    exercise_grid = make_exercise_grid(config.n_steps) if model == "GBM" else ()
    specifications = basis_specifications(model, config.basis_terms)
    runs = replications * (len(path_grid) + len(exercise_grid) + len(specifications))
    path_steps = replications * (
        config.n_steps * sum(path_grid)
        + config.n_paths * sum(exercise_grid)
        + len(specifications) * config.n_paths * config.n_steps
    )
    return runs, path_steps


def studies_to_csv(studies: NumericalStudiesResult) -> str:
    """Return run-level CSV with reproducibility and model-input metadata."""
    output = io.StringIO()
    fieldnames = [
        "study_type",
        "setting_value",
        "basis_terms",
        "base_seed",
        "replication_seed",
        "estimated_value",
        "reported_standard_error",
        "spot",
        "strike",
        "maturity_years",
        "volatility",
        "risk_free_rate",
        "asset_model",
        "exercise_style",
        "path_count",
        "time_step_count",
        "heston_mean_reversion_speed",
        "heston_long_run_variance",
        "heston_volatility_of_variance",
        "heston_correlation",
        "heston_initial_variance",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    study_items = [
        ("path_count", studies.path_count),
        ("basis_sensitivity", studies.basis_sensitivity),
    ]
    if studies.exercise_grid is not None:
        study_items.insert(1, ("exercise_grid", studies.exercise_grid))
    for study_type, study in study_items:
        for point in study.points:
            for run in point.runs:
                path_count = (
                    int(point.setting_value)
                    if study_type == "path_count"
                    else study.path_count
                )
                time_steps = (
                    int(point.setting_value)
                    if study_type == "exercise_grid"
                    else study.time_steps
                )
                writer.writerow(
                    {
                        "study_type": study_type,
                        "setting_value": point.setting_value,
                        "basis_terms": ", ".join(point.basis_terms),
                        "base_seed": study.base_seed,
                        "replication_seed": run.seed,
                        "estimated_value": run.estimated_price,
                        "reported_standard_error": run.standard_error,
                        "spot": study.market.spot,
                        "strike": study.contract.strike,
                        "maturity_years": study.contract.maturity,
                        "volatility": study.market.volatility,
                        "risk_free_rate": study.market.risk_free_rate,
                        "asset_model": study.model,
                        "exercise_style": ExerciseStyle.AMERICAN.value,
                        "path_count": path_count,
                        "time_step_count": time_steps,
                        "heston_mean_reversion_speed": (
                            study.heston_inputs.mean_reversion_speed
                            if study.heston_inputs is not None
                            else ""
                        ),
                        "heston_long_run_variance": (
                            study.heston_inputs.long_run_variance
                            if study.heston_inputs is not None
                            else ""
                        ),
                        "heston_volatility_of_variance": (
                            study.heston_inputs.volatility_of_variance
                            if study.heston_inputs is not None
                            else ""
                        ),
                        "heston_correlation": (
                            study.heston_inputs.correlation
                            if study.heston_inputs is not None
                            else ""
                        ),
                        "heston_initial_variance": (
                            study.heston_inputs.initial_variance
                            if study.heston_inputs is not None
                            else ""
                        ),
                    }
                )
    return output.getvalue()
