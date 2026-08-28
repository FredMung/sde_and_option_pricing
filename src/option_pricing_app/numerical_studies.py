"""Repeated-run numerical studies built on the existing pricing service.

This module deliberately contains no Streamlit or Plotly imports. Each replication
fits the LSMC policy on a freshly simulated *training* path set, then evaluates that
frozen policy -- without refitting -- on an independently simulated *validation* path
set (see ``lsmc_policy.py``). Reporting the out-of-sample estimate rather than the
same-sample one avoids mistaking in-sample fit for genuine sensitivity: a same-sample
estimate can look better purely because the stopping rule was chosen to fit that exact
data. Full path arrays are discarded immediately after each fit/evaluation.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from option_pricing_app.domain import (
    HESTON_BASIS_TERMS,
    ExerciseStyle,
    HestonInputs,
    MarketInputs,
    PutContract,
    SimulationConfig,
)
from option_pricing_app.lsmc_policy import (
    evaluate_independent_policy,
    fit_lsmc_policy,
    generate_paths,
)
from option_pricing_app.policy_validation import derive_paired_seeds
from option_pricing_app.service import crr_american_put_price

PATH_COUNT_CANDIDATES = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000)
EXERCISE_GRID_CANDIDATES = (5, 10, 25, 50, 100, 250, 500, 1_000)


@dataclass(frozen=True)
class ExperimentRun:
    training_seed: int
    valuation_seed: int
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
    bias_vs_binomial: float | None
    rmse_vs_binomial: float | None


@dataclass(frozen=True)
class PathCountStudyResult:
    base_seed: int
    derived_seeds: tuple[tuple[int, int], ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    time_steps: int
    basis_terms: tuple[str, ...]
    heston_inputs: HestonInputs | None
    binomial_reference: float | None


@dataclass(frozen=True)
class ExerciseGridStudyResult:
    base_seed: int
    derived_seeds: tuple[tuple[int, int], ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    path_count: int
    basis_terms: tuple[str, ...]
    heston_inputs: HestonInputs | None
    binomial_reference: float | None


@dataclass(frozen=True)
class BasisSensitivityStudyResult:
    base_seed: int
    derived_seeds: tuple[tuple[int, int], ...]
    points: tuple[ExperimentPoint, ...]
    contract: PutContract
    market: MarketInputs
    model: str
    path_count: int
    time_steps: int
    heston_inputs: HestonInputs | None
    binomial_reference: float | None


@dataclass(frozen=True)
class NumericalStudiesResult:
    path_count: PathCountStudyResult
    exercise_grid: ExerciseGridStudyResult | None
    basis_sensitivity: BasisSensitivityStudyResult


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


def generate_base_seed() -> int:
    """Generate and return an explicit seed for an initially unseeded study."""
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def derive_validation_seed(base_seed: int) -> int:
    """Derive one deterministic child seed for a single out-of-sample evaluation."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    child = np.random.SeedSequence(base_seed).spawn(1)[0]
    return int(child.generate_state(1, dtype=np.uint32)[0])


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
    binomial_reference: float | None,
) -> ExperimentPoint:
    estimates = np.asarray([run.estimated_price for run in runs], dtype=float)
    bias_vs_binomial = None
    rmse_vs_binomial = None
    if binomial_reference is not None:
        errors = estimates - binomial_reference
        bias_vs_binomial = float(np.mean(errors))
        rmse_vs_binomial = float(np.sqrt(np.mean(errors**2)))
    return ExperimentPoint(
        setting_value=setting_value,
        setting_label=setting_label,
        basis_terms=basis_terms,
        runs=tuple(runs),
        mean_estimate=float(np.mean(estimates)),
        empirical_standard_deviation=float(np.std(estimates, ddof=1)),
        lower_empirical_quantile=float(np.quantile(estimates, 0.025)),
        upper_empirical_quantile=float(np.quantile(estimates, 0.975)),
        bias_vs_binomial=bias_vs_binomial,
        rmse_vs_binomial=rmse_vs_binomial,
    )


def _run_point(
    *,
    setting_value: int | str,
    setting_label: str,
    contract: PutContract,
    market: MarketInputs,
    model: str,
    heston_inputs: HestonInputs | None,
    training_path_count: int,
    validation_path_count: int,
    time_steps: int,
    basis_terms: tuple[str, ...],
    seed_pairs: tuple[tuple[int, int], ...],
    fit_policy_function,
    generate_paths_function,
    evaluate_function,
    binomial_reference: float | None,
) -> ExperimentPoint:
    runs = []
    for training_seed, valuation_seed in seed_pairs:
        # Fit the LSMC stopping policy on the training paths, then evaluate that
        # frozen policy -- never refit -- on an independently simulated validation
        # set. The out-of-sample estimate is what makes this a genuine sensitivity
        # study rather than an artifact of same-sample in-sample fit.
        training_config = SimulationConfig(
            training_path_count, time_steps, training_seed, 1, basis_terms
        )
        policy = fit_policy_function(contract, market, training_config, model, heston_inputs)
        validation_config = SimulationConfig(
            validation_path_count, time_steps, valuation_seed, 1, basis_terms
        )
        validation_paths, validation_variance_paths = generate_paths_function(
            contract, market, validation_config, model, heston_inputs
        )
        evaluation = evaluate_function(policy, validation_paths, validation_variance_paths)
        runs.append(
            ExperimentRun(
                training_seed, valuation_seed, evaluation.mean_estimate, evaluation.standard_error
            )
        )
    return _aggregate_point(setting_value, setting_label, basis_terms, runs, binomial_reference)


def run_path_count_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    fit_policy_function=fit_lsmc_policy,
    generate_paths_function=generate_paths,
    evaluate_function=evaluate_independent_policy,
) -> PathCountStudyResult:
    # Training path count is the setting under test; the validation set stays fixed
    # at the currently selected path count so validation noise does not confound
    # with the training-size effect being measured.
    seed_pairs = derive_paired_seeds(base_seed, replications)
    binomial_reference = crr_american_put_price(contract, market) if model == "GBM" else None
    points = tuple(
        _run_point(
            setting_value=path_count,
            setting_label=f"{path_count:,}",
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            training_path_count=path_count,
            validation_path_count=config.n_paths,
            time_steps=config.n_steps,
            basis_terms=config.basis_terms,
            seed_pairs=seed_pairs,
            fit_policy_function=fit_policy_function,
            generate_paths_function=generate_paths_function,
            evaluate_function=evaluate_function,
            binomial_reference=binomial_reference,
        )
        for path_count in make_path_count_grid(config.n_paths)
    )
    return PathCountStudyResult(
        base_seed,
        seed_pairs,
        points,
        contract,
        market,
        model,
        config.n_steps,
        config.basis_terms,
        heston_inputs,
        binomial_reference,
    )


def run_exercise_grid_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    fit_policy_function=fit_lsmc_policy,
    generate_paths_function=generate_paths,
    evaluate_function=evaluate_independent_policy,
) -> ExerciseGridStudyResult:
    if model != "GBM":
        raise ValueError("The exercise-grid study is currently limited to GBM")
    # Both sides use the grid's own step count -- a policy can only be evaluated on
    # paths sharing its exercise dates -- with path count pinned at the current
    # setting so only the exercise-grid effect varies.
    seed_pairs = derive_paired_seeds(base_seed, replications)
    binomial_reference = crr_american_put_price(contract, market)
    points = tuple(
        _run_point(
            setting_value=time_steps,
            setting_label=f"{time_steps:,}",
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            training_path_count=config.n_paths,
            validation_path_count=config.n_paths,
            time_steps=time_steps,
            basis_terms=config.basis_terms,
            seed_pairs=seed_pairs,
            fit_policy_function=fit_policy_function,
            generate_paths_function=generate_paths_function,
            evaluate_function=evaluate_function,
            binomial_reference=binomial_reference,
        )
        for time_steps in make_exercise_grid(config.n_steps)
    )
    return ExerciseGridStudyResult(
        base_seed,
        seed_pairs,
        points,
        contract,
        market,
        model,
        config.n_paths,
        config.basis_terms,
        heston_inputs,
        binomial_reference,
    )


def run_basis_sensitivity_study(
    contract: PutContract,
    market: MarketInputs,
    config: SimulationConfig,
    model: str,
    heston_inputs: HestonInputs | None,
    base_seed: int,
    replications: int,
    fit_policy_function=fit_lsmc_policy,
    generate_paths_function=generate_paths,
    evaluate_function=evaluate_independent_policy,
) -> BasisSensitivityStudyResult:
    # Path count and step count are pinned at the current setting; only the basis
    # specification varies, on both the training and validation side.
    seed_pairs = derive_paired_seeds(base_seed, replications)
    specifications = basis_specifications(model, config.basis_terms)
    binomial_reference = crr_american_put_price(contract, market) if model == "GBM" else None
    points = tuple(
        _run_point(
            setting_value=" + ".join(terms),
            setting_label=" + ".join(terms),
            contract=contract,
            market=market,
            model=model,
            heston_inputs=heston_inputs,
            training_path_count=config.n_paths,
            validation_path_count=config.n_paths,
            time_steps=config.n_steps,
            basis_terms=terms,
            seed_pairs=seed_pairs,
            fit_policy_function=fit_policy_function,
            generate_paths_function=generate_paths_function,
            evaluate_function=evaluate_function,
            binomial_reference=binomial_reference,
        )
        for terms in specifications
    )
    return BasisSensitivityStudyResult(
        base_seed,
        seed_pairs,
        points,
        contract,
        market,
        model,
        config.n_paths,
        config.n_steps,
        heston_inputs,
        binomial_reference,
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


CRR_CONVERGENCE_STEPS = (100, 250, 500, 1_000, 2_000, 4_000, 8_000)


@dataclass(frozen=True)
class CRRConvergencePoint:
    steps: int
    price: float
    change_from_previous: float | None


def crr_convergence_table(
    contract: PutContract,
    market: MarketInputs,
    steps_grid: tuple[int, ...] = CRR_CONVERGENCE_STEPS,
) -> tuple[CRRConvergencePoint, ...]:
    """Show the CRR binomial tree's own price stabilising as step count increases.

    This validates the choice of reference step count used for the bias and RMSE
    comparisons elsewhere: it is a property of the binomial tree itself, independent
    of any LSMC simulation or replication.
    """
    points = []
    previous_price = None
    for steps in steps_grid:
        price = crr_american_put_price(contract, market, steps)
        change_from_previous = None if previous_price is None else price - previous_price
        points.append(CRRConvergencePoint(steps, price, change_from_previous))
        previous_price = price
    return tuple(points)


def estimate_workload(config: SimulationConfig, model: str, replications: int) -> tuple[int, int]:
    """Return (pricing runs, simulated path-steps) without executing a study.

    Each replication now runs two simulations -- a training fit and an independent
    out-of-sample evaluation -- instead of one same-sample pricing run, roughly
    doubling the workload of the previous same-sample design.
    """
    path_grid = make_path_count_grid(config.n_paths)
    exercise_grid = make_exercise_grid(config.n_steps) if model == "GBM" else ()
    specifications = basis_specifications(model, config.basis_terms)
    runs = 2 * replications * (len(path_grid) + len(exercise_grid) + len(specifications))
    path_steps = replications * (
        config.n_steps * (sum(path_grid) + len(path_grid) * config.n_paths)
        + 2 * config.n_paths * sum(exercise_grid)
        + 2 * len(specifications) * config.n_paths * config.n_steps
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
        "training_seed",
        "valuation_seed",
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
        "binomial_reference_price",
        "error_vs_binomial_reference",
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
                        "training_seed": run.training_seed,
                        "valuation_seed": run.valuation_seed,
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
                        "binomial_reference_price": (
                            study.binomial_reference
                            if study.binomial_reference is not None
                            else ""
                        ),
                        "error_vs_binomial_reference": (
                            run.estimated_price - study.binomial_reference
                            if study.binomial_reference is not None
                            else ""
                        ),
                    }
                )
    return output.getvalue()
