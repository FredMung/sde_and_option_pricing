import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from option_pricing_app.domain import MarketInputs, PutContract, SimulationConfig
from option_pricing_app.numerical_studies import (
    NumericalStudiesResult,
    basis_specifications,
    make_exercise_grid,
    make_path_count_grid,
    run_basis_sensitivity_study,
    run_exercise_grid_study,
    run_path_count_study,
    studies_to_csv,
)
from option_pricing_app.stats import GBMPriceSimulator, LSMCPriceSimulator

CONTRACT = PutContract(100, 1)
MARKET = MarketInputs(100, 0.2, 0.05)


def _fake_functions():
    """Fakes for the fit/generate/evaluate seams that avoid real Monte Carlo.

    ``generate_paths`` smuggles the validation ``SimulationConfig`` through as the
    "paths" return value, which ``evaluate`` reads back -- this keeps orchestration
    tests fast while still letting them assert on exactly which config each seam saw.
    """
    fit_calls = []
    generate_calls = []
    evaluate_calls = []

    def fit_policy(contract, market, config, model, heston_inputs):
        fit_calls.append((contract, market, config, model, heston_inputs))
        return SimpleNamespace(training_config=config)

    def generate_paths(contract, market, config, model, heston_inputs):
        generate_calls.append((contract, market, config, model, heston_inputs))
        return config, None

    def evaluate(policy, paths, variance_paths):
        config = paths
        evaluate_calls.append((policy, config))
        estimate = config.n_paths / 1_000 + config.n_steps / 100 + config.seed % 17 / 100
        return SimpleNamespace(mean_estimate=estimate, standard_error=0.123)

    return fit_policy, generate_paths, evaluate, fit_calls, generate_calls, evaluate_calls


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (100, (100,)),
        (750, (100, 250, 500, 750)),
        (1_000, (100, 250, 500, 1_000)),
        (1_200, (100, 250, 500, 1_000, 1_200)),
    ],
)
def test_dynamic_path_count_grid_includes_exact_maximum(maximum, expected):
    assert make_path_count_grid(maximum) == expected


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (1, (1,)),
        (40, (5, 10, 25, 40)),
        (100, (5, 10, 25, 50, 100)),
        (120, (5, 10, 25, 50, 100, 120)),
    ],
)
def test_dynamic_exercise_grid_includes_exact_maximum(maximum, expected):
    assert make_exercise_grid(maximum) == expected


def test_path_count_study_trains_on_the_setting_and_validates_on_a_fixed_set():
    fit_policy, generate_paths, evaluate, fit_calls, generate_calls, _ = _fake_functions()
    config = SimulationConfig(500, 10, 42, 1, ("1", "S"))
    study = run_path_count_study(
        CONTRACT, MARKET, config, "GBM", None, 73, 3, fit_policy, generate_paths, evaluate
    )
    grid = make_path_count_grid(500)
    assert len(fit_calls) == len(grid) * 3
    assert len(generate_calls) == len(grid) * 3
    # Training path count is the setting under test...
    assert {call[2].n_paths for call in fit_calls} == set(grid)
    # ...but the independent validation set is always the currently selected count,
    # so validation noise cannot be mistaken for a training-size effect.
    assert {call[2].n_paths for call in generate_calls} == {500}
    for path_count in grid:
        assert [call[2].seed for call in fit_calls if call[2].n_paths == path_count] == [
            pair[0] for pair in study.derived_seeds
        ]
    assert all(call[2].basis_terms == ("1", "S") for call in fit_calls)
    assert all(call[2].basis_terms == ("1", "S") for call in generate_calls)


def test_different_path_counts_fit_different_lsmc_policies():
    np.random.seed(123)
    paths = GBMPriceSimulator().generate_price_paths(100, 1, 0.05, 0.2, 250, 10)
    small = LSMCPriceSimulator()
    large = LSMCPriceSimulator()
    small.backward_induction(paths[:, :100], 100, 0.05, 0.1)
    large.backward_induction(paths, 100, 0.05, 0.1)
    shared_step = min(set(small.continuation_models) & set(large.continuation_models))
    assert not np.allclose(
        small.continuation_models[shared_step].coefficients,
        large.continuation_models[shared_step].coefficients,
    )


def test_repeated_estimates_use_sample_std_and_empirical_quantiles():
    fit_policy, generate_paths, evaluate, *_ = _fake_functions()
    study = run_path_count_study(
        CONTRACT,
        MARKET,
        SimulationConfig(100, 10, 42, 1),
        "GBM",
        None,
        99,
        5,
        fit_policy,
        generate_paths,
        evaluate,
    )
    point = study.points[0]
    estimates = np.array([run.estimated_price for run in point.runs])
    assert point.mean_estimate == pytest.approx(np.mean(estimates))
    assert point.empirical_standard_deviation == pytest.approx(np.std(estimates, ddof=1))
    assert point.lower_empirical_quantile == pytest.approx(np.quantile(estimates, 0.025))
    assert point.upper_empirical_quantile == pytest.approx(np.quantile(estimates, 0.975))


def test_exercise_grid_orchestration_holds_paths_and_basis_fixed():
    fit_policy, generate_paths, evaluate, fit_calls, generate_calls, _ = _fake_functions()
    config = SimulationConfig(100, 25, 42, 1, ("1", "S"))
    study = run_exercise_grid_study(
        CONTRACT, MARKET, config, "GBM", None, 42, 3, fit_policy, generate_paths, evaluate
    )
    grid = make_exercise_grid(25)
    # Both sides share the grid's own step count -- a policy can only be evaluated
    # against paths sharing its exercise dates.
    assert {call[2].n_steps for call in fit_calls} == set(grid)
    assert {call[2].n_steps for call in generate_calls} == set(grid)
    assert all(call[2].n_paths == 100 for call in fit_calls)
    assert all(call[2].n_paths == 100 for call in generate_calls)
    assert all(call[2].basis_terms == ("1", "S") for call in fit_calls)
    assert len(fit_calls) == len(study.points) * 3
    assert len(generate_calls) == len(study.points) * 3


def test_basis_orchestration_holds_paths_and_steps_fixed_and_includes_selection():
    fit_policy, generate_paths, evaluate, fit_calls, generate_calls, _ = _fake_functions()
    selected = ("1", "S²")
    config = SimulationConfig(100, 10, 42, 1, selected)
    study = run_basis_sensitivity_study(
        CONTRACT, MARKET, config, "GBM", None, 42, 3, fit_policy, generate_paths, evaluate
    )
    assert selected in basis_specifications("GBM", selected)
    assert {call[2].basis_terms for call in fit_calls} == set(basis_specifications("GBM", selected))
    assert {call[2].basis_terms for call in generate_calls} == set(
        basis_specifications("GBM", selected)
    )
    assert all(call[2].n_paths == 100 and call[2].n_steps == 10 for call in fit_calls)
    assert all(call[2].n_paths == 100 and call[2].n_steps == 10 for call in generate_calls)
    assert len(fit_calls) == len(study.points) * 3


def test_heston_uses_documented_nested_basis_set_and_disables_grid_study():
    specifications = basis_specifications("Heston", ("1", "S", "v"))
    assert specifications[0] == ("1", "S")
    assert ("1", "S", "S²", "v", "S·v") in specifications
    assert len(specifications) <= 6
    with pytest.raises(ValueError, match="limited to GBM"):
        run_exercise_grid_study(
            CONTRACT,
            MARKET,
            SimulationConfig(100, 10, 42, 1),
            "Heston",
            None,
            42,
            3,
        )


def test_csv_contains_run_level_reproducibility_and_input_metadata():
    fit_policy, generate_paths, evaluate, *_ = _fake_functions()
    config = SimulationConfig(100, 5, 42, 1)
    path = run_path_count_study(
        CONTRACT, MARKET, config, "GBM", None, 42, 3, fit_policy, generate_paths, evaluate
    )
    exercise = run_exercise_grid_study(
        CONTRACT, MARKET, config, "GBM", None, 42, 3, fit_policy, generate_paths, evaluate
    )
    basis = run_basis_sensitivity_study(
        CONTRACT, MARKET, config, "GBM", None, 42, 3, fit_policy, generate_paths, evaluate
    )
    csv_text = studies_to_csv(NumericalStudiesResult(path, exercise, basis))
    header, first_row, *_ = csv_text.splitlines()
    assert "study_type" in header
    assert "training_seed" in header
    assert "valuation_seed" in header
    assert "reported_standard_error" in header
    assert "path_count" in header
    assert "time_step_count" in header
    assert "path_count" in first_row
    assert str(path.base_seed) in first_row


def test_numerical_experiment_module_has_no_streamlit_or_plotly_imports():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "option_pricing_app"
        / "numerical_studies.py"
    ).read_text()
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "streamlit" not in imports
    assert "plotly" not in imports
