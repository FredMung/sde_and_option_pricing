from types import SimpleNamespace

import pytest

from option_pricing_app.domain import MarketInputs, PutContract, SimulationConfig
from option_pricing_app.policy_validation import (
    PAIRED_VALIDATION_BASE_SEED,
    derive_paired_seeds,
    matched_grid_crr_price,
    paired_validation_to_csv,
    run_paired_validation_study,
)
from option_pricing_app.service import CRR_BINOMIAL_STEPS, crr_american_put_price

CONTRACT = PutContract(100, 1)
MARKET = MarketInputs(100, 0.2, 0.05)


def test_paired_seeds_are_deterministic_and_training_valuation_seeds_are_distinct():
    first = derive_paired_seeds(PAIRED_VALIDATION_BASE_SEED, 5)
    second = derive_paired_seeds(PAIRED_VALIDATION_BASE_SEED, 5)
    assert first == second
    all_seeds = [seed for pair in first for seed in pair]
    assert len(set(all_seeds)) == len(all_seeds) == 10
    for training_seed, valuation_seed in first:
        assert training_seed != valuation_seed


def test_paired_seeds_reject_out_of_range_replications():
    with pytest.raises(ValueError, match="replications must be"):
        derive_paired_seeds(PAIRED_VALIDATION_BASE_SEED, 2)
    with pytest.raises(ValueError, match="replications must be"):
        derive_paired_seeds(PAIRED_VALIDATION_BASE_SEED, 21)


def test_matched_grid_crr_equals_ordinary_crr_at_full_binomial_steps():
    matched = matched_grid_crr_price(CONTRACT, MARKET, CRR_BINOMIAL_STEPS)
    ordinary = crr_american_put_price(CONTRACT, MARKET)
    assert matched == ordinary


def test_matched_grid_crr_differs_from_the_continuous_reference_at_a_coarse_grid():
    coarse = matched_grid_crr_price(CONTRACT, MARKET, 10)
    continuous = crr_american_put_price(CONTRACT, MARKET)
    assert coarse != continuous


def _fake_policy_and_evaluation_functions():
    """Deterministic fakes for the fit/generate/evaluate seams.

    Mirrors the pattern in ``test_numerical_studies.py``: ``generate_paths`` smuggles
    the ``SimulationConfig`` through as "paths" so ``evaluate`` can derive a
    reproducible estimate from it without running real Monte Carlo.
    """
    fit_calls = []
    generate_calls = []

    def fit_policy(contract, market, config, model, heston_inputs):
        fit_calls.append(config)
        estimate = config.seed / 1_000
        return SimpleNamespace(same_sample_price=estimate, same_sample_standard_error=0.1)

    def generate_paths(contract, market, config, model, heston_inputs):
        generate_calls.append(config)
        return config, None

    def evaluate(policy, paths, variance_paths):
        estimate = policy.same_sample_price - 0.05
        return SimpleNamespace(mean_estimate=estimate, standard_error=0.2)

    return fit_policy, generate_paths, evaluate, fit_calls, generate_calls


def test_run_paired_validation_study_threads_distinct_seeds_per_replication(monkeypatch):
    fit_policy, generate_paths, evaluate, fit_calls, generate_calls = (
        _fake_policy_and_evaluation_functions()
    )
    monkeypatch.setattr("option_pricing_app.policy_validation.fit_lsmc_policy", fit_policy)
    monkeypatch.setattr("option_pricing_app.policy_validation.generate_paths", generate_paths)
    monkeypatch.setattr(
        "option_pricing_app.policy_validation.evaluate_independent_policy", evaluate
    )

    config = SimulationConfig(200, 10, 1, 1)
    study = run_paired_validation_study(CONTRACT, MARKET, config, base_seed=123, replications=4)

    assert len(study.replications) == 4
    training_seeds = [call.seed for call in fit_calls]
    valuation_seeds = [call.seed for call in generate_calls]
    assert len(set(training_seeds)) == 4
    assert len(set(valuation_seeds)) == 4
    assert set(training_seeds).isdisjoint(valuation_seeds)
    for row, training_seed, valuation_seed in zip(
        study.replications, training_seeds, valuation_seeds, strict=True
    ):
        assert row.training_seed == training_seed
        assert row.valuation_seed == valuation_seed
        # Same-sample minus independent should equal the paired difference exactly.
        assert row.paired_difference == pytest.approx(
            row.same_sample_estimate - row.independent_estimate
        )


def test_paired_validation_summary_aggregates_match_manual_computation(monkeypatch):
    fit_policy, generate_paths, evaluate, *_ = _fake_policy_and_evaluation_functions()
    monkeypatch.setattr("option_pricing_app.policy_validation.fit_lsmc_policy", fit_policy)
    monkeypatch.setattr("option_pricing_app.policy_validation.generate_paths", generate_paths)
    monkeypatch.setattr(
        "option_pricing_app.policy_validation.evaluate_independent_policy", evaluate
    )

    config = SimulationConfig(200, 10, 1, 1)
    study = run_paired_validation_study(CONTRACT, MARKET, config, base_seed=123, replications=3)

    same_sample = [row.same_sample_estimate for row in study.replications]
    independent = [row.independent_estimate for row in study.replications]
    assert study.mean_same_sample_estimate == pytest.approx(sum(same_sample) / 3)
    assert study.mean_independent_estimate == pytest.approx(sum(independent) / 3)
    assert study.mean_paired_difference == pytest.approx(0.05)
    assert study.matched_grid_crr == matched_grid_crr_price(CONTRACT, MARKET, config.n_steps)
    assert study.continuous_crr == crr_american_put_price(CONTRACT, MARKET)


def test_paired_validation_csv_contains_all_replications_and_both_seeds():
    config = SimulationConfig(150, 8, 5, 1)
    study = run_paired_validation_study(CONTRACT, MARKET, config, base_seed=123, replications=3)
    csv_text = paired_validation_to_csv(study)
    lines = csv_text.strip().splitlines()
    header = lines[0]
    assert "training_seed" in header
    assert "valuation_seed" in header
    assert len(lines) == 1 + len(study.replications) == 4
    for row, line in zip(study.replications, lines[1:], strict=True):
        assert str(row.training_seed) in line
        assert str(row.valuation_seed) in line
