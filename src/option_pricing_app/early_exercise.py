"""Run-level early-exercise policy summary for the currently generated simulation.

This module deliberately contains no Streamlit or Plotly imports, mirroring the
separation between computation and presentation used in ``numerical_studies``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from option_pricing_app.domain import MarketInputs, PricingResult, PutContract
from option_pricing_app.lsmc_policy import IndependentPolicyEvaluation
from option_pricing_app.service import black_scholes_put_price, crr_american_put_price


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
    independent_evaluation: IndependentPolicyEvaluation | None,
) -> tuple[EarlyExercisePolicyRow, ...]:
    """Longstaff & Schwartz (2001), Table 1-style summary for the current run.

    Reports the American value, European value and their difference (the early
    exercise value) from an independent reference -- a CRR binomial tree plus the
    closed-form European price, standing in for the paper's finite-difference
    benchmark -- and from the fitted LSMC policy. The reference row is only included
    under GBM, since CRR and the closed-form price both assume constant-volatility
    lognormal dynamics.

    The LSMC row is the frozen policy's *out-of-sample* value: ``independent_
    evaluation`` is the training run's fitted policy evaluated, without refitting,
    on an independently simulated validation path set -- computed once by the
    caller and shared with the headline metrics, so both parts of the dashboard
    report the same number. This measures what the fitted policy is actually
    worth on new paths, rather than the same-sample training estimate, which is
    biased high because the stopping rule was chosen to fit that exact data. The
    European value has no fitted stopping decision, so it carries no such bias and
    is left as the training run's same-sample Monte Carlo estimate.
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
    if independent_evaluation is not None:
        rows.append(
            EarlyExercisePolicyRow(
                method="LSMC (out-of-sample policy value)",
                american_value=independent_evaluation.mean_estimate,
                american_standard_error=independent_evaluation.standard_error,
                european_value=result.european_mc_price,
                early_exercise_value=(
                    independent_evaluation.mean_estimate - result.european_mc_price
                ),
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
