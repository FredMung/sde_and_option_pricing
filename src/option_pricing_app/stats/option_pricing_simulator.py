"""Option-pricing simulators and configurable LSMC regression machinery."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from option_pricing_app.domain import PRICE_BASIS_TERMS
from option_pricing_app.stats.asset_pricing_simulator import GBMPriceSimulator


BASIS_TERM_EXPONENTS = {
    "1": (0, 0),
    "S": (1, 0),
    "S²": (2, 0),
    "v": (0, 1),
    "v²": (0, 2),
    "S·v": (1, 1),
    "S²·v": (2, 1),
    "S·v²": (1, 2),
    "S²·v²": (2, 2),
}


@dataclass(frozen=True)
class ContinuationRegression:
    """Scaled least-squares continuation model fitted at one exercise timestep."""

    timestep: int
    basis_terms: tuple[str, ...]
    coefficients: np.ndarray
    column_scales: np.ndarray

    @property
    def uses_variance(self):
        """Return whether prediction requires the Heston variance state."""
        return any(BASIS_TERM_EXPONENTS[term][1] for term in self.basis_terms)

    def predict(self, prices, variances=None):
        """Evaluate the fitted cross-sectional regression without refitting it."""
        design_matrix = LSMCPriceSimulator.build_design_matrix(
            prices, self.basis_terms, variances
        )
        return (design_matrix / self.column_scales) @ self.coefficients


class LSMCPriceSimulator:
    def __init__(self, price_path_simulator=None):
        """Initialise LSMC with a configurable price-path simulator."""
        # Use composition so that GBM can later be replaced by another path model.
        self.price_path_simulator = price_path_simulator or GBMPriceSimulator()
        self.option_value_paths = None
        self.exercise_steps = None
        self.continuation_models = {}

    def generate_price_paths(self, *args, **kwargs):
        """Generate paths on the exercise-time grid used by LSMC."""
        return self.price_path_simulator.generate_price_paths(*args, **kwargs)

    def plot_price_paths(self, price_paths, max_paths=50):
        """Plot simulated underlying-price paths."""
        self.price_path_simulator.plot_price_paths(price_paths, max_paths)

    def plot_option_value_paths(self, max_paths=50):
        """Plot the pathwise values produced during backward induction."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.option_value_paths[:, :max_paths])
        plt.xlabel("Time")
        plt.ylabel("Discounted cash-flow value")
        plt.title("LSMC Pathwise Option Values")
        plt.grid(True)
        plt.show()
    
    def calculate_payoff(self, S, K):
        """Calculate the immediate payoff of an American put."""
        return np.maximum(K - S, 0)
    
    def estimate_continuation_value(
        self,
        prices,
        future_cash_flows,
        r,
        dt,
        basis_terms=None,
        variances=None,
    ):
        """Estimate continuation values from discounted future cash flows."""
        model = self.fit_continuation_model(
            prices,
            future_cash_flows,
            r,
            dt,
            basis_terms,
            variances,
            timestep=-1,
        )
        return model.predict(prices, variances)

    def fit_continuation_model(
        self,
        prices,
        future_cash_flows,
        r,
        dt,
        basis_terms=None,
        variances=None,
        timestep=-1,
    ):
        """Fit and return the scaled regression used by the stopping decision."""
        terms = tuple(basis_terms or PRICE_BASIS_TERMS)
        design_matrix = self.build_design_matrix(prices, terms, variances)
        # Column scaling leaves the polynomial span unchanged and improves the
        # conditioning of mixed price-variance regressions.
        column_scales = np.linalg.norm(design_matrix, axis=0)
        column_scales[column_scales == 0] = 1.0
        scaled_design = design_matrix / column_scales
        regression_target = np.exp(-r * dt) * future_cash_flows
        coefficients = np.linalg.lstsq(scaled_design, regression_target, rcond=None)[0]
        return ContinuationRegression(
            timestep=timestep,
            basis_terms=terms,
            coefficients=coefficients.copy(),
            column_scales=column_scales.copy(),
        )

    @staticmethod
    def build_design_matrix(prices, basis_terms, variances=None):
        """Construct selected price, variance and cross-term regression columns."""
        columns = []
        for term in basis_terms:
            try:
                price_power, variance_power = BASIS_TERM_EXPONENTS[term]
            except KeyError as exc:
                raise ValueError(f"Unsupported LSMC basis term: {term}") from exc
            if variance_power and variances is None:
                raise ValueError(f"LSMC basis term {term} requires Heston variance paths")
            column = np.power(prices, price_power)
            if variance_power:
                column = column * np.power(variances, variance_power)
            columns.append(column)
        if not columns:
            raise ValueError("At least one LSMC basis term is required")
        return np.column_stack(columns)

    def backward_induction(
        self,
        price_paths,
        K,
        r,
        dt,
        basis_terms=None,
        variance_paths=None,
    ):
        """Apply backward induction across the simulated exercise dates."""
        n_time_points = price_paths.shape[0]
        n_paths = price_paths.shape[1]
        discount_factor = np.exp(-r * dt)
        if variance_paths is not None and variance_paths.shape != price_paths.shape:
            raise ValueError("variance_paths must have the same shape as price_paths")
        self.continuation_models = {}
        
        # Calculate max(K-S_T, 0) for each path at maturity.
        cash_flows = self.calculate_payoff(price_paths[-1], K)
        # -1 denotes expiry with no payoff. A maturity payoff is exercise at the
        # final step unless an earlier stopping decision subsequently replaces it.
        exercise_steps = np.where(cash_flows > 0, n_time_points - 1, -1)
        option_value_paths = np.empty_like(price_paths)
        option_value_paths[-1, :] = cash_flows
        
        # Work backwards from T-1 to 1; the time-zero decision is handled below.
        for t in range(n_time_points - 2, 0, -1):

            prices_at_t = price_paths[t, :]

            # Only select in-the-money paths
            in_the_money = prices_at_t < K
            
            # Calculate immediate exercise value
            immediate_exercise_value = self.calculate_payoff(prices_at_t, K)

            exercise_decision = np.zeros(n_paths, dtype=bool)
            if np.any(in_the_money):
                # Estimate continuation value using regression.
                # Out-of-the-money paths are excluded from the regression.
                continuation_model = self.fit_continuation_model(
                    price_paths[t, in_the_money],
                    cash_flows[in_the_money],
                    r,
                    dt,
                    basis_terms,
                    None if variance_paths is None else variance_paths[t, in_the_money],
                    timestep=t,
                )
                self.continuation_models[t] = continuation_model
                continuation_value = continuation_model.predict(
                    price_paths[t, in_the_money],
                    None if variance_paths is None else variance_paths[t, in_the_money],
                )

                # Determine the exercise decision for in-the-money paths.
                exercise_decision[in_the_money] = (
                    immediate_exercise_value[in_the_money] > continuation_value
                )

            # Discount once, then replace paths whose policy exercises at time t.
            cash_flows = discount_factor * cash_flows
            cash_flows[exercise_decision] = immediate_exercise_value[exercise_decision]
            exercise_steps[exercise_decision] = t
            option_value_paths[t, :] = cash_flows
        
        # Move the time-1 cash flows to time zero.
        discounted_cash_flows = discount_factor * cash_flows
        # Take the average as the option price
        continuation_value_at_zero = np.mean(discounted_cash_flows)

        # Exercise immediately if in-the-money already at time 0
        immediate_exercise_at_zero = self.calculate_payoff(price_paths[0, 0], K)
        if (
            immediate_exercise_at_zero > 0
            and immediate_exercise_at_zero >= continuation_value_at_zero
        ):
            option_value_paths[0, :] = immediate_exercise_at_zero
            self.option_value_paths = option_value_paths
            self.exercise_steps = np.zeros(n_paths, dtype=int)
            return float(immediate_exercise_at_zero), 0.0

        option_value_paths[0, :] = discounted_cash_flows
        self.option_value_paths = option_value_paths
        self.exercise_steps = exercise_steps
        standard_error = np.std(discounted_cash_flows) / np.sqrt(n_paths)
        return float(continuation_value_at_zero), float(standard_error)
    
    def calculate_lsmc_price(
        self, S0, T, r, sigma, K, n_paths, n_steps, basis_terms=None
    ):
        """Estimate an American put price and its Monte Carlo standard error."""
        dt = T / n_steps
        price_paths = self.generate_price_paths(S0, T, r, sigma, n_paths, n_steps)
        lsmc_price, std_error = self.backward_induction(price_paths, K, r, dt, basis_terms)
        return lsmc_price, std_error
