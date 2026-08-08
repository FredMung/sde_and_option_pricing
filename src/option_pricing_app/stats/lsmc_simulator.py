"""LSMC simulator migrated from the dissertation without algorithm changes."""

import numpy as np
import matplotlib.pyplot as plt

from option_pricing_app.stats.gbm_simulator import GBMPriceSimulator

class LSMCPriceSimulator:
    def __init__(self, price_path_simulator=None):
        """Initialise LSMC with a configurable price-path simulator."""
        # Use composition so that GBM can later be replaced by another path model.
        self.price_path_simulator = price_path_simulator or GBMPriceSimulator()
        self.option_value_paths = None

    def generate_price_paths(self, S0, T, r, sigma, n_paths, n_steps):
        """Generate paths on the exercise-time grid used by LSMC."""
        return self.price_path_simulator.generate_price_paths(S0, T, r, sigma, n_paths, n_steps)

    def plot_price_paths(self, price_paths, max_paths=50):
        """Plot simulated underlying-price paths."""
        self.price_path_simulator.plot_price_paths(price_paths, max_paths)

    def plot_option_value_paths(self, max_paths=50):
        """Plot the pathwise values produced during backward induction."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.option_value_paths[:, :max_paths])
        plt.xlabel('Time')
        plt.ylabel('Discounted cash-flow value')
        plt.title('LSMC Pathwise Option Values')
        plt.grid(True)
        plt.show()
    
    def calculate_payoff(self, S, K):
        """Calculate the immediate payoff of an American put."""
        return np.maximum(K - S, 0)
    
    def estimate_continuation_value(self, prices, future_cash_flows, r, dt, basis_functions=None):
        """Estimate continuation values from discounted future cash flows."""

        # default basis function
        if basis_functions is None:
            basis_functions = [
                lambda S: np.ones_like(S),
                lambda S: S,
                lambda S: S ** 2,
            ]

        design_matrix = np.column_stack([basis_function(prices) for basis_function in basis_functions])
        regression_target = np.exp(-r * dt) * future_cash_flows
        coefficients = np.linalg.lstsq(
            design_matrix, regression_target, rcond=None
        )[0]

        return design_matrix @ coefficients

    def backward_induction(self, price_paths, K, r, dt, basis_functions=None):
        """Apply backward induction across the simulated exercise dates."""
        n_time_points = price_paths.shape[0]
        n_paths = price_paths.shape[1]
        discount_factor = np.exp(-r * dt)
        
        # Calculate the terminal option payoff max(K-S_T,0) for each path at at maturity (time T)
        cash_flows = self.calculate_payoff(price_paths[-1], K)
        option_value_paths = np.empty_like(price_paths)
        option_value_paths[-1, :] = cash_flows
        
        # Loop time step t from T-1 to 1 (CF at T is calculated above, CF at 0 is not needed) 
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
                continuation_value = self.estimate_continuation_value(
                    price_paths[t, in_the_money],
                    cash_flows[in_the_money],
                    r,
                    dt,
                    basis_functions,
                )

                # Determine the exercise decision for in-the-money paths.
                exercise_decision[in_the_money] = (
                    immediate_exercise_value[in_the_money] > continuation_value
                )

            # Move future cash flows from t+1 to t by discounting once, then replace exercised paths.
            cash_flows = discount_factor * cash_flows
            cash_flows[exercise_decision] = immediate_exercise_value[exercise_decision]
            option_value_paths[t, :] = cash_flows
        
        # Move the time-1 cash flows to time zero.
        discounted_cash_flows = discount_factor * cash_flows
        # Take the average as the option price
        continuation_value_at_zero = np.mean(discounted_cash_flows)

        # Exercise immediately if in-the-money already at time 0
        immediate_exercise_at_zero = self.calculate_payoff(price_paths[0, 0], K)
        if immediate_exercise_at_zero >= continuation_value_at_zero:
            option_value_paths[0, :] = immediate_exercise_at_zero
            self.option_value_paths = option_value_paths
            return float(immediate_exercise_at_zero), 0.0

        option_value_paths[0, :] = discounted_cash_flows
        self.option_value_paths = option_value_paths
        standard_error = np.std(discounted_cash_flows) / np.sqrt(n_paths)
        return float(continuation_value_at_zero), float(standard_error)
    
    def calculate_lsmc_price(self, S0, T, r, sigma, K, n_paths, n_steps, basis_functions=None):
        """Estimate an American put price and its Monte Carlo standard error."""
        dt = T / n_steps
        price_paths = self.generate_price_paths(S0, T, r, sigma, n_paths, n_steps)
        lsmc_price, std_error = self.backward_induction(
            price_paths, K, r, dt, basis_functions
        )
        return lsmc_price, std_error
