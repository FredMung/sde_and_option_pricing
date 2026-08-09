"""Risk-neutral asset-price simulators used by the option-pricing application."""

import matplotlib.pyplot as plt
import numpy as np


class GBMPriceSimulator:
    def generate_price_paths(self, S0, T, r, sigma, n_paths, n_steps):
        """Generate risk-neutral GBM paths over the interval from zero to T."""
        dt = T / n_steps
        price_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        price_paths[0, :] = S0

        for t in range(1, n_steps + 1):
            Z = np.random.randn(n_paths)
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(
                (r - 0.5 * sigma**2) * dt
                + sigma * np.sqrt(dt) * Z
            )

        return price_paths

    def plot_price_paths(self, price_paths, max_paths=50):
        """Plot simulated price paths."""
        plt.figure(figsize=(10, 5))
        plt.plot(price_paths[:, :max_paths])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Simulated Price Paths")
        plt.grid(True)
        plt.show()

    def calculate_option_price(self, S0, T, r, sigma, K, n_paths, n_steps):
        """Estimate a European put price from simulated terminal prices."""
        price_paths = self.generate_price_paths(
            S0, T, r, sigma, n_paths, n_steps
        )
        terminal_prices = price_paths[-1, :]
        terminal_payoffs = np.maximum(K - terminal_prices, 0)
        discounted_payoffs = np.exp(-r * T) * terminal_payoffs
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

        return float(option_price), float(standard_error)


class HestonPriceSimulator:
    def generate_price_paths(
        self, S0, T, r, kappa, theta, sigma_v, rho, v0, n_paths, n_steps
    ):
        """Generate risk-neutral Heston model paths over the interval from zero to T."""
        dt = T / n_steps

        # Step 0: Initialise arrays for the simulated price and variance paths.
        price_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        price_paths[0, :] = S0

        var_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        var_paths[0, :] = v0

        for t in range(1, n_steps + 1):
            # Generate dependent normals for the correlated Brownian motions.
            Zv = np.random.randn(n_paths)
            Zs = rho * Zv + np.sqrt(1 - rho**2) * np.random.randn(n_paths)

            # Update variance with a non-negative truncated Euler step.
            previous_variance = np.maximum(var_paths[t - 1, :], 0.0)
            var_paths[t, :] = np.maximum(
                previous_variance
                + kappa * (theta - previous_variance) * dt
                + sigma_v * np.sqrt(previous_variance * dt) * Zv,
                0,
            )

            # Freeze v_t over the log-Euler asset step to preserve positive prices.
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(
                (r - 0.5 * previous_variance) * dt
                + np.sqrt(previous_variance * dt) * Zs
            )

        return price_paths, var_paths

    def plot_price_paths(self, price_paths, max_paths=50):
        """Plot simulated Heston asset-price paths."""
        plt.figure(figsize=(10, 5))
        plt.plot(price_paths[:, :max_paths])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("Simulated Heston Price Paths")
        plt.grid(True)
        plt.show()

    def plot_variance_paths(self, variance_paths, max_paths=50):
        """Plot the instantaneous variance paths produced by Heston simulation."""
        plt.figure(figsize=(10, 5))
        plt.plot(variance_paths[:, :max_paths])
        plt.xlabel("Time")
        plt.ylabel("Variance")
        plt.title("Simulated Heston Variance Paths")
        plt.grid(True)
        plt.show()

    def calculate_option_price(
        self, S0, T, r, kappa, theta, sigma_v, rho, v0, K, n_paths, n_steps
    ):
        """Estimate a European put price from Heston terminal prices."""
        price_paths, _variance_paths = self.generate_price_paths(
            S0, T, r, kappa, theta, sigma_v, rho, v0, n_paths, n_steps
        )
        terminal_payoffs = np.maximum(K - price_paths[-1, :], 0.0)
        discounted_payoffs = np.exp(-r * T) * terminal_payoffs
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        return float(option_price), float(standard_error)
