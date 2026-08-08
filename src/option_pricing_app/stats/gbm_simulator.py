"""GBM simulator migrated from the dissertation without algorithm changes."""

import numpy as np
import matplotlib.pyplot as plt

class GBMPriceSimulator:
    def generate_price_paths(self, S0, T, r, sigma, n_paths, n_steps):
        """Generate risk-neutral GBM paths over the interval from zero to T."""
        dt = T / n_steps
        price_paths = np.empty((n_steps + 1, n_paths), dtype=np.float64)
        price_paths[0, :] = S0

        for t in range(1, n_steps + 1):
            Z = np.random.randn(n_paths)
            price_paths[t, :] = price_paths[t - 1, :] * np.exp(
                (r - 0.5 * sigma ** 2) * dt
                + sigma * np.sqrt(dt) * Z
            )

        return price_paths

    def plot_price_paths(self, price_paths, max_paths=50):
        """Plot simulated price paths."""
        plt.figure(figsize=(10, 5))
        plt.plot(price_paths[:, :max_paths])
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Simulated Price Paths')
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
