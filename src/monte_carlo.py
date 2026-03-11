"""
Monte Carlo simulation for stress testing and scenario analysis.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass 
class ScenarioResult:
    """Container for scenario analysis results."""
    scenarios: np.ndarray
    composite_index_values: np.ndarray
    mean: float
    std: float
    var_5: float  # Value at Risk 5%
    var_1: float  # Value at Risk 1%
    expected_shortfall_5: float


class MonteCarloSimulator:
    """
    Monte Carlo scenario generator for stress testing.
    
    Parameters
    ----------
    n_scenarios : int
        Number of scenarios to generate
    random_state : int, optional
        Random seed
    """
    
    def __init__(self, n_scenarios: int = 10000, random_state: Optional[int] = None):
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(random_state)
    
    def generate_scenarios(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        n_periods: int = 1
    ) -> np.ndarray:
        """Generate multivariate normal scenarios."""
        K = len(mean)
        return self.rng.multivariate_normal(mean, cov, size=(self.n_scenarios, n_periods))
    
    def historical_simulation(
        self,
        historical_data: np.ndarray,
        n_periods: int = 12
    ) -> np.ndarray:
        """Bootstrap historical scenarios."""
        T = len(historical_data)
        indices = self.rng.integers(0, T - n_periods, size=self.n_scenarios)
        
        scenarios = np.zeros((self.n_scenarios, n_periods, historical_data.shape[1]))
        for i, idx in enumerate(indices):
            scenarios[i] = historical_data[idx:idx + n_periods]
        
        return scenarios
    
    def stress_test(
        self,
        baseline: np.ndarray,
        weights: np.ndarray,
        cov: np.ndarray,
        shock_multiplier: float = 2.0
    ) -> ScenarioResult:
        """
        Perform stress test with amplified volatility.
        
        Parameters
        ----------
        baseline : np.ndarray
            Baseline indicator values (K,)
        weights : np.ndarray
            Composite index weights (K,)
        cov : np.ndarray
            Covariance matrix (K x K)
        shock_multiplier : float
            Multiply covariance by this factor for stress
        """
        stressed_cov = cov * shock_multiplier
        scenarios = self.generate_scenarios(baseline, stressed_cov)
        
        # Compute composite index for each scenario
        ci_values = scenarios @ weights
        
        return ScenarioResult(
            scenarios=scenarios,
            composite_index_values=ci_values.flatten(),
            mean=ci_values.mean(),
            std=ci_values.std(),
            var_5=np.percentile(ci_values, 5),
            var_1=np.percentile(ci_values, 1),
            expected_shortfall_5=ci_values[ci_values <= np.percentile(ci_values, 5)].mean()
        )
    
    def sobol_sensitivity(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        n_samples: int = 5000
    ) -> Dict[str, np.ndarray]:
        """
        Compute Sobol sensitivity indices.
        
        Returns first-order indices showing which indicators
        contribute most to composite index variance.
        """
        K = len(weights)
        
        # Simplified first-order indices based on variance decomposition
        variances = np.diag(cov)
        weighted_vars = (weights ** 2) * variances
        total_var = weighted_vars.sum()
        
        first_order = weighted_vars / total_var if total_var > 0 else np.ones(K) / K
        
        return {
            'first_order': first_order,
            'total_variance': total_var
        }


if __name__ == "__main__":
    np.random.seed(42)
    
    # Demo
    K = 4
    baseline = np.zeros(K)
    weights = np.array([0.3, 0.3, 0.25, 0.15])
    cov = np.eye(K) * 0.01
    
    sim = MonteCarloSimulator(n_scenarios=10000, random_state=42)
    result = sim.stress_test(baseline, weights, cov, shock_multiplier=3.0)
    
    print("Stress Test Results")
    print("=" * 40)
    print(f"Mean CI:              {result.mean:.4f}")
    print(f"Std CI:               {result.std:.4f}")
    print(f"VaR 5%:               {result.var_5:.4f}")
    print(f"VaR 1%:               {result.var_1:.4f}")
    print(f"Expected Shortfall:   {result.expected_shortfall_5:.4f}")
