"""Tests for stochastic optimization."""
import numpy as np
import sys
sys.path.insert(0, '../src')

def test_saa_convergence():
    """Test SAA recovers true weights."""
    from stochastic_opt import SAAOptimizer
    
    np.random.seed(42)
    T, K = 100, 3
    true_weights = np.array([0.5, 0.3, 0.2])
    
    X = np.random.randn(T, K)
    y = X @ true_weights + 0.1 * np.random.randn(T)
    
    optimizer = SAAOptimizer(n_samples=500, random_state=42)
    result = optimizer.fit(X, y)
    
    assert result.converged
    assert np.allclose(result.weights, true_weights, atol=0.15)
    print("PASS: SAA convergence test")

if __name__ == "__main__":
    test_saa_convergence()
