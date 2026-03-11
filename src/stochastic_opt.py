"""
Stochastic Optimization for Composite Index Weight Determination.

Methods:
- Sample Average Approximation (SAA)
- Robust Optimization
- Adaptive Stochastic Gradient Descent
"""

import numpy as np
from scipy import optimize
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
import warnings

try:
    import cvxpy as cp
except ImportError:
    cp = None


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray
    objective_value: float
    n_iterations: int
    converged: bool
    uncertainty_set: Optional[np.ndarray] = None
    weight_history: Optional[List[np.ndarray]] = None


class SAAOptimizer:
    """
    Sample Average Approximation optimizer for composite index weights.
    
    Solves: min_w (1/N) * sum_{n=1}^N L(C(w, X + xi_n), GDP)
    
    where xi_n are Monte Carlo samples of data perturbations.
    
    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo samples for SAA
    loss : str
        Loss function: 'mse', 'mae', 'quantile'
    quantile : float
        Quantile level if loss='quantile' (default 0.5)
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        loss: str = 'mse',
        quantile: float = 0.5,
        random_state: Optional[int] = None
    ):
        self.n_samples = n_samples
        self.loss = loss
        self.quantile = quantile
        self.rng = np.random.default_rng(random_state)
        
    def _loss_function(
        self,
        predicted: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute loss value."""
        residuals = predicted - target
        
        if self.loss == 'mse':
            return np.mean(residuals ** 2)
        elif self.loss == 'mae':
            return np.mean(np.abs(residuals))
        elif self.loss == 'quantile':
            return np.mean(
                np.where(
                    residuals >= 0,
                    self.quantile * residuals,
                    (self.quantile - 1) * residuals
                )
            )
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
    
    def _generate_perturbations(
        self,
        cov_matrix: np.ndarray,
        n_indicators: int
    ) -> np.ndarray:
        """Generate Monte Carlo perturbation samples."""
        return self.rng.multivariate_normal(
            mean=np.zeros(n_indicators),
            cov=cov_matrix,
            size=self.n_samples
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        initial_weights: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Fit optimal weights using SAA.
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
        y : np.ndarray
            Target variable (T,), e.g., GDP growth
        cov_matrix : np.ndarray, optional
            Covariance matrix for perturbations.
            If None, estimated from X.
        initial_weights : np.ndarray, optional
            Starting weights for optimization
            
        Returns
        -------
        OptimizationResult
            Optimal weights and metadata
        """
        T, K = X.shape
        
        # Estimate covariance if not provided
        if cov_matrix is None:
            # Use scaled empirical covariance (measurement error assumption)
            cov_matrix = np.cov(X.T) * 0.01  # 1% measurement error scale
        
        # Generate perturbations
        perturbations = self._generate_perturbations(cov_matrix, K)
        
        # Initial weights
        if initial_weights is None:
            initial_weights = np.ones(K) / K  # Equal weights
        
        # Objective function (SAA)
        def objective(w):
            total_loss = 0.0
            for xi in perturbations:
                # Perturbed indicators
                X_perturbed = X + xi
                # Composite index
                C = X_perturbed @ w
                # Loss
                total_loss += self._loss_function(C, y)
            return total_loss / self.n_samples
        
        # Constraints: sum(w) = 1, w >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(K)]
        
        # Optimize
        result = optimize.minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        return OptimizationResult(
            weights=result.x,
            objective_value=result.fun,
            n_iterations=result.nit,
            converged=result.success
        )


class RobustOptimizer:
    """
    Robust optimization for worst-case composite index weights.
    
    Solves: min_w max_{xi in U} L(C(w, X + xi), GDP)
    
    where U is the uncertainty set.
    
    Parameters
    ----------
    uncertainty_type : str
        Type of uncertainty set: 'ellipsoidal', 'box', 'polyhedral'
    gamma : float
        Uncertainty set size parameter
    """
    
    def __init__(
        self,
        uncertainty_type: str = 'ellipsoidal',
        gamma: float = 1.0
    ):
        self.uncertainty_type = uncertainty_type
        self.gamma = gamma
        
        if cp is None:
            raise ImportError("cvxpy required for robust optimization")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Fit robust optimal weights.
        
        For ellipsoidal uncertainty, the robust counterpart becomes:
        min_w ||y - Xw||_2 + gamma * ||Sigma^{1/2} w||_2
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
        y : np.ndarray
            Target variable (T,)
        cov_matrix : np.ndarray, optional
            Covariance for uncertainty set
            
        Returns
        -------
        OptimizationResult
        """
        T, K = X.shape
        
        if cov_matrix is None:
            cov_matrix = np.cov(X.T) * 0.01
        
        # CVXPY formulation
        w = cp.Variable(K)
        
        # Objective: minimize worst-case loss
        if self.uncertainty_type == 'ellipsoidal':
            # Robust regression with ellipsoidal uncertainty
            sqrt_cov = np.linalg.cholesky(cov_matrix + 1e-6 * np.eye(K))
            
            objective = cp.norm(y - X @ w, 2) + self.gamma * cp.norm(sqrt_cov @ w, 2)
        else:
            # Box uncertainty (simpler)
            objective = cp.norm(y - X @ w, 2) + self.gamma * cp.norm(w, 1)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        # Solve
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.ECOS)
        
        return OptimizationResult(
            weights=w.value,
            objective_value=problem.value,
            n_iterations=problem.solver_stats.num_iters if problem.solver_stats else 0,
            converged=problem.status == 'optimal',
            uncertainty_set=cov_matrix
        )


class AdaptiveSGDOptimizer:
    """
    Adaptive Stochastic Gradient Descent for online weight updates.
    
    Suitable for real-time composite index adjustment as new data arrives.
    
    Parameters
    ----------
    learning_rate : float
        Initial learning rate
    decay : float
        Learning rate decay factor
    momentum : float
        Momentum coefficient
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        decay: float = 0.99,
        momentum: float = 0.9
    ):
        self.lr = learning_rate
        self.decay = decay
        self.momentum = momentum
        
        self._velocity = None
        self._weights = None
        self._iteration = 0
    
    def initialize(self, n_indicators: int):
        """Initialize weights and velocity."""
        self._weights = np.ones(n_indicators) / n_indicators
        self._velocity = np.zeros(n_indicators)
        self._iteration = 0
    
    def step(
        self,
        X_t: np.ndarray,
        y_t: float,
        perturbation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Single SGD update step.
        
        Parameters
        ----------
        X_t : np.ndarray
            Current period indicators (K,)
        y_t : float
            Current period target
        perturbation : np.ndarray, optional
            Random perturbation for stochastic gradient
            
        Returns
        -------
        np.ndarray
            Updated weights
        """
        K = len(X_t)
        
        if self._weights is None:
            self.initialize(K)
        
        # Add perturbation
        if perturbation is not None:
            X_perturbed = X_t + perturbation
        else:
            X_perturbed = X_t
        
        # Compute gradient (MSE loss)
        prediction = X_perturbed @ self._weights
        error = prediction - y_t
        gradient = 2 * error * X_perturbed
        
        # Update with momentum
        self._velocity = self.momentum * self._velocity - self.lr * gradient
        self._weights = self._weights + self._velocity
        
        # Project onto simplex (sum = 1, w >= 0)
        self._weights = self._project_simplex(self._weights)
        
        # Decay learning rate
        self.lr *= self.decay
        self._iteration += 1
        
        return self._weights.copy()
    
    def _project_simplex(self, w: np.ndarray) -> np.ndarray:
        """Project weights onto probability simplex."""
        # Clip negative values
        w = np.maximum(w, 0)
        # Normalize
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = np.ones_like(w) / len(w)
        return w
    
    def fit_online(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        return_history: bool = False
    ) -> OptimizationResult:
        """
        Fit weights using online SGD over entire dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
        y : np.ndarray
            Target variable (T,)
        cov_matrix : np.ndarray, optional
            Covariance for perturbation sampling
        return_history : bool
            If True, return weight history
            
        Returns
        -------
        OptimizationResult
        """
        T, K = X.shape
        
        self.initialize(K)
        
        if cov_matrix is None:
            cov_matrix = np.cov(X.T) * 0.01
        
        rng = np.random.default_rng(42)
        history = [] if return_history else None
        
        for t in range(T):
            perturbation = rng.multivariate_normal(np.zeros(K), cov_matrix)
            self.step(X[t], y[t], perturbation)
            
            if return_history:
                history.append(self._weights.copy())
        
        # Compute final objective
        predictions = X @ self._weights
        final_loss = np.mean((predictions - y) ** 2)
        
        return OptimizationResult(
            weights=self._weights,
            objective_value=final_loss,
            n_iterations=T,
            converged=True,
            weight_history=history
        )


def compare_optimizers(
    X: np.ndarray,
    y: np.ndarray,
    cov_matrix: Optional[np.ndarray] = None
) -> dict:
    """
    Compare different optimization methods.
    
    Returns
    -------
    dict
        Results from each optimizer
    """
    results = {}
    
    # SAA
    saa = SAAOptimizer(n_samples=500)
    results['saa'] = saa.fit(X, y, cov_matrix)
    
    # Robust (if cvxpy available)
    if cp is not None:
        robust = RobustOptimizer(gamma=0.5)
        results['robust'] = robust.fit(X, y, cov_matrix)
    
    # Adaptive SGD
    sgd = AdaptiveSGDOptimizer(learning_rate=0.05)
    results['adaptive_sgd'] = sgd.fit_online(X, y, cov_matrix)
    
    return results


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    
    # Simulate data
    T, K = 100, 5
    true_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    X = np.random.randn(T, K)
    y = X @ true_weights + 0.1 * np.random.randn(T)
    
    print("Stochastic Optimization Demo")
    print("=" * 50)
    print(f"True weights: {true_weights}")
    
    # SAA
    saa = SAAOptimizer(n_samples=1000, random_state=42)
    result = saa.fit(X, y)
    print(f"\nSAA weights:  {result.weights.round(3)}")
    print(f"SAA loss:     {result.objective_value:.6f}")
    
    # Adaptive SGD
    sgd = AdaptiveSGDOptimizer(learning_rate=0.05)
    result_sgd = sgd.fit_online(X, y)
    print(f"\nSGD weights:  {result_sgd.weights.round(3)}")
    print(f"SGD loss:     {result_sgd.objective_value:.6f}")
