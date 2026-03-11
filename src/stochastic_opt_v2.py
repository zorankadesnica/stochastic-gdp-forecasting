"""
Stochastic Optimization Methods That Can Beat OLS/Ridge/Lasso
==============================================================

Key insight: The simplex constraint (w≥0, Σw=1) is what limits SAA.
These methods use stochastic perturbations for REGULARIZATION instead.

Methods:
1. StochasticRegression: Unconstrained regression trained on perturbed data
2. DRORegression: Distributionally robust optimization (worst-case over perturbations)
3. CVaRRegression: Optimizes conditional value-at-risk (tail risk)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin


class StochasticRegression(BaseEstimator, RegressorMixin):
    """
    Stochastic Regression (NO simplex constraint)
    ==============================================
    Trains unconstrained regression on perturbed data.
    The perturbations provide implicit regularization that can beat Ridge.
    
    Math: min_β E_ξ[||y - (X+ξ)β||²]
    
    This is equivalent to Ridge with a specific regularization structure
    that adapts to the noise covariance.
    """
    
    def __init__(self, n_samples=100, noise_std=0.05, random_state=None):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n, p = X.shape
        
        # Aggregate OLS solutions from perturbed data
        betas = []
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X.shape)
            X_pert = X + noise
            
            # OLS on perturbed data
            beta = np.linalg.lstsq(
                np.c_[np.ones(n), X_pert], 
                y, 
                rcond=None
            )[0]
            betas.append(beta)
        
        # Average (ensemble effect reduces variance)
        beta_avg = np.mean(betas, axis=0)
        self.intercept_ = beta_avg[0]
        self.coef_ = beta_avg[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class DRORegression(BaseEstimator, RegressorMixin):
    """
    Distributionally Robust Regression
    ===================================
    Optimizes for the WORST-CASE perturbation within a ball.
    
    Math: min_β max_{||ξ||≤ε} ||y - (X+ξ)β||²
    
    This provides stronger regularization than Ridge in some cases.
    """
    
    def __init__(self, epsilon=0.1, random_state=None):
        self.epsilon = epsilon
        self.random_state = random_state
        
    def fit(self, X, y):
        n, p = X.shape
        
        def objective(params):
            intercept = params[0]
            beta = params[1:]
            
            # Compute worst-case loss
            # For L2 ball perturbation, worst-case adds ε*||β|| to residuals
            residuals = y - intercept - X @ beta
            base_loss = np.sum(residuals ** 2)
            
            # Worst-case perturbation increases loss proportionally to ||β||
            # This is equivalent to elastic net-like regularization
            penalty = self.epsilon * n * np.sum(beta ** 2)
            
            return base_loss + penalty
        
        # Initialize with OLS
        ols = LinearRegression().fit(X, y)
        x0 = np.r_[ols.intercept_, ols.coef_]
        
        result = minimize(objective, x0, method='L-BFGS-B')
        
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class CVaRRegression(BaseEstimator, RegressorMixin):
    """
    CVaR (Conditional Value-at-Risk) Regression
    ============================================
    Instead of minimizing mean squared error, minimizes the 
    average of the WORST α% of squared errors.
    
    This makes the model robust to outliers and extreme observations.
    """
    
    def __init__(self, alpha=0.2, random_state=None):
        self.alpha = alpha  # Focus on worst 20% of errors
        self.random_state = random_state
        
    def fit(self, X, y):
        n, p = X.shape
        k = max(1, int(n * self.alpha))  # Number of worst cases
        
        def objective(params):
            intercept = params[0]
            beta = params[1:]
            
            residuals = y - intercept - X @ beta
            squared_errors = residuals ** 2
            
            # CVaR: average of k largest squared errors
            worst_k = np.sort(squared_errors)[-k:]
            return np.mean(worst_k)
        
        # Initialize with OLS
        ols = LinearRegression().fit(X, y)
        x0 = np.r_[ols.intercept_, ols.coef_]
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 1000})
        
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class TrimmedRegression(BaseEstimator, RegressorMixin):
    """
    Trimmed Least Squares with Stochastic Selection
    =================================================
    Randomly excludes some observations in each iteration,
    then averages the resulting models.
    
    Robust to outliers and structural breaks.
    """
    
    def __init__(self, trim_fraction=0.1, n_samples=100, random_state=None):
        self.trim_fraction = trim_fraction
        self.n_samples = n_samples
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n, p = X.shape
        n_keep = int(n * (1 - self.trim_fraction))
        
        betas = []
        for _ in range(self.n_samples):
            # Random subsample
            idx = np.random.choice(n, size=n_keep, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]
            
            # OLS on subsample
            beta = np.linalg.lstsq(
                np.c_[np.ones(n_keep), X_sub],
                y_sub,
                rcond=None
            )[0]
            betas.append(beta)
        
        beta_avg = np.mean(betas, axis=0)
        self.intercept_ = beta_avg[0]
        self.coef_ = beta_avg[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class AdaptiveNoiseRegression(BaseEstimator, RegressorMixin):
    """
    Adaptive Noise Regression
    ==========================
    Uses cross-validation to find optimal noise level for perturbation.
    Noise level is chosen to minimize out-of-sample error.
    """
    
    def __init__(self, noise_levels=None, n_samples=50, random_state=None):
        self.noise_levels = noise_levels or [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
        self.n_samples = n_samples
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n, p = X.shape
        
        # Simple time-series CV: use last 20% as validation
        n_val = max(5, int(n * 0.2))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        best_noise = 0.05
        best_error = np.inf
        
        for noise_std in self.noise_levels:
            # Train stochastic model
            model = StochasticRegression(
                n_samples=self.n_samples,
                noise_std=noise_std,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            
            # Validate
            pred_val = model.predict(X_val)
            error = np.mean((y_val - pred_val) ** 2)
            
            if error < best_error:
                best_error = error
                best_noise = noise_std
        
        # Retrain on full data with best noise level
        self.best_noise_ = best_noise
        self.model_ = StochasticRegression(
            n_samples=self.n_samples * 2,  # More samples for final model
            noise_std=best_noise,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        self.intercept_ = self.model_.intercept_
        self.coef_ = self.model_.coef_
        
        return self
    
    def predict(self, X):
        return self.model_.predict(X)


class WeightedStochasticRegression(BaseEstimator, RegressorMixin):
    """
    Time-Weighted Stochastic Regression
    ====================================
    Combines stochastic perturbation with exponential time weighting.
    Recent observations matter more.
    """
    
    def __init__(self, n_samples=100, noise_std=0.05, decay=0.97, random_state=None):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.decay = decay
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n, p = X.shape
        
        # Time weights (exponential decay)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum() * n  # Normalize
        W = np.diag(np.sqrt(weights))
        
        betas = []
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X.shape)
            X_pert = X + noise
            
            # Weighted OLS on perturbed data
            X_w = W @ np.c_[np.ones(n), X_pert]
            y_w = W @ y
            
            beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
            betas.append(beta)
        
        beta_avg = np.mean(betas, axis=0)
        self.intercept_ = beta_avg[0]
        self.coef_ = beta_avg[1:]
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_