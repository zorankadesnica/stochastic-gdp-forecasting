"""
Composite Index construction with uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class CompositeIndexResult:
    """Container for composite index results."""
    values: np.ndarray
    weights: np.ndarray
    lower_ci: Optional[np.ndarray] = None
    upper_ci: Optional[np.ndarray] = None
    credible_level: float = 0.95


class CompositeIndex:
    """
    Composite Economic Index builder.
    
    Implements weighted aggregation of economic indicators
    with optional uncertainty quantification.
    
    Parameters
    ----------
    indicator_names : list of str
        Names of indicators to include
    normalize : bool
        If True, standardize indicators before aggregation
    """
    
    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        normalize: bool = True
    ):
        self.indicator_names = indicator_names or []
        self.normalize = normalize
        self._weights = None
        self._means = None
        self._stds = None
    
    def fit(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> 'CompositeIndex':
        """
        Fit the composite index (compute normalization parameters).
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
        weights : np.ndarray, optional
            Fixed weights. If None, use equal weights.
        """
        K = X.shape[1]
        
        if weights is None:
            self._weights = np.ones(K) / K
        else:
            self._weights = weights / weights.sum()
        
        if self.normalize:
            self._means = np.mean(X, axis=0)
            self._stds = np.std(X, axis=0)
            self._stds[self._stds == 0] = 1  # Avoid division by zero
        
        return self
    
    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute composite index values.
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
            
        Returns
        -------
        np.ndarray
            Composite index values (T,)
        """
        if self._weights is None:
            raise ValueError("Must call fit() first")
        
        if self.normalize:
            X_norm = (X - self._means) / self._stds
        else:
            X_norm = X
        
        return X_norm @ self._weights
    
    def fit_transform(
        self,
        X: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, weights)
        return self.transform(X)
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        cov_matrix: np.ndarray,
        n_samples: int = 1000,
        credible_level: float = 0.95
    ) -> CompositeIndexResult:
        """
        Compute composite index with uncertainty bounds.
        
        Uses Monte Carlo propagation of indicator uncertainty.
        
        Parameters
        ----------
        X : np.ndarray
            Indicator matrix (T x K)
        cov_matrix : np.ndarray
            Covariance matrix of indicator uncertainty
        n_samples : int
            Monte Carlo samples
        credible_level : float
            Credible interval level (e.g., 0.95 for 95%)
            
        Returns
        -------
        CompositeIndexResult
            Point estimates and credible intervals
        """
        T, K = X.shape
        rng = np.random.default_rng(42)
        
        # Generate Monte Carlo samples
        samples = np.zeros((n_samples, T))
        
        for i in range(n_samples):
            perturbation = rng.multivariate_normal(np.zeros(K), cov_matrix, size=T)
            X_perturbed = X + perturbation
            samples[i] = self.transform(X_perturbed)
        
        # Compute statistics
        values = self.transform(X)
        alpha = (1 - credible_level) / 2
        lower_ci = np.percentile(samples, 100 * alpha, axis=0)
        upper_ci = np.percentile(samples, 100 * (1 - alpha), axis=0)
        
        return CompositeIndexResult(
            values=values,
            weights=self._weights,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            credible_level=credible_level
        )


class SerbianCompositeIndex(CompositeIndex):
    """
    Serbian Composite Index implementation.
    
    Based on methodology from Vukmirovic et al. (2009-2015).
    
    Default indicators:
    - Industrial Production Index
    - Retail Trade Turnover
    - Employment
    - Construction Activity
    """
    
    DEFAULT_INDICATORS = ['ipi', 'retail', 'employment', 'construction']
    
    def __init__(
        self,
        indicators: Optional[List[str]] = None
    ):
        super().__init__(
            indicator_names=indicators or self.DEFAULT_INDICATORS,
            normalize=True
        )
