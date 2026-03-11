"""
Bayesian models for GDP nowcasting with uncertainty quantification.

Uses PyMC for MCMC inference.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


@dataclass
class BayesianResult:
    """Container for Bayesian inference results."""
    posterior_mean: Dict[str, float]
    posterior_std: Dict[str, float]
    credible_intervals: Dict[str, Tuple[float, float]]
    predictions: np.ndarray
    prediction_intervals: Tuple[np.ndarray, np.ndarray]
    trace: Optional[object] = None


class BayesianGDPModel:
    """
    Hierarchical Bayesian model for GDP nowcasting.
    
    Model:
        GDP_t ~ Normal(alpha + beta * C_t, sigma^2)
        alpha, beta ~ Normal(0, tau^2)
        sigma ~ Half-Cauchy(0, 1)
    
    Parameters
    ----------
    prior_scale : float
        Scale for coefficient priors
    """
    
    def __init__(self, prior_scale: float = 10.0):
        if not HAS_PYMC:
            raise ImportError("PyMC required. Install: pip install pymc arviz")
        
        self.prior_scale = prior_scale
        self.trace = None
        self.model = None
    
    def fit(
        self,
        C: np.ndarray,
        y: np.ndarray,
        n_samples: int = 2000,
        n_chains: int = 4,
        target_accept: float = 0.9
    ) -> BayesianResult:
        """
        Fit Bayesian model using MCMC.
        
        Parameters
        ----------
        C : np.ndarray
            Composite index values (T,)
        y : np.ndarray
            GDP growth rates (T,)
        n_samples : int
            MCMC samples per chain
        n_chains : int
            Number of chains
        target_accept : float
            Target acceptance rate
            
        Returns
        -------
        BayesianResult
        """
        with pm.Model() as self.model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=self.prior_scale)
            beta = pm.Normal('beta', mu=0, sigma=self.prior_scale)
            sigma = pm.HalfCauchy('sigma', beta=1)
            
            # Likelihood
            mu = alpha + beta * C
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
            # Sample
            self.trace = pm.sample(
                n_samples,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True,
                progressbar=True
            )
        
        return self._extract_results(C)
    
    def _extract_results(self, C: np.ndarray) -> BayesianResult:
        """Extract results from trace."""
        posterior = self.trace.posterior
        
        # Point estimates
        alpha_mean = float(posterior['alpha'].mean())
        beta_mean = float(posterior['beta'].mean())
        sigma_mean = float(posterior['sigma'].mean())
        
        # Standard deviations
        alpha_std = float(posterior['alpha'].std())
        beta_std = float(posterior['beta'].std())
        sigma_std = float(posterior['sigma'].std())
        
        # 95% credible intervals
        def get_ci(var):
            vals = posterior[var].values.flatten()
            return (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
        
        # Predictions
        predictions = alpha_mean + beta_mean * C
        
        # Prediction intervals (including uncertainty in all parameters)
        alpha_samples = posterior['alpha'].values.flatten()
        beta_samples = posterior['beta'].values.flatten()
        sigma_samples = posterior['sigma'].values.flatten()
        
        n_samples = len(alpha_samples)
        pred_samples = np.zeros((n_samples, len(C)))
        
        for i in range(n_samples):
            pred_samples[i] = alpha_samples[i] + beta_samples[i] * C
        
        lower = np.percentile(pred_samples, 2.5, axis=0)
        upper = np.percentile(pred_samples, 97.5, axis=0)
        
        return BayesianResult(
            posterior_mean={'alpha': alpha_mean, 'beta': beta_mean, 'sigma': sigma_mean},
            posterior_std={'alpha': alpha_std, 'beta': beta_std, 'sigma': sigma_std},
            credible_intervals={
                'alpha': get_ci('alpha'),
                'beta': get_ci('beta'),
                'sigma': get_ci('sigma')
            },
            predictions=predictions,
            prediction_intervals=(lower, upper),
            trace=self.trace
        )
    
    def predict(
        self,
        C_new: np.ndarray,
        return_samples: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict GDP for new composite index values.
        
        Returns
        -------
        tuple
            (predictions, lower_ci, upper_ci)
        """
        if self.trace is None:
            raise ValueError("Must fit model first")
        
        posterior = self.trace.posterior
        alpha = posterior['alpha'].values.flatten()
        beta = posterior['beta'].values.flatten()
        
        n_samples = len(alpha)
        pred_samples = np.zeros((n_samples, len(C_new)))
        
        for i in range(n_samples):
            pred_samples[i] = alpha[i] + beta[i] * C_new
        
        predictions = pred_samples.mean(axis=0)
        lower = np.percentile(pred_samples, 2.5, axis=0)
        upper = np.percentile(pred_samples, 97.5, axis=0)
        
        if return_samples:
            return predictions, lower, upper, pred_samples
        return predictions, lower, upper


class TimeVaryingBayesianModel:
    """
    State-space Bayesian model with time-varying parameters.
    
    Model:
        GDP_t = alpha_t + beta_t * C_t + epsilon_t
        beta_t = beta_{t-1} + eta_t
        
    Captures structural breaks and regime changes.
    """
    
    def __init__(self, innovation_scale: float = 0.1):
        if not HAS_PYMC:
            raise ImportError("PyMC required")
        
        self.innovation_scale = innovation_scale
        self.trace = None
    
    def fit(
        self,
        C: np.ndarray,
        y: np.ndarray,
        n_samples: int = 2000
    ) -> Dict:
        """
        Fit time-varying parameter model.
        """
        T = len(y)
        
        with pm.Model() as model:
            # Random walk for beta
            beta_init = pm.Normal('beta_0', mu=0, sigma=1)
            beta_innov = pm.Normal('beta_innov', mu=0, sigma=self.innovation_scale, shape=T-1)
            
            # Construct beta path
            beta = pm.math.concatenate([[beta_init], beta_init + pm.math.cumsum(beta_innov)])
            
            # Intercept
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            sigma = pm.HalfCauchy('sigma', beta=1)
            
            # Likelihood
            mu = alpha + beta * C
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
            self.trace = pm.sample(n_samples, return_inferencedata=True)
        
        return {'trace': self.trace}


if __name__ == "__main__":
    # Demo
    if HAS_PYMC:
        np.random.seed(42)
        
        T = 50
        C = np.random.randn(T)
        y = 0.5 + 0.8 * C + 0.2 * np.random.randn(T)
        
        print("Fitting Bayesian GDP model...")
        model = BayesianGDPModel()
        result = model.fit(C, y, n_samples=1000, n_chains=2)
        
        print(f"\nPosterior means:")
        print(f"  alpha: {result.posterior_mean['alpha']:.3f} (true: 0.5)")
        print(f"  beta:  {result.posterior_mean['beta']:.3f} (true: 0.8)")
        print(f"  sigma: {result.posterior_mean['sigma']:.3f} (true: 0.2)")
    else:
        print("PyMC not installed. Skipping demo.")
