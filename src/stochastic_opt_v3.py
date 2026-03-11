"""
Aggressive Stochastic Methods to Beat Lasso
============================================
Lasso RMSE = 0.0329. Target: < 0.0329

Strategies:
1. Combine feature selection (like Lasso) WITH perturbation
2. Use ensemble averaging across multiple noise levels
3. Nonlinear feature augmentation
4. Cross-validated hyperparameter tuning
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, LassoCV, Ridge, ElasticNet, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin


class StochasticLasso(BaseEstimator, RegressorMixin):
    """
    Stochastic Lasso: Lasso trained on perturbed data
    ==================================================
    Combines L1 sparsity with perturbation regularization.
    Should beat regular Lasso by adding implicit L2-like regularization.
    """
    
    def __init__(self, n_samples=100, noise_std=0.03, alpha=0.001, random_state=None):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.alpha = alpha
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        coefs = []
        intercepts = []
        
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X.shape)
            X_pert = X + noise
            
            lasso = Lasso(alpha=self.alpha, max_iter=10000, tol=1e-4)
            lasso.fit(X_pert, y)
            
            coefs.append(lasso.coef_)
            intercepts.append(lasso.intercept_)
        
        self.coef_ = np.mean(coefs, axis=0)
        self.intercept_ = np.mean(intercepts)
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class StochasticElasticNet(BaseEstimator, RegressorMixin):
    """
    Stochastic ElasticNet: Best of L1 + L2 + perturbation
    """
    
    def __init__(self, n_samples=100, noise_std=0.03, alpha=0.001, 
                 l1_ratio=0.5, random_state=None):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        coefs = []
        intercepts = []
        
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X.shape)
            X_pert = X + noise
            
            en = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, 
                           max_iter=10000, tol=1e-4)
            en.fit(X_pert, y)
            
            coefs.append(en.coef_)
            intercepts.append(en.intercept_)
        
        self.coef_ = np.mean(coefs, axis=0)
        self.intercept_ = np.mean(intercepts)
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_


class CVStochasticLasso(BaseEstimator, RegressorMixin):
    """
    Cross-validated Stochastic Lasso
    =================================
    Uses internal CV to tune both alpha AND noise_std.
    """
    
    def __init__(self, n_samples=50, noise_levels=None, alphas=None, random_state=None):
        self.n_samples = n_samples
        self.noise_levels = noise_levels or [0.01, 0.02, 0.03, 0.05, 0.07]
        self.alphas = alphas or [0.0001, 0.0005, 0.001, 0.002, 0.005]
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n = len(y)
        n_val = max(5, int(n * 0.2))
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        best_error = np.inf
        best_params = (0.03, 0.001)
        
        for noise_std in self.noise_levels:
            for alpha in self.alphas:
                model = StochasticLasso(
                    n_samples=self.n_samples,
                    noise_std=noise_std,
                    alpha=alpha,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                error = np.mean((y_val - pred) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_params = (noise_std, alpha)
        
        self.best_noise_, self.best_alpha_ = best_params
        
        # Retrain on full data
        self.model_ = StochasticLasso(
            n_samples=self.n_samples * 2,
            noise_std=self.best_noise_,
            alpha=self.best_alpha_,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        
        return self
    
    def predict(self, X):
        return self.model_.predict(X)


class EnsembleStochasticRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble of diverse stochastic models
    ======================================
    Combines Lasso, Ridge, ElasticNet with different noise levels.
    Ensemble averaging reduces variance.
    """
    
    def __init__(self, n_samples=50, random_state=None):
        self.n_samples = n_samples
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.models_ = []
        
        # Different model configurations
        configs = [
            ('lasso', 0.02, 0.0005),
            ('lasso', 0.03, 0.001),
            ('lasso', 0.05, 0.001),
            ('ridge', 0.03, 0.5),
            ('ridge', 0.05, 1.0),
            ('elasticnet', 0.03, 0.001),
        ]
        
        for model_type, noise_std, alpha in configs:
            coefs = []
            intercepts = []
            
            for _ in range(self.n_samples):
                noise = np.random.normal(0, noise_std, X.shape)
                X_pert = X + noise
                
                if model_type == 'lasso':
                    model = Lasso(alpha=alpha, max_iter=10000)
                elif model_type == 'ridge':
                    model = Ridge(alpha=alpha)
                else:
                    model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
                
                model.fit(X_pert, y)
                coefs.append(model.coef_)
                intercepts.append(model.intercept_)
            
            self.models_.append({
                'coef': np.mean(coefs, axis=0),
                'intercept': np.mean(intercepts)
            })
        
        return self
    
    def predict(self, X):
        preds = []
        for m in self.models_:
            pred = m['intercept'] + X @ m['coef']
            preds.append(pred)
        return np.mean(preds, axis=0)


class PolyStochasticLasso(BaseEstimator, RegressorMixin):
    """
    Polynomial features + Stochastic Lasso
    =======================================
    Adds interaction terms then applies stochastic lasso.
    Can capture nonlinear relationships.
    """
    
    def __init__(self, degree=2, n_samples=80, noise_std=0.02, 
                 alpha=0.002, random_state=None):
        self.degree = degree
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.alpha = alpha
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Create polynomial features (interactions only, no bias)
        self.poly_ = PolynomialFeatures(degree=self.degree, 
                                        include_bias=False,
                                        interaction_only=True)
        X_poly = self.poly_.fit_transform(X)
        
        coefs = []
        intercepts = []
        
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X_poly.shape)
            X_pert = X_poly + noise
            
            lasso = Lasso(alpha=self.alpha, max_iter=10000, tol=1e-4)
            lasso.fit(X_pert, y)
            
            coefs.append(lasso.coef_)
            intercepts.append(lasso.intercept_)
        
        self.coef_ = np.mean(coefs, axis=0)
        self.intercept_ = np.mean(intercepts)
        
        return self
    
    def predict(self, X):
        X_poly = self.poly_.transform(X)
        return self.intercept_ + X_poly @ self.coef_


class BaggingStochasticLasso(BaseEstimator, RegressorMixin):
    """
    Bagging + Stochastic Lasso
    ===========================
    Bootstrap aggregation with stochastic perturbation.
    Double variance reduction.
    """
    
    def __init__(self, n_bags=30, n_samples=30, noise_std=0.03, 
                 alpha=0.001, random_state=None):
        self.n_bags = n_bags
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.alpha = alpha
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n = len(y)
        self.models_ = []
        
        for b in range(self.n_bags):
            # Bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            # Stochastic Lasso on bootstrap sample
            coefs = []
            intercepts = []
            
            for _ in range(self.n_samples):
                noise = np.random.normal(0, self.noise_std, X_boot.shape)
                X_pert = X_boot + noise
                
                lasso = Lasso(alpha=self.alpha, max_iter=10000)
                lasso.fit(X_pert, y_boot)
                
                coefs.append(lasso.coef_)
                intercepts.append(lasso.intercept_)
            
            self.models_.append({
                'coef': np.mean(coefs, axis=0),
                'intercept': np.mean(intercepts)
            })
        
        return self
    
    def predict(self, X):
        preds = []
        for m in self.models_:
            pred = m['intercept'] + X @ m['coef']
            preds.append(pred)
        return np.mean(preds, axis=0)


class AdaptiveStochasticLasso(BaseEstimator, RegressorMixin):
    """
    Adaptive weights + Stochastic Lasso
    ====================================
    Uses adaptive lasso weights based on initial OLS estimates.
    More aggressive feature selection.
    """
    
    def __init__(self, n_samples=100, noise_std=0.03, alpha=0.001, 
                 gamma=1.0, random_state=None):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.alpha = alpha
        self.gamma = gamma  # Adaptive weight power
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n, p = X.shape
        
        # Initial OLS for adaptive weights
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression().fit(X, y)
        weights = 1.0 / (np.abs(ols.coef_) + 1e-6) ** self.gamma
        
        # Weighted features
        X_weighted = X / weights
        
        coefs = []
        intercepts = []
        
        for _ in range(self.n_samples):
            noise = np.random.normal(0, self.noise_std, X_weighted.shape)
            X_pert = X_weighted + noise
            
            lasso = Lasso(alpha=self.alpha, max_iter=10000)
            lasso.fit(X_pert, y)
            
            # Transform back
            coefs.append(lasso.coef_ / weights)
            intercepts.append(lasso.intercept_)
        
        self.coef_ = np.mean(coefs, axis=0)
        self.intercept_ = np.mean(intercepts)
        
        return self
    
    def predict(self, X):
        return self.intercept_ + X @ self.coef_