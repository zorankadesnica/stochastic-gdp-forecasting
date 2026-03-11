"""
Analysis Script v3: Beat Lasso
==============================
Target: RMSE < 0.0329
"""

from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

from src.data_loader import FREDDataLoader

# Copy stochastic_opt_v3.py to src/ folder first!
from src.stochastic_opt_v3 import (
    StochasticLasso,
    StochasticElasticNet,
    CVStochasticLasso,
    EnsembleStochasticRegressor,
    PolyStochasticLasso,
    BaggingStochasticLasso,
    AdaptiveStochasticLasso
)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

# =============================================================================
# DATA
# =============================================================================
print("="*70)
print("STOCHASTIC OPTIMIZATION v3 - BEAT LASSO")
print("="*70)

api_key = os.getenv('FRED_API_KEY')
loader = FREDDataLoader(api_key=api_key)

print("\n[1] Loading data...")

gdp_rs = loader.fred.get_series('CLVMNACNSAB1GQRS', observation_start='2000-01-01')
house_prices = loader.fred.get_series('QRSR628BIS', observation_start='2000-01-01')
gdp_de = loader.fred.get_series('CLVMNACSCAB1GQDE', observation_start='2000-01-01')
oil_daily = loader.fred.get_series('DCOILBRENTEU', observation_start='2000-01-01')
oil_quarterly = oil_daily.resample('QS').mean()

gdp_rs_growth = gdp_rs.pct_change().dropna()
hp_growth = house_prices.pct_change().dropna()
gdp_de_growth = gdp_de.pct_change().dropna()
oil_growth = oil_quarterly.pct_change().dropna()

df = pd.DataFrame({
    'gdp_rs': gdp_rs_growth,
    'gdp_rs_lag1': gdp_rs_growth.shift(1),
    'gdp_rs_lag2': gdp_rs_growth.shift(2),
    'gdp_rs_lag3': gdp_rs_growth.shift(3),
    'hp_lag1': hp_growth.shift(1),
    'gdp_de_lag1': gdp_de_growth.shift(1),
    'oil_lag1': oil_growth.shift(1),
}).dropna()

print(f"    Dataset: {len(df)} observations")

feature_cols = ['gdp_rs_lag1', 'gdp_rs_lag2', 'gdp_rs_lag3', 
                'hp_lag1', 'gdp_de_lag1', 'oil_lag1']

y_full = df['gdp_rs'].values
X_full = df[feature_cols].values

# Normalize
X_mean, X_std = X_full.mean(axis=0), X_full.std(axis=0)
X_norm = (X_full - X_mean) / X_std

# =============================================================================
# MODELS
# =============================================================================
print("\n[2] Running rolling evaluation...")

min_train = 40
n_total = len(y_full)

models_config = {
    'Lasso': lambda: Lasso(alpha=0.001),
    'Ridge': lambda: Ridge(alpha=1.0),
    'OLS': lambda: LinearRegression(),
    'Stoch Lasso': lambda: StochasticLasso(n_samples=80, noise_std=0.03, alpha=0.001, random_state=42),
    'Stoch ElasticNet': lambda: StochasticElasticNet(n_samples=80, noise_std=0.03, alpha=0.001, l1_ratio=0.5, random_state=42),
    'CV Stoch Lasso': lambda: CVStochasticLasso(n_samples=40, random_state=42),
    'Ensemble Stoch': lambda: EnsembleStochasticRegressor(n_samples=40, random_state=42),
    'Poly Stoch Lasso': lambda: PolyStochasticLasso(degree=2, n_samples=60, noise_std=0.02, alpha=0.003, random_state=42),
    'Bagging Stoch': lambda: BaggingStochasticLasso(n_bags=20, n_samples=25, noise_std=0.03, alpha=0.001, random_state=42),
    'Adaptive Stoch': lambda: AdaptiveStochasticLasso(n_samples=80, noise_std=0.03, alpha=0.001, gamma=1.0, random_state=42),
}

predictions = {name: [] for name in models_config}
actuals = []

print("    Progress: ", end="")

for t in range(min_train, n_total):
    if t % 10 == 0:
        print(".", end="", flush=True)
    
    X_train, y_train = X_norm[:t], y_full[:t]
    X_test = X_norm[t:t+1]
    
    actuals.append(y_full[t])
    
    for name, model_fn in models_config.items():
        try:
            model = model_fn()
            model.fit(X_train, y_train)
            predictions[name].append(model.predict(X_test)[0])
        except Exception as e:
            predictions[name].append(np.nan)

print(" Done!")

actuals = np.array(actuals)

# =============================================================================
# RESULTS
# =============================================================================
print("\n[3] Results:")
print("\n" + "="*70)
print("OUT-OF-SAMPLE FORECAST COMPARISON")
print("="*70 + "\n")

results = []
for name, preds in predictions.items():
    preds = np.array(preds)
    mask = ~np.isnan(preds)
    if mask.sum() > 0:
        results.append({
            'Model': name,
            'RMSE': rmse(actuals[mask], preds[mask]),
            'R²': r2_score(actuals[mask], preds[mask]),
            'Dir.Acc': directional_accuracy(actuals[mask], preds[mask])
        })

results_df = pd.DataFrame(results).set_index('Model').sort_values('RMSE')

print(results_df.round(4).to_string())

print("\n" + "-"*70)
print("RANKING (★ = stochastic method):")
print("-"*70)

lasso_rmse = results_df.loc['Lasso', 'RMSE']

for i, (name, row) in enumerate(results_df.iterrows(), 1):
    marker = "★" if name not in ['Lasso', 'Ridge', 'OLS'] else " "
    beat = "✓" if row['RMSE'] < lasso_rmse and name != 'Lasso' else " "
    print(f"  {i:2d}. {marker} {name:18s}  RMSE={row['RMSE']:.4f}  R²={row['R²']:.3f}  {beat}")

# Check if any stochastic method beat Lasso
stoch_methods = [n for n in results_df.index if n not in ['Lasso', 'Ridge', 'OLS']]
best_stoch = results_df.loc[stoch_methods, 'RMSE'].idxmin()
best_stoch_rmse = results_df.loc[best_stoch, 'RMSE']

print("\n" + "="*70)
if best_stoch_rmse < lasso_rmse:
    improvement = (lasso_rmse - best_stoch_rmse) / lasso_rmse * 100
    print(f"🎉 {best_stoch} BEATS LASSO!")
    print(f"   Improvement: {improvement:.2f}%")
else:
    gap = (best_stoch_rmse - lasso_rmse) / lasso_rmse * 100
    print(f"Best stochastic: {best_stoch} (RMSE={best_stoch_rmse:.4f})")
    print(f"Gap to Lasso: {gap:.2f}%")
print("="*70)

# Statistical significance
best_preds = np.array(predictions[best_stoch])
lasso_preds = np.array(predictions['Lasso'])
mask = ~np.isnan(best_preds)

best_errors = (actuals[mask] - best_preds[mask]) ** 2
lasso_errors = (actuals[mask] - lasso_preds[mask]) ** 2

d = lasso_errors - best_errors  # Positive = stochastic better
t_stat, p_value = stats.ttest_1samp(d, 0)

print(f"\nDiebold-Mariano: {best_stoch} vs Lasso")
print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
if d.mean() > 0:
    print(f"  → {best_stoch} better by {d.mean():.6f} avg squared error")
else:
    print(f"  → Lasso better")

# Save
results_df.to_csv('results_v3.csv')
print("\nSaved: results_v3.csv")