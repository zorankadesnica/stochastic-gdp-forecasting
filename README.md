````md
# Stochastic Optimization for Robust Economic Forecasting

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Stochastic optimization methods for GDP nowcasting and composite economic index construction under measurement uncertainty.

---

## Overview

This repository studies forecasting when predictor variables are noisy. Instead of treating economic data as perfectly observed, the project models measurement error directly and uses stochastic optimization to estimate forecasting models.

The main empirical result is that stochastic lasso performs on par with strong linear benchmarks such as Lasso, Ridge, and OLS, while clearly outperforming a naive random-walk baseline.

---

## Repository Structure

```text
.
├── main.tex
├── src/
│   ├── data_loader.py
│   ├── stochastic_opt.py
│   ├── stochastic_opt_v2.py
│   ├── stochastic_opt_v3.py
│   └── monte_carlo.py
├── run_analysis.py
├── requirements.txt
```

---

## Implemented Methods

### Stochastic models

* `StochasticLasso`
* `StochasticElasticNet`
* `CVStochasticLasso`
* `EnsembleStochasticRegressor`
* `BaggingStochasticLasso`
* `AdaptiveStochasticLasso`

### Other stochastic methods

* `StochasticRegression`
* `DRORegression`
* `CVaRRegression`
* `TrimmedRegression`
* `WeightedStochasticRegression`

### Core SAA methods

* `SAAOptimizer`
* `SAACompositeModel`

---

## Data

All data are loaded from FRED.

| Variable            | FRED Code          | Frequency         |
| ------------------- | ------------------ | ----------------- |
| Serbia Real GDP     | `CLVMNACNSAB1GQRS` | Quarterly         |
| Serbia House Prices | `QRSR628BIS`       | Quarterly         |
| Germany Real GDP    | `CLVMNACSCAB1GQDE` | Quarterly         |
| Brent Oil Price     | `DCOILBRENTEU`     | Daily → Quarterly |

Final dataset: **99 quarterly observations** from **2002Q3 to 2025Q3**.

---

## Quick Start

### Installation

```bash
git clone https://github.com/username/stochastic-gdp-forecasting.git
cd stochastic-gdp-forecasting
```

### Set API Key

Create a `.env` file:

```env
FRED_API_KEY=your_key_here
```

### Run

```bash
python run_analysis.py
```
---

## Example Usage

```python
from src.data_loader import FREDDataLoader
from src.stochastic_opt_v3 import StochasticLasso

loader = FREDDataLoader(api_key="your_key")
gdp = loader.fred.get_series("CLVMNACNSAB1GQRS")

model = StochasticLasso(n_samples=100, noise_std=0.03, alpha=0.001)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Empirical Summary

### Out-of-sample ranking

```text
1. Lasso               RMSE=0.0329
2. Stochastic Lasso    RMSE=0.0330
3. Ensemble Stochastic RMSE=0.0332
4. Ridge               RMSE=0.0335
5. OLS                 RMSE=0.0336
...
7. Random Walk         RMSE=0.1034
```

### Diebold-Mariano tests

| Comparison                  | p-value | Result                    |
| --------------------------- | ------- | ------------------------- |
| Stoch. Lasso vs Lasso       | 0.452   | No significant difference |
| Stoch. Lasso vs OLS         | 0.734   | No significant difference |
| Stoch. Lasso vs Ridge       | 0.775   | No significant difference |
| Stoch. Lasso vs Random Walk | <0.001  | Stochastic Lasso better   |

### Crisis period analysis

| Period     | Lasso  | Stochastic Lasso |
| ---------- | ------ | ---------------- |
| Pre-COVID  | 0.0192 | 0.0195           |
| COVID      | 0.0523 | 0.0518           |
| Post-COVID | 0.0153 | 0.0157           |

The crisis-period difference is small and should be interpreted cautiously.

---

## Interpretation

This project does not claim that stochastic methods dominate all benchmark models in this dataset.

The main conclusion is:

* stochastic methods are competitive with strong regularized baselines,
* they outperform a naive random-walk model,
* they provide a principled framework for forecasting under measurement uncertainty.

---


---

## License

MIT License. See `LICENSE`.

```
```
