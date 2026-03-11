# Methodology

## Problem Formulation

Minimize expected forecast error under data uncertainty:

```
min_w E_ξ[ L(C(w, X + ξ), GDP) ]
```

## Sample Average Approximation (SAA)

Replace expectation with sample mean using N Monte Carlo samples.

## Bayesian Model

```
GDP_t ~ Normal(α + β·C_t, σ²)
α, β ~ Normal(0, τ²)
σ ~ Half-Cauchy(0, 1)
```

See main.tex for full mathematical formulation.
