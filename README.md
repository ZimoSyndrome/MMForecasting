# Multi-Model Market Forecasting & Risk Engine

A quantitative research project comparing three forecasting paradigms for equity returns, with an emphasis on statistical validity, out-of-sample evaluation, and the limits of predictability in financial markets.

- **ARIMA-GARCH** (Econometric) — Linear mean dynamics with conditional heteroskedasticity
- **XGBoost** (Machine Learning) — Non-linear feature interactions under tabular structure
- **LSTM** (Deep Learning) — Sequence-based models for temporal dependence

## Quick Start

> **Note:** This project is designed for research and reproducibility, not real-time trading.
> Python 3.10+ is recommended.

```bash
# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/forecasting_analysis.ipynb
```

## Project Structure

```
MMForecasting/
├── .venv/                          # Python virtual environment
├── notebooks/
│   └── forecasting_analysis.ipynb  # Main analysis notebook
├── requirements.txt                # Dependencies
└── README.md
```

## Key Features

- **Structured ARIMA–GARCH diagnostic pipeline**
  - Stationarity testing (ADF)
  - ARIMA order selection via AIC/BIC
  - ARCH detection and GARCH distribution comparison

- **Strict walk-forward evaluation**
  - Expanding-window retraining
  - One-step-ahead forecasts
  - No look-ahead bias or data leakage

- **Model comparison across paradigms**
  - Error metrics: MSE, MAE
  - Directional accuracy
  - Forecast stability analysis

- **Research-grade implementation**
  - Modular, reproducible code
  - Designed for extensibility and experimentation

## Configuration

These parameters control the empirical setup and are intentionally exposed to encourage experimentation.

Edit variables at the top of the notebook:

```python
TICKER = "SPY"              # Asset to analyze
START_DATE = "2018-01-01"   # Data start
END_DATE = "2024-12-31"     # Data end
TEST_SIZE = 56              # Backtest days
```

## Important Notes on Interpretation

- This project does **not** claim persistent predictability of equity returns.
- Residual dependence may remain due to regime shifts, structural breaks, or long-memory effects.
- Model performance is evaluated statistically, not economically (no transaction costs or trading rules).
- Results should be interpreted as **model comparison**, not trading advice.

## License

MIT
