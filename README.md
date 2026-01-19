# Multi-Model Market Forecasting & Risk Engine

A rigorous quantitative research project comparing three forecasting paradigms for equity returns:

- **ARIMA-GARCH** (Econometric) — Volatility clustering & fat tails
- **XGBoost** (Machine Learning) — Non-linear feature interactions
- **LSTM** (Deep Learning) — Sequential dependencies

## Quick Start

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

## Features

- **10-step diagnostic pipeline** for ARIMA-GARCH (ADF tests, distribution selection)
- **Walk-forward validation** (no data leakage)
- **Consolidated comparison** with MSE, MAE, Direction Accuracy
- **Production-grade code** ready for extension

## Configuration

Edit variables at the top of the notebook:

```python
TICKER = "SPY"              # Asset to analyze
START_DATE = "2018-01-01"   # Data start
END_DATE = "2024-12-31"     # Data end
TEST_SIZE = 56              # Backtest days
```

## License

MIT
