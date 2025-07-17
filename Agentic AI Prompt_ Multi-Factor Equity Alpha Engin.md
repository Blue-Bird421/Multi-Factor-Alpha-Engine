<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Agentic AI Prompt: Multi-Factor Equity Alpha Engine Project (Resume-Ready)

### Objective

**Build a production-quality Python quant research project** that reflects a "Multi-Factor Equity Alpha Engine" as described below. Every deliverable must align tightly with the priorities of quant finance recruiters: clear engineering, academic grounding, modularity, full reproducibility, and strong automation. Do **not** include features, boilerplate, or datasets beyond this outline.

### 1. Project Scope and Elevator Pitch

Construct a reproducible Python framework that ingests U.S. equity data, engineers 25+ academic factors (value, size, momentum, quality, liquidity, etc.), trains Ridge, XGBoost, and feed-forward Neural Network models to forecast 1-month excess stock returns, aggregates signals, builds a Kelly-optimized, dollar-neutral long-short portfolio, and backtests it against the S\&P 1500 (or S\&P 500) from 2005–2024. Package everything for *one-command reproducibility* and research transparency.

### 2. System Design \& Technical Stack

| Layer | Technology/Library |
| :-- | :-- |
| Data Ingestion | `yfinance`, `pandas` |
| Feature Engg | `numpy`, `pandas`, `talib` |
| Modelling | `scikit-learn` (Ridge), `xgboost`, `tensorflow/keras` (FFNN) |
| Portfolio | `PyPortfolioOpt`, custom Kelly sizing |
| Backtesting | Custom logic, integrated with `quantstats` |
| Visualization | `matplotlib`, `seaborn`, `quantstats` |
| Packaging | Modular `alpha_engine/` src folder, `config.yaml`, use `poetry` or `pipenv` for env management |

### 3. Functional Requirements

#### 3.1. Data Pipeline

- Download daily OHLCV data for the S\&P 1500 (or Russell 3000 as an option), 2000–2024.
- Clean for splits, mergers, delistings, and fill missing values.
- Save all processed data in Parquet format for efficient reloads.


#### 3.2. Feature Engineering (≥25 Factors)

Include academic and practitioner factors:

- Value (book-to-market, earnings yield)
- Momentum (12-1M return, short-term reversal)
- Size (market cap)
- Quality (ROE, profit margin, accruals)
- Liquidity (turnover, Amihud illiquidity)
- Volatility (21-day std)
- Carry/term structure (for extension)
- Ratio and z-score transforms

Store factor definitions/configuration in `features.yaml`.

#### 3.3. Modeling Layer

Implement:

- Ridge regression (scikit-learn)
- Gradient boosting (XGBoost)
- Feed-forward neural network (Tensorflow/Keras)

Standardize inputs; outputs are per-stock, per-month expected returns.

Single command per model:

```bash
python train.py --model ridge
python train.py --model xgb
python train.py --model nn
```


#### 3.4. Portfolio Construction

- Blend model signals equally or by IR-weighting.
- Use dollar-neutral, long-short sizing.
- Sizing via Kelly (formula: \$ w = f^* \Sigma^{-1} \mu \$, half-Kelly with rolling 6M covariances).
- Constraints: net exposure zero, max 1% per ticker, 10% per sector.


#### 3.5. Backtest \& Analytics

- Walk-forward backtest 2005–2024.
- Compute and report CAGR, Sharpe, max drawdown, and hit rate vs. S\&P 1500.
- Output equity curve, rolling Sharpe, and monthly heatmap.
- Generate QuantStats HTML tear-sheet and clear research outputs.


### 4. Repository Structure

```
alpha-engine/
├── README.md          # Project pitch, install, and usage
├── config.yaml        # Only user-editable file (universe, dates, paths)
├── data/
│   └── raw/           # Parquet daily OHLCV
├── notebooks/         # EDA and end-to-end demo
├── alpha_engine/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── portfolio.py
│   └── backtest.py
└── reports/
    ├── tearsheet.html
    └── ...
```


### 5. Usage Example

Provide a single script for full reproducibility:

```bash
python run_pipeline.py --start 2005-01-01 --end 2024-06-30 \
                       --universe sp1500 --capital 1e6
```

Outputs should include:

- `reports/tearsheet.html`
- `plots/equity_curve.png`
- `results/weights_latest.csv`


### 6. Code Quality \& Transparency

- All functions must have docstrings.
- Use `pytest` for basic unit and integration tests.
- No absolute paths—use only config variables.
- Documentation and code comments should allow easy adaptation by a new analyst.
- README must include installation, config instructions, one-line demo, and an animated GIF of equity curve (optional).


### 7. Reporting and Metrics

- Clearly compare portfolio performance to the S\&P 1500:
    - If 2005–2024: Portfolio: 13% CAGR, 1.3 Sharpe; S\&P 1500: 7% CAGR, 0.6 Sharpe
- Include Kelly growth curve, performance table, and visual benchmarks.


### 8. Post-Completion (Optional)

- Suggest stretch goals as pinned `todo` in repo: GPU inference, alt data, execution model, Streamlit dashboard.


### 9. Exclusions

- **Do not** add asset classes (crypto, FX, international equities).
- **Do not** add features, pipelines, or dashboards not described above.
- Keep dependencies minimal, modern, and industry-standard.


### 10. Naming \& GitHub

- Repo name: `multi-factor-equity-alpha-engine`
- Use tags: `#python #quantitative-finance #machinelearning`
- MIT license, clean repo badges, clear modular structure.

**End of prompt.**
This project must deliver ALL elements above to faithfully reflect the original resume/pitch and ensure recruiter impact.

