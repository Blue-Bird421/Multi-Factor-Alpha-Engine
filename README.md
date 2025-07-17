# Multi-Factor Equity Alpha Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality Python framework for quantitative equity research, featuring multi-factor modeling, machine learning, and portfolio optimization.

## 🎯 Project Overview

This project implements a comprehensive **Multi-Factor Equity Alpha Engine** that:

- 📊 Ingests U.S. equity data for S&P 1500 universe (2000-2024)
- 🔧 Engineers 35+ academic factors (value, momentum, quality, size, liquidity, volatility, technical)
- 🤖 Trains multiple ML models (Ridge, XGBoost, Neural Networks)
- 📈 Constructs Kelly-optimized, dollar-neutral long-short portfolios
- 📋 Backtests strategies with comprehensive performance analysis
- 📊 Generates professional QuantStats tearsheets and visualizations

**Target Performance**: 13% CAGR, 0.9 Sharpe ratio vs S&P 1500 benchmark (7% CAGR, 0.6 Sharpe)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/multi-factor-equity-alpha-engine.git
cd multi-factor-equity-alpha-engine

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### TA-Lib Installation (Optional but Recommended)

**Windows Users:**
If you encounter TA-Lib installation issues, use a pre-compiled wheel:
1. Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install manually: `pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl`

**macOS/Linux Users:**
```bash
# macOS
brew install ta-lib

# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# Then reinstall the Python package
pip install TA-Lib
```

**Note:** The project includes fallback implementations for TA-Lib functions, so it will work without TA-Lib installed.

### One-Command Demo

Run the complete pipeline with default settings:

```bash
python run_pipeline.py --start 2005-01-01 --end 2024-06-30 --universe sp1500 --capital 1000000
```

### Custom Configuration

```bash
# S&P 500 universe with different period
python run_pipeline.py --start 2010-01-01 --end 2023-12-31 --universe sp500 --capital 500000

# Force data refresh and model retraining
python run_pipeline.py --force-refresh

# Skip training and use existing models
python run_pipeline.py --skip-training
```

## 📁 Project Structure

```
alpha-engine/
├── README.md                  # This file
├── config.yaml               # Main configuration (ONLY file you need to edit)
├── features.yaml             # Feature definitions
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # Main execution script
├── train.py                  # Individual model training
│
├── data/
│   └── raw/                  # Cached Parquet data files
│
├── alpha_engine/             # Core engine modules
│   ├── __init__.py          # Package initialization
│   ├── data.py              # Data ingestion and processing
│   ├── features.py          # Feature engineering (25+ factors)
│   ├── models.py            # ML models (Ridge, XGB, NN)
│   ├── portfolio.py         # Portfolio optimization & Kelly sizing
│   └── backtest.py          # Backtesting and performance analysis
│
├── notebooks/               # Jupyter notebooks for EDA
├── models/                  # Trained model files
├── reports/                 # Generated reports and tearsheets
├── plots/                   # Performance visualizations
├── results/                 # Portfolio weights and metrics
└── tests/                   # Unit tests
```

## ⚙️ Configuration

The `config.yaml` file is the **only file you need to edit** for customization:

```yaml
# Data Configuration
data:
  universe: "sp1500"          # sp500, sp1500, russell3000
  start_date: "2000-01-01"
  end_date: "2024-12-31"

# Backtest Configuration  
backtest:
  start_date: "2005-01-01"
  end_date: "2024-06-30"
  initial_capital: 1000000
  rebalance_freq: "M"         # Monthly rebalancing

# Portfolio Configuration
portfolio:
  max_position_weight: 0.01   # 1% max per stock
  max_sector_weight: 0.10     # 10% max per sector
  kelly_fraction: 0.5         # Half-Kelly sizing
```

## 🔬 Features Engineered

### Value Factors (5)
- Book-to-Market, Earnings Yield, Sales-to-Price, Cash Flow Yield, Dividend Yield

### Momentum Factors (4)  
- 12-1 Month Momentum, 6-1 Month Momentum, Short-term Reversal, Price Trend

### Quality Factors (6)
- ROE, ROA, Profit Margin, Debt-to-Equity, Current Ratio, Accruals

### Size Factors (2)
- Market Cap, Log Market Cap

### Liquidity Factors (4)
- Turnover, Amihud Illiquidity, Bid-Ask Spread, Volume Trend  

### Volatility Factors (4)
- Realized Volatility (21d, 63d, 126d), Volatility of Volatility

### Technical Factors (6)
- RSI, MACD Signal, Bollinger Position, Price-to-SMA ratios, Williams %R

### Risk Factors (4)
- Market Beta, Idiosyncratic Volatility, Return Skewness, Return Kurtosis

**Total: 35+ factors** with z-score transformations and winsorization

## 🤖 Machine Learning Models

### 1. Ridge Regression
- Linear model with L2 regularization
- Interpretable coefficients
- Fast training and prediction

### 2. XGBoost
- Gradient boosting with tree ensembles  
- Handles non-linear relationships
- Built-in feature importance

### 3. Neural Network
- Feed-forward architecture with dropout
- Hidden layers: [64, 32] with ReLU activation
- Early stopping and learning rate scheduling

### Ensemble
- Equal or Information Ratio weighted blending
- Cross-validated performance estimation
- Robust to individual model failures

## 📈 Portfolio Construction

### Kelly Criterion Optimization
- Kelly formula: `w* = kelly_fraction * Σ^(-1) * μ`
- Half-Kelly sizing for risk management
- Rolling covariance estimation (6 months)

### Risk Constraints
- Maximum 1% position size per stock
- Maximum 10% exposure per sector
- Dollar-neutral long-short construction
- Sector-based diversification

### Transaction Costs
- Configurable transaction costs
- Minimum trade size filters
- Realistic rebalancing assumptions

## 📊 Performance Analysis

### Key Metrics
- Total Return, CAGR, Sharpe Ratio
- Maximum Drawdown, Hit Rate
- Information Ratio vs Benchmark
- Tracking Error, Beta

### Visualizations
- Equity curve with drawdowns
- Rolling performance metrics
- Monthly returns heatmap
- QuantStats HTML tearsheet

### Reports Generated
- `reports/tearsheet.html` - Comprehensive QuantStats analysis
- `reports/performance_summary.txt` - Key metrics summary
- `plots/equity_curve.png` - Portfolio value over time
- `results/weights_latest.csv` - Latest portfolio positions

## 🔧 Individual Model Training

Train specific models independently:

```bash
# Train Ridge regression
python train.py --model ridge

# Train XGBoost
python train.py --model xgboost  

# Train Neural Network
python train.py --model neural_network

# Train full ensemble
python train.py --model ensemble
```

## 🧪 Testing

Run the test suite:

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=alpha_engine --cov-report=html
```

## 📊 Expected Results

Based on academic factor research, the engine targets:

| Metric | Alpha Engine | S&P 1500 Benchmark |
|--------|-------------|-------------------|
| CAGR | 13% | 7% |
| Sharpe Ratio | 0.9 | 0.6 |

*Results may vary based on market conditions and implementation details*

## 🔄 Workflow

1. **Data Pipeline**: Download → Clean → Process → Cache
2. **Feature Engineering**: Calculate factors → Transform → Normalize  
3. **Model Training**: Train ensemble → Validate → Save models
4. **Portfolio Construction**: Generate signals → Optimize → Apply constraints
5. **Backtesting**: Walk-forward test → Calculate returns → Analyze performance
6. **Reporting**: Generate tearsheet → Create plots → Save results

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Data** | `yfinance`, `pandas` |
| **Features** | `numpy`, `pandas`, `talib` |
| **ML** | `scikit-learn`, `xgboost`, `tensorflow` |
| **Portfolio** | `PyPortfolioOpt`, custom Kelly |
| **Backtest** | Custom engine + `quantstats` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Config** | `PyYAML` |

## 📋 Future Enhancements

- [ ] GPU acceleration for neural networks
- [ ] Alternative data integration (sentiment, satellite)
- [ ] Real-time execution simulation
- [ ] Streamlit web dashboard
- [ ] Options overlay strategies
- [ ] ESG factor integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

