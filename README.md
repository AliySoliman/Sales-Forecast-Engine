# 📈 Sales Forecast Engine

> Ensemble Machine Learning pipeline for retail sales & price forecasting — with a live interactive dashboard and a professional PDF report.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn&logoColor=white)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react&logoColor=white)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🗂️ Project Structure

```
sales-forecast-engine/
│
├── train_forecast.py          # ML training pipeline (run this first)
├── generate_report.py         # PDF report generator
├── sales_dashboard.jsx        # Live React dashboard
├── training_results.json      # Output: model scores, forecasts, features
├── sales_forecast_report.pdf  # Output: professional 5-section PDF report
│
├── data/
│   └── sales_data.xlsx        # Input dataset (Jan 2014 – Dec 2017)
│
└── README.md
```

---

## 🚀 What This Project Does

This project takes a raw retail sales Excel file and produces:

1. **A trained Weighted Ensemble model** (R² = 0.978, MAE = $348) combining Gradient Boosting, Random Forest, and Ridge Regression
2. **A 3-month sales & price forecast** with 80% and 95% prediction intervals via residual bootstrap
3. **A live React dashboard** with real-time training logs, model comparison charts, diagnostics, and an AI analyst chatbot
4. **A professional PDF report** with 5 sections: executive summary, data overview, model performance, forecast tables, and strategic insights

---

## 📊 Model Results

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Ridge (L2) | -0.49 | $37,638 | $88,272 | 18.9% |
| Random Forest | 0.73 | $1,811 | $2,829 | 12.0% |
| Gradient Boosting | 0.81 | $1,491 | $2,417 | — |
| SVR (RBF) | 0.47 | $2,082 | $3,258 | 55.7% |
| **Weighted Ensemble** | **0.978** | **$348** | **$631** | **13.2%** |

### 3-Month Forecast (Q1 2018)

| Month | Forecast | 80% Interval |
|---|---|---|
| January 2018 | $9,329 | $6,215 – $10,548 |
| February 2018 | $7,030 | $5,446 – $8,093 |
| March 2018 | $9,611 | $7,256 – $10,762 |
| **Q1 Total** | **$25,970** | — |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sales-forecast-engine.git
cd sales-forecast-engine

# Install Python dependencies
pip install pandas numpy scikit-learn matplotlib reportlab openpyxl
```

> No virtual environment is strictly required, but recommended:
> ```bash
> python -m venv venv && source venv/bin/activate  # macOS/Linux
> venv\Scripts\activate                             # Windows
> ```

---

## 🏃 Usage

### Step 1 — Train the models

```bash
python train_forecast.py
```

This reads `data/sales_data.xlsx`, engineers 52 features, trains 4 models + ensemble, runs TimeSeriesSplit cross-validation, generates a 3-month forecast, and writes `training_results.json`.

### Step 2 — Generate the PDF report

```bash
python generate_report.py
```

Reads `training_results.json` and `data/sales_data.xlsx`, renders 5 embedded charts, and outputs `sales_forecast_report.pdf`.

### Step 3 — Launch the live dashboard

The `sales_dashboard.jsx` is a React component. Drop it into any React app or open it directly in [Claude.ai Artifacts](https://claude.ai) — it runs the full simulation automatically and connects to the Anthropic API for AI-powered Q&A.

---

## 🧠 Methodology

### Feature Engineering (52 features)
- **Lag features** — previous 1, 2, 3, 6, and 12 months of log-sales
- **Rolling statistics** — 2, 3, and 6-month rolling mean, std, max, min
- **Seasonal harmonics** — sin/cos encoding at 1×, 2×, and 3× annual frequency
- **Trend features** — normalized time index, year-over-year ratio (lag-12 / lag-24)
- **Momentum** — lag-1 minus lag-2
- **Category breakdown** — monthly sales for Furniture, Technology, Office Supplies
- **Region breakdown** — monthly sales for West, East, Central, South
- **Segment breakdown** — Consumer, Corporate, Home Office
- **Interaction terms** — trend × sin/cos seasonality

### Why Log-Transform?
Sales values are heavily right-skewed (mean $217, single-transaction max ~$10.5K). Applying `log1p` before training compresses extremes, makes residuals more Gaussian, and dramatically improves Ridge and SVR fits.

### Cross-Validation Strategy
Standard k-fold leaks future data into training folds on time series. `TimeSeriesSplit` with 6 folds ensures every test fold lies strictly after its training fold — producing honest out-of-sample error estimates.

### Ensemble Weighting
Each of the top 3 models by R² is assigned a weight proportional to `1 / MAE`. This automatically down-weights poorly calibrated models and lets Gradient Boosting and Random Forest dominate while Ridge contributes as a smoother.

### Prediction Intervals
100 Monte Carlo draws from the empirical residual distribution are used to construct 80% and 95% bands — no normality assumption required.

---

## 📁 Output Files

| File | Description |
|---|---|
| `training_results.json` | All model scores, in-sample predictions, 3-month forecast, feature importances, training log |
| `sales_forecast_report.pdf` | 5-section professional report with embedded charts |

---

## 🖥️ Dashboard Features

The React dashboard (`sales_dashboard.jsx`) has 5 tabs:

| Tab | Contents |
|---|---|
| **Overview** | Actual vs Fitted chart for all models (toggleable), R² and MAPE bar charts |
| **Models** | Full scorecard table, feature importance chart, methodology explanations |
| **Forecast** | 12-month forecast area chart with 80%/95% bands, monthly detail table |
| **Diagnostics** | Residuals over time, actual vs predicted scatter, residual stats |
| **AI Insights** | Ask questions about the model via Claude API; suggested questions included |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading and aggregation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | All ML models and CV |
| `matplotlib` | ≥ 3.7 | Chart generation for PDF |
| `reportlab` | ≥ 4.0 | PDF creation |
| `openpyxl` | ≥ 3.1 | Excel file reading |

---

## 📌 Data Format

The input Excel file should contain at minimum these columns:

| Column | Type | Example |
|---|---|---|
| `Order Date` | date | `2017-11-15` |
| `Sales` | float | `452.30` |
| `Profit` | float | `68.40` |
| `Quantity` | int | `3` |
| `Discount` | float | `0.20` |
| `Category` | string | `Technology` |
| `Region` | string | `West` |
| `Segment` | string | `Consumer` |
| `Order ID` | string | `CA-2017-152156` |
| `Customer ID` | string | `CG-12520` |
| `Product ID` | string | `FUR-BO-10001798` |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built with Python, scikit-learn, React, and ReportLab.  
For questions or improvements, open an issue or submit a pull request.
