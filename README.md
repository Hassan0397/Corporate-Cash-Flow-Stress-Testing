# 🏦 AI-Driven Corporate Cash Flow Stress Testing Platform

## 📌 Overview

Modern multinational enterprises operate in highly volatile environments impacted by inflation, FX fluctuations, interest rate shocks, and macroeconomic uncertainty. Traditional Excel-based financial models lack probabilistic forecasting, automated risk quantification, and scalable executive reporting.

This project is a **production-grade enterprise financial analytics platform** built with Streamlit that transforms raw financial data into:

---


- Architected and deployed a **production-grade financial analytics platform** enabling CFO and treasury teams to perform predictive cash flow forecasting, probabilistic risk modeling, and macroeconomic stress testing within a unified dashboard.

- Engineered a modular **9-page analytics workflow** including Data Loading, Data Cleaning, EDA, Time-Series Forecasting, Monte Carlo Simulation (10K–50K runs), Scenario Modeling, ML Engine, and AI-Generated Executive Reporting.

- Built a **multi-model ensemble forecasting pipeline (Prophet, ARIMA, LSTM)** with automated hyperparameter tuning and confidence interval estimation, improving forecast robustness under volatile financial conditions.

- Designed a high-performance **Monte Carlo risk engine** computing VaR (90/95/99%), CVaR, survival probability, and cash runway projections, enabling quantified downside liquidity assessment.

- Implemented a macroeconomic **stress testing & sensitivity analysis framework**, modeling recession, inflation, FX crisis, and rate shock scenarios with tornado charts and break-even contour analysis.

- Developed ML pipelines for **cash flow regression, default risk classification, and anomaly detection**, integrating advanced feature engineering (lag features, rolling statistics, financial ratios) and model explainability techniques.

- Integrated **Generative AI (GPT-4 / Gemini)** to auto-generate executive summaries, strategic recommendations, and board-ready risk reports directly from quantitative outputs.

- Engineered enterprise-grade UI/UX using Streamlit with session state management, caching optimization, modular architecture, and custom professional design components.


It combines financial domain expertise with advanced data science, machine learning, and Generative AI in a modular, scalable architecture.

---

# 🎯 Business Problem Solved

Corporate finance teams often struggle with:

- Reactive reporting instead of proactive planning  
- Limited macroeconomic scenario modeling  
- Poor visibility into downside liquidity risk  
- Fragmented tools across Excel, BI, and ML notebooks  
- Manual executive reporting  

This platform enables:

- Forecast-driven financial planning  
- Quantified downside risk assessment  
- Cash runway survival probability modeling  
- Board-ready AI-generated reports  
- Data-driven strategic decision making  

---

# 🏗️ System Architecture

## 🔄 Data Flow Pipeline

LOAD → CLEAN → EXPLORE → FORECAST → MONTE CARLO → ML → GEN AI → REPORT


## 🧩 Core Modules

| Module | Purpose |
|--------|----------|
| `app.py` | Enterprise dashboard controller & session management |
| `load_data.py` | Data ingestion & 6-step validation pipeline |
| `clean_data.py` | Financial preprocessing & feature engineering |
| `exploratory_data_analysis.py` | Financial health scoring & analytics |
| `time_series_forecasting.py` | Prophet, ARIMA, LSTM ensemble forecasting |
| `monte_carlo_simulations.py` | 10K+ simulation probabilistic risk engine |
| `scenario_analysis.py` | Macroeconomic stress & what-if modeling |
| `ml_models.py` | Regression, classification, anomaly detection |
| `genai_insights.py` | GPT-4 / Gemini executive reporting |
| `helpers.py` | Utility, statistical & formatting foundation |

---

# 📊 Core Capabilities

## 1️⃣ Financial Intelligence
- Automated financial health scoring  
- Seasonality & correlation analysis  
- Executive-level insights  

## 2️⃣ Multi-Model Forecasting
- Prophet (trend & seasonality)
- ARIMA/SARIMA
- LSTM Neural Networks
- Ensemble predictions with confidence intervals

## 3️⃣ Monte Carlo Risk Engine
- 10,000–50,000 simulations
- Value at Risk (VaR 90/95/99%)
- Conditional VaR (CVaR)
- Cash runway survival curves
- Revenue & cost shock testing

## 4️⃣ Scenario & Sensitivity Analysis
- Recession / Inflation / FX Crisis / Rate Shock modeling
- Tornado charts
- Break-even contour analysis
- Risk-return comparison matrix

## 5️⃣ Machine Learning Engine
- Regression (cash flow forecasting)
- Classification (default risk prediction)
- Anomaly detection (fraud/error detection)
- Feature importance analysis
- Hyperparameter tuning with cross-validation

## 6️⃣ Generative AI Reporting
- GPT-4 & Gemini integration
- Executive summaries
- Strategic recommendations
- Board-ready reports
- Risk narrative explanations

---

# 📈 Business Impact

This platform allows enterprises to:

- Quantify downside liquidity risk  
- Optimize capital allocation  
- Stress test macroeconomic shocks  
- Improve board-level communication  
- Transition from reactive reporting to predictive strategy  

---

# 🛠 Technology Stack

**Frontend:** Streamlit, Plotly, Custom CSS  
**Data Processing:** Pandas, NumPy, SciPy  
**Statistics:** StatsModels  
**Machine Learning:** Scikit-learn, XGBoost, LightGBM, TensorFlow  
**Forecasting:** Prophet, ARIMA  
**Generative AI:** OpenAI GPT-4, Google Gemini  

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ai-corporate-cashflow-platform.git
cd ai-corporate-cashflow-platform

```

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```
## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

 ## Use API Key using PowerShell Command
 ```bash
 $env:OPENAI_API_KEY="your_openai_api_key_here"
$env:GEMINI_API_KEY="your_gemini_api_key_here"
 ```
## ▶️ Run the Application

 ```bash
streamlit run app.py
 ```
