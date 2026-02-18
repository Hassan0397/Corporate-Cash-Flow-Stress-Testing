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

# 📄 Page-by-Page Breakdown

---

## 1️⃣ Data Load Page

**[Data Loading Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Loading%20Preview%201.png)**

**[Data Loading Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Loading%20Preview%202.png)**

**[Data Loading Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Loading%20Preview%203.png)**

**[Data Loading Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Loading%20Preview%204.png)**

**Purpose:** Enterprise-grade financial data ingestion & validation  

- 6-step validation pipeline (data types, date integrity, missing values, outliers)  
- Data quality scoring dashboard  
- Statistical summaries & memory optimization tracking  
- Financial schema validation (Revenue, Capex, Debt, Cash, FCF, FX, Inflation)

---

## 2️⃣ Data Cleaning Page  

**[Data Cleaning Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Cleaning%20Preview%201.png)**

**[Data Cleaning Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Cleaning%20Preview%202.png)**

**[Data Cleaning Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Cleaning%20Preview%203.png)**

**[Data Cleaning Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Data%20Cleaning%20Preview%204.png)**

**Purpose:** Financial preprocessing & intelligent feature engineering  

- Domain-aware imputation (median, mean, forward/backward fill)  
- Outlier treatment (Winsorization / flagging)  
- Lag feature engineering (1–12 months)  
- Rolling statistics & financial ratio engineering  
- Automated quality scoring (completeness, consistency, validity)

---

## 3️⃣ Exploratory Data Analysis (EDA) Page  

**[EDA Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%201.png)**

**[EDA Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%202.png)**

**[EDA Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%203.png)**

**[EDA Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%204.png)**

**[EDA Preview 5](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%205.png)**

**[EDA Preview 6](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%206.png)**

**[EDA Preview 7](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%207.png)**

**[EDA Preview 8](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%208.png)**

**[EDA Preview 9](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/EDA%20Preview%209.png)**

**Purpose:** Executive-level financial intelligence  

- Financial health score (Profitability, Liquidity, Stability, Growth)  
- Correlation heatmaps & relationship explorer  
- Seasonality detection & YoY trend analysis  
- Automated strategic insights & recommendations

---

## 4️⃣ Time Series Forecasting Page 

**[Time Series Forecasting Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%201.png)**

**[Time Series Forecasting Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%202.png)**

**[Time Series Forecasting Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%203.png)**

**[Time Series Forecasting Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%204.png)**

**[Time Series Forecasting Preview 5](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%205.png)**

**[Time Series Forecasting Preview 6](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%206.png)**

**[Time Series Forecasting Preview 7](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%207.png)**

**[Time Series Forecasting Preview 8](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%208.png)**

**[Time Series Forecasting Preview 9](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%209.png)**

**[Time Series Forecasting Preview 10](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Time%20Series%20Forecasting%20Preview%2010.png)**

**Purpose:** Predictive cash flow modeling  

- Prophet (trend + seasonality + changepoints)  
- ARIMA/SARIMA with stationarity testing  
- LSTM deep learning sequence modeling  
- Ensemble forecasting with uncertainty bands  
- Performance metrics (MAE, RMSE, MAPE, R²)

---

## 5️⃣ Monte Carlo Simulation Page  

**[Monte Carlo Simulations Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Monte%20Carlo%20Simulations%20Preview%201.png)**

**[Monte Carlo Simulations Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Monte%20Carlo%20Simulations%20Preview%202.png)**

**[Monte Carlo Simulations Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Monte%20Carlo%20Simulations%20Preview%203.png)**

**[Monte Carlo Simulations Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Monte%20Carlo%20Simulations%20Preview%204.png)**

**[Monte Carlo Simulations Preview 5](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Monte%20Carlo%20Simulations%20Preview%205.png)**



**Purpose:** Probabilistic risk quantification  

- 10,000–50,000 simulations  
- VaR & Conditional VaR calculation  
- Cash runway survival probability  
- Burn rate forecasting  
- Revenue & cost shock stress testing  

---

## 6️⃣ Scenario Analysis Page  

**[Scenario Analysis Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%201.png)**

**[Scenario Analysis Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%202.png)**

**[Scenario Analysis Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%203.png)**

**[Scenario Analysis Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%204.png)**

**[Scenario Analysis Preview 5](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%205.png)**

**[Scenario Analysis Preview 6](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%206.png)**

**[Scenario Analysis Preview 7](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%207.png)**

**[Scenario Analysis Preview 8](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%208.png)**

**[Scenario Analysis Preview 9](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%209.png)**

**[Scenario Analysis Preview 10](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%2010.png)**

**[Scenario Analysis Preview 11](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%2011.png)**

**[Scenario Analysis Preview 12](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Scenario%20Analysis%20Preview%2012.png)**

**Purpose:** Macroeconomic stress modeling  

- Prebuilt recession, inflation, FX, and interest rate shock scenarios  
- Tornado sensitivity charts  
- Break-even contour modeling  
- Risk-return comparison dashboards  

---

## 7️⃣ Machine Learning Page  

**[Machine learning Model Preview 1](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Machine%20learning%20Model%20Preview%201.png)**

**[Machine learning Model Preview 2](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Machine%20learning%20Model%20Preview%202.png)**

**[Machine learning Model Preview 3](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Machine%20learning%20Model%20Preview%203.png)**

**[Machine learning Model Preview 4](https://github.com/Hassan0397/Corporate-Cash-Flow-Stress-Testing/blob/main/Machine%20learning%20Model%20Preview%204.png)**

**Purpose:** Advanced predictive modeling  

- Regression (cash flow forecasting)  
- Classification (default risk prediction)  
- Anomaly detection (Isolation Forest, One-Class SVM)  
- Feature importance & hyperparameter tuning  

---

## 8️⃣ GenAI Insights Page  
**Purpose:** AI-powered executive reporting  

- GPT-4 / Gemini integration  
- Board-ready executive summaries  
- Risk narrative explanation  
- Strategic action recommendations  
- Customizable report configuration  

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
