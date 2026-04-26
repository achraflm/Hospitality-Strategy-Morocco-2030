# 🇲🇦 Hospitality Strategy: Morocco 2030

## 📌 Project Overview
This project aims to analyze and forecast the development of Morocco’s tourism and hospitality sector in the context of the FIFA World Cup 2030. The goal is to support strategic decision-making regarding hotel investments by leveraging data science and time series modeling.

## 🎯 Objectives
- Predict the evolution of tourism demand in Morocco (arrivals, overnight stays)
- Estimate key hospitality indicators such as ADR (Average Daily Rate) and RevPAR
- Analyze seasonality and customer behavior patterns
- Evaluate the potential profitability of hotel investments by 2030
- Simulate the impact of major events (e.g., World Cup 2030)

## 🧠 Methodology

### 1. Data Sources
The project integrates multiple datasets:
- 📊 Moroccan tourism statistics (arrivals, nights, revenues)
- 🏨 Hotel booking dataset (customer behavior, ADR, cancellations)
- 🌍 Macroeconomic indicators (GDP, inflation, investment)
- 🎉 Event data (World Cup, seasonal effects)

### 2. Data Processing
- Extraction of data from PDF reports (automated pipeline)
- Cleaning and normalization of datasets
- Conversion into time series format (monthly granularity)
- Feature engineering (seasonality, lag features, demand indicators)

### 3. Modeling Approach
- Regression models to estimate ADR in the Moroccan context
- Time series forecasting (ARIMA, Prophet, ML models)
- Scenario simulation (with and without major events)
- Final predictive model combining all features

### 4. Key Metrics
- Tourist arrivals
- Overnight stays (nuitées)
- Estimated ADR (Average Daily Rate)
- RevPAR (Revenue per Available Room)
- Investment score (custom indicator)

## 📈 Expected Outcomes
- Forecast of tourism demand up to 2030
- Estimation of hotel profitability trends
- Data-driven insights on optimal investment timing
- Evaluation of World Cup 2030 impact on the hospitality sector

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Time Series Models (ARIMA, Prophet)
- Machine Learning (XGBoost, ANN)
- Data Visualization (Matplotlib, Seaborn)

## 📂 Project Structure
