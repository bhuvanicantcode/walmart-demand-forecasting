# 📦 Walmart Demand Forecasting & Inventory Optimization

An interactive Supply Chain Management dashboard built with Python and Streamlit.

## 🔍 What it does
- Forecasts weekly department-level demand using **SARIMA**
- Derives optimal inventory policy using **EOQ + Safety Stock**
- Quantifies cost of under/over-stocking at each service level
- Lets users stress-test assumptions interactively

## 🛠️ Tech Stack
Python | pandas | statsmodels | matplotlib | Streamlit

## 📊 Dataset
[Walmart Store Sales Forecasting — Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)  
Download `train.csv` and place it in a folder called `walmart data/`

## 🚀 How to Run
```bash
pip install streamlit pandas numpy matplotlib statsmodels
streamlit run walmart_demandforecast.py
```

## 👤 Author
Pallapolu Bhuvan Chandra — BITS Pilani
