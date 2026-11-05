#!/usr/bin/env python
# coding: utf-8

# In[162]:


from main_script_ready import (load_finance_data,
    time_split_relocate,
    merging_data,
    daily_returns,
    pairing,
    ols_loop,
    stats_significance,
    mean_std_2025,
    unseen_residual,
    residuals_zscores_features,
    trade_signals,
    time_series_features,
    create_sequences,
    prepare_training_data, 
    initialise_lstm,
    test_evaluation,
    lstm_trade_signals,
    prepare_forecast_df,
    evaluate_strategy,
    compare_strategies
)
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from itertools import combinations
import statsmodels.api as sm  
import warnings
from fastapi import FastAPI, HTTPException
import uvicorn
import inspect

warnings.filterwarnings("ignore")


# In[163]:


def load_finance_data(ticker1, ticker2):

    # Download both stocks
    df = yf.download(
        [ticker1, ticker2],
        start="2020-01-01",
        end="2026-11-01"
    )

    # Reset index
    df = df.reset_index()

    # Add date attributes
    df = time_split_relocate(df)

    # Convert MultiIndex columns into original naming convention
    processed_data = pd.DataFrame()
    
    for col in df.columns:

        if isinstance(col, tuple):
            column_name = "-".join(
                str(x) for x in col if x not in ("", None)
            )
        else:
            column_name = str(col)
    
        column_name = (
            column_name
            .replace("Month-", "Month")
            .replace("Year-", "Year")
            .replace("Quarter-", "Quarter")
        )

        processed_data[column_name] = df[col]

    # Set Date as index, retaining the original structure
    processed_data.index = pd.to_datetime(processed_data["Date"])
    processed_data = processed_data.drop(columns=["Date"])

    return processed_data

price_data = load_finance_data('AMZN','TSLA')
price_data


# In[165]:


def run_pair_trading_model(ticker1, ticker2):

    # 1. Download and prepare stock data
    masterdata = load_finance_data(ticker1, ticker2)

    # 2. Split data into training and testing periods
    train_data = masterdata[
        masterdata["Year"] <= 2025
    ].copy()
    
    test_data = masterdata[
        masterdata["Year"] == 2026
    ].copy()
    
    # 5. Run OLS and cointegration tests
    ols_results = ols_loop(train_data, ticker1, ticker2)

    residuals_forward = ols_results["forward"].resid
    residuals_backward = ols_results["backward"].resid

    coin_test = stats_significance(residuals_forward,residuals_backward)
    return coin_test


# In[166]:


app = FastAPI(title="Pair Trading API", version="1.0")

@app.get("/")
def health_check():
    return {"status": "online", "message": "API is running smoothly."}
    


# In[167]:


@app.get("/predict")
def get_prediction(ticker1: str, ticker2: str):
    try:
        # Your function runs using whatever tickers the user types into the URL
        result = run_pair_trading_model(ticker1, ticker2)
        
        return {
            "status": "success",
            "inputs": {
                "ticker1": ticker1,
                "ticker2": ticker2
            },
            "result": result  # Matches the 'result' variable above
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during inference: {str(e)}")


# In[ ]:


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# In[ ]:




