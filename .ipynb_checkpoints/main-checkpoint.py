#!/usr/bin/env python
# coding: utf-8

# In[20]:


from main_script_ready import (time_split_relocate,
    merging_data,
    daily_returns,
    pairing,
    ols_loop,
    stats_significane,
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
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from itertools import combinations
import statsmodels.api as sm  
import warnings
from fastapi import FastAPI, HTTPException
import uvicorn

warnings.filterwarnings("ignore")


# In[21]:


app = FastAPI(title="Pair Trading API", version="1.0")

@app.get("/")
def health_check():
    return {"status": "online", "message": "API is running smoothly."}
    


# In[22]:


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




