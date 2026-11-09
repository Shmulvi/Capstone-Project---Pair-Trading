#!/usr/bin/env python
# coding: utf-8

# In[38]:


from main_script_ready import (time_split_relocate,
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
    create_sequences
)
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
import io
import base64
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

warnings.filterwarnings("ignore")


# In[34]:


def load_finance_data(ticker1, ticker2):

    # Download both stocks
    df = yf.download(
        [ticker1, ticker2],
        start="2020-01-01",
        end="2026-07-31"
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


# In[42]:


def plot_spread(pair_data, ticker1, ticker2):
    plt.figure(figsize=(12, 5))
    plt.plot(pair_data.index, label="Date")
    plt.plot(pair_data['Residuals'], label="Residuals")
    plt.legend()
    plt.title(f"Spread of Cointegrated Pair {ticker1} & {ticker2}")
    plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close() # Crucial to free memory and prevent leaks on server
    
    return plot_base64


# In[7]:


def run_pair_trading_model(ticker1, ticker2):
    import os
    import psutil
    
    process = psutil.Process(os.getpid())
    masterdata = load_finance_data(ticker1, ticker2)
    print(
        f"Memory before download: "
        f"{process.memory_info().rss / 1024**2:.2f} MB",
        flush=True
    )
    # 1. Download and prepare pair
    print("PREDICT: Data loaded", flush=True)
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
    close_A = train_data[f"Close-{ticker1}"]
    close_B = train_data[f"Close-{ticker2}"]
    
    coin_test = stats_significance(residuals_forward,residuals_backward,close_A,close_B)

    checks = [
        coin_test['ADF_A']['p_value'] < 0.05,
        coin_test['Engle_Granger']['p_value'] < 0.05,
        coin_test['Johansen']['trace_statistic'] > coin_test['Johansen']['critical_value_5pct']
    ]
    print("PREDICT: Cointegration completed", flush=True)
    if sum(checks) >= 2:
        cointegration = True
        assessment_text = f"🟢 Strong Cointegration ({sum(checks)}/3 checks passed). Favorable pair for trading."
        mean, std, model = mean_std_2025(close_A, close_B)
        print("PREDICT: Mean and std calculated", flush=True)
        residuals_zscores_features(
            test_data, close_A, close_B, model, mean, std
        )
        print("PREDICT: Z-scores calculated", flush=True)
        trading_signals = trade_signals(test_data, threshold=1)
        signal_counts = trading_signals.groupby('Signal').size()
        signal_summary = {
            str(signal): int(count)
            for signal, count in signal_counts.items()
            if signal != 'Null'
        }
        plot_str = plot_spread(test_data, ticker1, ticker2)
    elif sum(checks) == 1:
        cointegration = False
        assessment_text = f"🟡 Moderate Support ({sum(checks)}/3 checks passed). Does not meet criteria."
        signal_summary = 'No trading signal required'
    else:
        cointegration = False
        assessment_text = f"🔴 Weak / Inconclusive Cointegration ({sum(checks)}/3 checks passed). Does not meet criteria."
        signal_summary = 'No trading signal required'
    return coin_test, {
        "cointegrated": bool(cointegration),
        "checks_passed": int(sum(checks)),
        "assessment": assessment_text,
        "trading signals": signal_summary,
        "spread plot": plot_str
    }


# In[4]:


app = FastAPI(title="Pair Trading API", version="1.0")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pair Trading Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); width: 400px; text-align: center; }
            h2 { margin-bottom: 20px; color: #007bff; }
            input { width: 90%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; }
            button { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 16px; cursor: pointer; width: 100%; margin-top: 10px; }
            button:hover { background-color: #0056b3; }
            
            /* Nicely formatted results styling */
            #result { margin-top: 20px; text-align: left; display: none; background: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #e9ecef; }
            .result-table { width: 100%; border-collapse: collapse; margin-top: 5px; }
            .result-table th, .result-table td { padding: 8px 10px; border-bottom: 1px solid #dee2e6; font-size: 14px; }
            .result-table th { text-align: left; color: #495057; }
            .result-table td { text-align: right; font-weight: bold; color: #212529; }
            .error-text { color: #dc3545; font-weight: bold; text-align: center; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Pair Trading Model</h2>
            <input type="text" id="ticker1" placeholder="Ticker 1 (e.g., AAPL)">
            <input type="text" id="ticker2" placeholder="Ticker 2 (e.g., MSFT)">
            <button onclick="getPrediction()">Run Prediction</button>
            <div id="result"></div>
        </div>

        <script>
            async function getPrediction() {
                const t1 = document.getElementById('ticker1').value.trim().toUpperCase();
                const t2 = document.getElementById('ticker2').value.trim().toUpperCase();
                const resultDiv = document.getElementById('result');
                
                if (!t1 || !t2) {
                    alert("Please enter both tickers.");
                    return;
                }
        
                resultDiv.style.display = "block";
                resultDiv.innerHTML = "<p style='text-align:center; color:#6c757d;'>Running model calculation...</p>";
        
                try {
                    const response = await fetch(`/predict?ticker1=${t1}&ticker2=${t2}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        let html = `<h4 style="margin:0 0 12px 0; color:#007bff; text-align:center;">Results for ${t1} & ${t2}</h4>`;
                        html += `<table class="result-table">`;
                        
                        // Recursive helper function for nested objects/sub-dictionaries
                        function formatValue(val) {
                            if (val === null || val === undefined) return 'N/A';
                            
                            if (typeof val === 'object') {
                                let subHtml = `<table class="result-table" style="margin: 4px 0; width: 100%; background: #fff; border: 1px solid #dee2e6;">`;
                                for (const [subKey, subVal] of Object.entries(val)) {
                                    subHtml += `<tr><th style="color: #495057; width: 50%; padding: 6px 8px; border-bottom: 1px solid #f1f3f5;">${subKey}</th><td style="text-align: right; padding: 6px 8px; border-bottom: 1px solid #f1f3f5;">${formatValue(subVal)}</td></tr>`;
                                }
                                subHtml += `</table>`;
                                return subHtml;
                            }
                            
                            if (typeof val === 'number' && !Number.isInteger(val)) {
                                return val.toFixed(4);
                            }
                            
                            return val;
                        }
        
                        // Loop through main response dictionary (skip 'assessment' as it's rendered separately below)
                        for (const [key, value] of Object.entries(data)) {
                            if (key === 'assessment' || key === 'spread plot') continue;
                    
                            if (typeof value === 'object' && value !== null) {
                                // Drop nested objects/dictionaries down to a clean full-width row
                                html += `<tr><td colspan="2" style="padding: 12px 8px; border-bottom: 1px solid #dee2e6;">`;
                                html += `<div style="font-weight: 600; color: #343a40; margin-bottom: 8px; font-size: 15px;">${key}:</div>`;
                                html += formatValue(value);
                                html += `</td></tr>`;
                            } else {
                                // Standard side-by-side row for simple values like ticker names or status
                                html += `<tr><th style="width: 35%; padding: 10px 8px; border-bottom: 1px solid #dee2e6;">${key}</th><td style="text-align: right; padding: 10px 8px; border-bottom: 1px solid #dee2e6; color: #212529; font-weight: bold;">${formatValue(value)}</td></tr>`;
                            }
                        }
                        
                        html += `</table>`;
                    
                        // Render the assessment banner passed down directly from Python
                        if (data.assessment) {
                            let color = data.assessment.color;
                            html += `<div class="assessment-box" style="border-left-color: ${color};">`;
                            html += `<strong>Model Assessment:</strong> <span style="color: ${color}; font-weight: bold;">${data.assessment.assessment}</span>`;
                            html += `</div>`;
                        }
                        if (data["spread plot"]) {
                            html += `<div style="margin-top: 15px; text-align: center;">`;
                            html += `<img src="data:image/png;base64,${data["spread plot"]}" alt="Pair Spread Plot" style="width: 100%; border-radius: 4px; border: 1px solid #dee2e6;" />`;
                            html += `</div>`;
                        }
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<p class="error-text">Error: ${data.detail || "Unknown error"}</p>`;
                    }
                } catch (err) {
                    resultDiv.innerHTML = `<p class="error-text">Failed to connect to backend server.</p>`;
                }
            } 
        </script>
        
        <style>
            .result-table { width: 100%; border-collapse: collapse; margin-top: 5px; font-size: 14px; }
            .result-table th { text-align: left; color: #343a40; font-weight: 600; }
            .error-text { color: #dc3545; font-weight: bold; text-align: center; }
            .assessment-box { margin-top: 15px; padding: 10px 12px; background-color: #fff; border-left: 4px solid #ccc; border-radius: 4px; font-size: 14px; text-align: left; }
        </style>
    </body>
    </html>
    """


# In[167]:


@app.get("/predict")
def get_prediction(ticker1: str, ticker2: str):

    print("PREDICT: Request received", flush=True)

    try:
        result, cointegration_check = run_pair_trading_model(ticker1, ticker2)

        print("PREDICT: Model completed", flush=True)

        return {
            "ticker1": ticker1,
            "ticker2": ticker2, 
            "status": "success", 
            "prediction": result,
            "Cointegration Check": cointegration_check
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# In[ ]:


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

