import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from itertools import combinations
import statsmodels.api as sm  
import warnings
def load_finance_data(ticker1,ticker2):
    df = yf.download(
        [ticker1,ticker2],
        start="2020-01-01",
        end="2026-01-01",
        auto_adjust=False
    )
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

        processed_data[column_name] = df[col]

    # Set Date as index, retaining the original structure
    processed_data.index = pd.to_datetime(processed_data["Date"])
    processed_data = processed_data.drop(columns=["Date"])

    return processed_data
def time_split_relocate(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Quarter'] = dataframe['Date'].dt.quarter
    dataframe = dataframe.iloc[:,[0,11,12,13,1,2,3,4,5,6,7,8,9,10]]

    return dataframe
def merging_data(df1,df2,df3,df4,df5,df6,df7,df8):
    merged_data = pd.merge(df1, df2, on=('Date','Month','Year','Quarter')).merge(df3, on=('Date','Month','Year','Quarter')).merge(df4, on=('Date','Month','Year','Quarter')).merge(df5, on=('Date','Month','Year','Quarter')).merge(df6, on=('Date','Month','Year','Quarter')).merge(df7, on=('Date','Month','Year','Quarter')).merge(df8, on=('Date','Month','Year','Quarter'))
    merged_data.index = merged_data['Date']
    merged_data = merged_data.drop(columns=['Date'])

    return merged_data
def daily_returns(dataframe, dataframe_sp):
    col = []
    col = dataframe.columns.tolist()
    col = [item for item in col if item not in ['Month','Year','Quarter']]
    
    ticker_list = []
    for x in col:
        ticker_list.append(str(x).split("-")[1])

    unique_tickers = list(dict.fromkeys(ticker_list))
                          
    for i in unique_tickers:
        dataframe['Daily Returns '+str(i)] = round(dataframe[f'Close-{i}'].pct_change() * 100, 2)

    dataframe['Daily Returns S&P500'] = round(dataframe_sp['Close','^GSPC'].pct_change() * 100, 2)

    return dataframe
def pairing(dataframe):

    pairings = []
    # Removing the daily returns so it's not included in the pairings
    returns = dataframe.iloc[:,43:53]
    
    dataframe = dataframe.drop(columns=dataframe.iloc[:,43:53].columns.tolist())
    
    # Keep first 4 date columns
    date_df = dataframe.iloc[:, :3]

    # Remove date columns
    feature_df = dataframe.iloc[:, 3:]

    # Split remaining columns into groups of 10
    groups = [
        feature_df.iloc[:, i:i+5]
        for i in range(0, feature_df.shape[1], 5)
    ]

    # Generate every possible pair of groups
    for i, j in combinations(range(len(groups)), 2):

        group1 = groups[i]
        group2 = groups[j]

        # Get the corresponding target/return for each group
        target1 = returns.iloc[:, i]
        target2 = returns.iloc[:, j]

        # Combine dates, predictors and relevant targets
        pair_df = pd.concat(
            [date_df, group1, group2, target1, target2],
            axis=1
        )

        pairings.append(pair_df)
    
    return pairings
def ols_loop(pair, ticker1, ticker2):

    stock_A_close = pair[f"Close-{ticker1}"]
    stock_B_close = pair[f"Close-{ticker2}"]

    X = sm.add_constant(np.log(stock_A_close))
    y = np.log(stock_B_close)

    model_forward = sm.OLS(y, X).fit()

    X_back = sm.add_constant(np.log(stock_B_close))
    y_back = np.log(stock_A_close)

    model_backward = sm.OLS(y_back, X_back).fit()

    return {
        "forward": model_forward,
        "backward": model_backward
    }
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
def stats_significance(residual_A, residual_B, close_A, close_B):

    # ADF tests on individual price series
    adf_A = adfuller(residual_A)
    adf_B = adfuller(residual_B)

    # Engle-Granger cointegration test
    eg_test = coint(close_A, close_B)

    # Johansen cointegration test
    joh_data = pd.concat([close_A, closeB], axis=1).dropna()

    joh_test = coint_johansen(
        joh_data,
        det_order=0,
        k_ar_diff=0
    )

    return {
        "ADF_A": {
            "statistic": adf_A[0],
            "p_value": adf_A[1]
        },
        "ADF_B": {
            "statistic": adf_B[0],
            "p_value": adf_B[1]
        },
        "Engle_Granger": {
            "statistic": eg_test[0],
            "p_value": eg_test[1]
        },
        "Johansen": {
            "trace_statistic": joh_test.lr1[0],
            "critical_value_5pct": joh_test.cvt[0, 1]
        }
    }
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
def mean_std_2025(coin_pairs,pairing,year_data):
    
    X_25 = np.log(coin_pairs[f'{pairing}'].loc[f'{year_data}-01-01':f'{year_data}-12-31'][coin_pairs[f'{pairing}'].columns[3]])
    Y_25 = np.log(coin_pairs[f'{pairing}'].loc[f'{year_data}-01-01':f'{year_data}-12-31'][coin_pairs[f'{pairing}'].columns[8]])

    X_25 = sm.add_constant(X_25)
    
    model = sm.OLS(Y_25,X_25).fit()

    predicted = model.predict(X_25)
    residuals_man = Y_25 - predicted
    residuals_func = model.resid
    
    
    return residuals_func.mean(), residuals_func.std()
def unseen_residual(coin_pairs,pairing,year_start_date, year_end_date):
    
    X_25 = np.log(coin_pairs[f'{pairing}'].loc[f'{year_start_date}-01-01':f'{year_end_date}-12-31'][coin_pairs[f'{pairing}'].columns[3]])
    Y_25 = np.log(coin_pairs[f'{pairing}'].loc[f'{year_start_date}-01-01':f'{year_end_date}-12-31'][coin_pairs[f'{pairing}'].columns[8]])

    X_25 = sm.add_constant(X_25)
    
    model = sm.OLS(Y_25,X_25).fit()

    predicted = model.predict(X_25)
    residuals = Y_25 - predicted

    return residuals
def residuals_zscores_features(main_dict, pairs, year_s, year_e, means, std):
    for i in range(len(pairs)):
        main_dict[pairs[i]]['Residuals'] = unseen_residual(main_dict,pairs[i],year_s, year_e)
        main_dict[pairs[i]]['Z-score'] = main_dict[pairs[i]]['Residuals'].apply(lambda x: (x - means[i])/std[i])

    return main_dict
def trade_signals(df, threshold):

    position = None
    signals = []

    for z in df['Z-score']:

        if position is None and z < -(threshold):
            signal = 'Long Enter'
            position = 'Long'
    
        elif position == 'Long' and z >= 0:
            signal = 'Long Exit'
            position = None
    
        elif position is None and z > threshold:
            signal = 'Short Enter'
            position = 'Short'
    
        elif position == 'Short' and z <= 0:
            signal = 'Short Exit'
            position = None
    
        else:
            signal = 'Null'
    
        signals.append(signal)

    df['Signal'] = signals

    return df
def time_series_features(pairs, coin_pairs):
    for x in coin_pairs:
        pairs[x]['lag1'] = pairs[x]['Z-score'].shift(1)
        pairs[x]['lag2'] = pairs[x]['Z-score'].shift(2)
        pairs[x]['lag3'] = pairs[x]['Z-score'].shift(3)
        pairs[x]['rolling_std_20'] = pairs[x]['Z-score'].rolling(window=20).std()

    return pairs
def create_sequences(X, y, seq_len, forecast):
    Xs, ys = [], []
    
    for i in range(len(X) - seq_len - forecast + 1):
        
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+forecast])
    
    return np.array(Xs), np.array(ys)
import torch
import torch.nn as nn
def prepare_training_data(pairing_name, pairing_dict, seq_len, forecast):

    features = ['Z-score', 'lag1', 'lag2', 'lag3', 'rolling_std_20']
    target = ['Z-score']

    X_data = pairing_dict[pairing_name][features].fillna(0)
    Y_data = pairing_dict[pairing_name][target]

    X_array = X_data.values
    Y_array = Y_data.values

    X_sequences, Y_sequences = create_sequences(X_array, Y_array, seq_len, forecast)

    X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_sequences, dtype=torch.float32)

    return X_tensor, Y_tensor
def initialise_lstm():

    lstm = nn.LSTM(
        input_size=5,
        hidden_size=8,
        batch_first=True
    )

    fc = nn.Linear(8, 5)

    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(fc.parameters()),
        lr=0.01
    )

    return lstm, fc, loss, optimizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def test_evaluation(pairing_data):

    results = {}

    for pair_id, data in pairing_data.items():

        lstm = data["lstm"]
        fc = data["fc"]

        lstm.eval()
        fc.eval()

        with torch.no_grad():

            out, _ = lstm(data["X_test"])
            predictions = fc(out[:, -1, :])

            actual = data["Y_test"].squeeze(-1)

            test_loss = data["loss"](predictions, actual)

        predictions = predictions.cpu().numpy()
        actual = actual.cpu().numpy()

        horizon_metrics = {}

        for horizon in range(5):

            rmse = np.sqrt(
                mean_squared_error(
                    actual[:, horizon],
                    predictions[:, horizon]
                )
            )

            mae = mean_absolute_error(
                actual[:, horizon],
                predictions[:, horizon]
            )

            horizon_metrics[f"t+{horizon+1}"] = {
                "RMSE": rmse,
                "MAE": mae
            }

        results[pair_id] = {
            "pairing_name": data["name"],
            "predictions": predictions,
            "actual": actual,
            "MSE": test_loss.item(),
            "horizon_metrics": horizon_metrics
        }

    return results
def lstm_trade_signals(df, threshold):

    position = None
    signals = []

    for _, row in df.iterrows():

        z = row['Z-score']
        forecast = row['t+1']

        if position is None and z < -(threshold) and forecast > z:
            signal = 'Long Enter'
            position = 'Long'

        elif position == 'Long' and z >= 0:
            signal = 'Long Exit'
            position = None

        elif position is None and z > threshold and forecast < z:
            signal = 'Short Enter'
            position = 'Short'

        elif position == 'Short' and z <= 0:
            signal = 'Short Exit'
            position = None

        else:
            signal = 'Null'

        signals.append(signal)

    df = df.copy()
    df['Signal'] = signals

    return df
def prepare_forecast_df(test_input_df, test_results, seq_len=20):

    forecast_data = {}
    actual_data = {}

    for pair_id, pair_results in test_results.items():

        pairing_name = pair_results['pairing_name']
    
        predictions = pair_results['predictions']
        actual = pair_results['actual']
        pair_test_input_df = test_input_dfs[pairing_name]

    
        origin_dates = pair_test_input_df.index[
            seq_len - 1 : seq_len - 1 + len(predictions)
        ]
    
        actual_dates = pair_test_input_df.index[
            seq_len : seq_len + len(actual)
        ]
    
        forecast_data[pairing_name] = pd.DataFrame(
            predictions,
            index=origin_dates,
            columns=['t+1', 't+2', 't+3', 't+4', 't+5']
        )
        
        forecast_data[pairing_name]['Z-score'] = (
            pair_test_input_df.loc[origin_dates, 'Z-score'].values
        )
    
        actual_data[pairing_name] = pd.DataFrame(
            actual,
            index=actual_dates,
            columns=['t+1', 't+2', 't+3', 't+4', 't+5']
        )

    return forecast_data, actual_data
def evaluate_strategy(signal_df, price_df, beta, stock_a, stock_b):

    df = signal_df.copy()

    # Daily hedged spread returns
    spread_return = (
        np.log(price_df[f'Close-{stock_b}']).diff()
        - beta * np.log(price_df[f'Close-{stock_a}']).diff()
    )

    position = 0
    positions = []

    for signal in df['Signal']:
        if signal == 'Long Enter':
            position = 1
        elif signal == 'Short Enter':
            position = -1
        elif signal in ['Long Exit', 'Short Exit']:
            position = 0

        positions.append(position)

    df['Position'] = positions
    # Apply positions from the previous day's signal
    df['Strategy Return'] = (
        df['Position'].shift(1) * spread_return
    ).fillna(0)

    # Identify completed trades
    entries = df['Signal'].isin(['Long Enter', 'Short Enter'])
    exits = df['Signal'].isin(['Long Exit', 'Short Exit'])

    trade_returns = []
    trade_return = 0
    in_trade = False

    for position, daily_return, entry, exit_ in zip(
        df['Position'], df['Strategy Return'], entries, exits
    ):
        if entry:
            trade_return = 0
            in_trade = True

        if in_trade:
            trade_return += daily_return

        if exit_ and in_trade:
            trade_returns.append(trade_return)
            in_trade = False

    total_trades = len(trade_returns)

    # Total strategy return
    total_return = (1 + df['Strategy Return']).prod() - 1

    # Percentage of completed trades that were profitable
    win_rate = (
        sum(r > 0 for r in trade_returns) / total_trades
        if total_trades > 0 else np.nan
    )

    return {
        'Trades': total_trades,
        'Total Return': total_return,
        'Win Rate': win_rate
    }
def compare_strategies(*strategy_dicts):

    results = []

    for name, strategy in strategy_dicts:
        for pair, metrics in strategy.items():
            results.append({
                'Pair': pair,
                'Strategy': name,
                'Trades': metrics['Trades'],
                'Total Return': metrics['Total Return'],
                'Win Rate': metrics['Win Rate']
            })

    return pd.DataFrame(results).sort_values(
        ['Pair', 'Strategy']
    ).reset_index(drop=True)


def main():
    global AMZN_df, AVGO_df, IBM_df, INTC_df, META_df, NVDA_df, ORCL_df, Pairing_names
    global Pairing_names_26, SP500_df, TSLA_df, VIX_df, X, X_array, X_sequences, X_tensor
    global X_test_tensors, Y_array, Y_sequences, Y_tensor, Y_test_tensors, a, actual_data, beta_dict
    global coin, cointegrated_pairs, cols, cols_26, column_name, comparison_df, d, df
    global fc, features, fitted_values, forecast, forecast_data, h, i, l
    global loss, lstm, lstm_results, masterdata, models, name, o, ols_model
    global optimizer, pair, pair1_mean_resid, pair1_std_resid, pair2_mean_resid, pair2_std_resid, pair3_mean_resid, pair3_std_resid
    global pair4_mean_resid, pair4_std_resid, pair5_mean_resid, pair5_std_resid, pair_df, pairing_2026, pairing_dict, pairing_dict_26
    global pairing_name, pairing_test_data, pairings, pairings_means, pairings_stds, processed_masterdata, processed_unseen, r
    global residuals, seq_len, stat_sig, stat_significance, statistical_results, statistical_sensitivity, stock_a, stock_b
    global strat_eval_lstm, strat_eval_mean_reversion, strat_eval_sensitive, target, test_input_df, test_input_dfs, test_results, training_tensors
    global unseen_df, x, y

    warnings.filterwarnings("ignore")
    SP500_df = yf.download("^GSPC", start="2020-01-01", end="2025-12-31")
    VIX_df = yf.download("^VIX", start="2020-01-01", end="2025-12-31")
    AMZN_df = yf.download("AMZN", start="2020-01-01", end="2025-12-31")
    TSLA_df = yf.download("TSLA", start="2020-01-01", end="2025-12-31")
    META_df = yf.download("META", start="2020-01-01", end="2025-12-31")
    ORCL_df = yf.download("ORCL", start="2020-01-01", end="2025-12-31")
    INTC_df = yf.download("INTC", start="2020-01-01", end="2025-12-31")
    IBM_df = yf.download("IBM", start="2020-01-01", end="2025-12-31")
    NVDA_df = yf.download("NVDA", start="2020-01-01", end="2025-12-31")
    AVGO_df = yf.download("AVGO", start="2020-01-01", end="2025-12-31")
    SP500_df = SP500_df.reset_index()
    AMZN_df = AMZN_df.reset_index()
    TSLA_df = TSLA_df.reset_index()
    META_df = META_df.reset_index()
    ORCL_df = ORCL_df.reset_index()
    INTC_df = INTC_df.reset_index()
    IBM_df = IBM_df.reset_index()
    NVDA_df = NVDA_df.reset_index()
    AVGO_df = AVGO_df.reset_index()
    VIX_df = VIX_df.reset_index()
    SP500_df = time_split_relocate(SP500_df)
    AMZN_df = time_split_relocate(AMZN_df)
    TSLA_df = time_split_relocate(TSLA_df)
    META_df = time_split_relocate(META_df)
    ORCL_df = time_split_relocate(ORCL_df)
    INTC_df = time_split_relocate(INTC_df)
    IBM_df = time_split_relocate(IBM_df)
    NVDA_df = time_split_relocate(NVDA_df)
    AVGO_df = time_split_relocate(AVGO_df)
    VIX_df = time_split_relocate(VIX_df)
    masterdata = merging_data(AMZN_df, TSLA_df, META_df, ORCL_df, INTC_df, IBM_df, NVDA_df, AVGO_df)
    masterdata
    cols = masterdata.columns.tolist()
    processed_masterdata = pd.DataFrame()   
    for r in cols:
        column_name = str(r).replace("(", "").replace(")","").replace("'","").replace(",","-").replace(" ","").replace("'", "").replace("Month-","Month").replace("Year-","Year").replace("Quarter-","Quarter")
        processed_masterdata[column_name] = masterdata[r]
    daily_returns(processed_masterdata, SP500_df)
    processed_masterdata = processed_masterdata.fillna(0)
    pairings = []
    pairings = pairing(processed_masterdata)
    Pairing_names = []
    for x in pairings:
        a = (f"pairing_"+str(str(x.columns[4]).split("-")[1].split("'")[0])+"_"+str(str(x.columns[9]).split("-")[1].split("'")[0]))
        Pairing_names.append(a)
    pairing_dict = {
        name: df
        for name, df in zip(Pairing_names, pairings)
    }
    Pairing_names
    models = ols_loop(pairing_dict,Pairing_names)
    residuals = pd.DataFrame(columns=[Pairing_names])
    for l in Pairing_names:
        residuals[l] = models[l]['forward'].resid # What does a staionary residual mean? Chat GPT states a spread is staionary is what we are looking for - linked to p-value
    fitted_values = pd.DataFrame(columns=[Pairing_names])
    for h in Pairing_names:
        fitted_values[h] = models[h]['forward'].fittedvalues
    stat_sig = pd.DataFrame(index=Pairing_names,columns=['ADF_Statistic','EG_Test','Johansen_Test_trace','Johansen_Test_cvt'])
    stat_significance = stats_significane(stat_sig,pairing_dict,residuals,three=['pairing_IBM_AVGO','pairing_IBM_NVDA'])
    stat_significance
    cointegrated_pairs = stat_sig[(stat_sig['ADF_Statistic'] < 0.05) & (stat_sig['EG_Test'] < 0.05) & (stat_sig['Johansen_Test_trace'] > stat_sig['Johansen_Test_cvt'])].index.tolist()
    stat_sig[(stat_sig['ADF_Statistic'] < 0.05) & (stat_sig['EG_Test'] < 0.05) & (stat_sig['Johansen_Test_trace'] > stat_sig['Johansen_Test_cvt'])]
    pair1_mean_resid, pair1_std_resid = mean_std_2025(pairing_dict, cointegrated_pairs[0], year_data='2025')
    pair2_mean_resid, pair2_std_resid = mean_std_2025(pairing_dict, cointegrated_pairs[1], year_data='2025')
    pair3_mean_resid, pair3_std_resid = mean_std_2025(pairing_dict, cointegrated_pairs[2], year_data='2025')
    pair4_mean_resid, pair4_std_resid = mean_std_2025(pairing_dict, cointegrated_pairs[3], year_data='2025')
    pair5_mean_resid, pair5_std_resid = mean_std_2025(pairing_dict, cointegrated_pairs[4], year_data='2025')
    SP500_df = yf.download("^GSPC", start="2026-01-01", end="2026-07-31")
    AMZN_df = yf.download("AMZN", start="2026-01-01", end="2026-07-31")
    TSLA_df = yf.download("TSLA", start="2026-01-01", end="2026-07-31")
    META_df = yf.download("META", start="2026-01-01", end="2026-07-31")
    ORCL_df = yf.download("ORCL", start="2026-01-01", end="2026-07-31")
    INTC_df = yf.download("INTC", start="2026-01-01", end="2026-07-31")
    IBM_df = yf.download("IBM", start="2026-01-01", end="2026-07-31")
    NVDA_df = yf.download("NVDA", start="2026-01-01", end="2026-07-31")
    AVGO_df = yf.download("AVGO", start="2026-01-01", end="2026-07-31")
    SP500_df = SP500_df.reset_index()
    AMZN_df = AMZN_df.reset_index()
    TSLA_df = TSLA_df.reset_index()
    META_df = META_df.reset_index()
    ORCL_df = ORCL_df.reset_index()
    INTC_df = INTC_df.reset_index()
    IBM_df = IBM_df.reset_index()
    NVDA_df = NVDA_df.reset_index()
    AVGO_df = AVGO_df.reset_index()
    SP500_df = time_split_relocate(SP500_df)
    AMZN_df = time_split_relocate(AMZN_df)
    TSLA_df = time_split_relocate(TSLA_df)
    META_df = time_split_relocate(META_df)
    ORCL_df = time_split_relocate(ORCL_df)
    INTC_df = time_split_relocate(INTC_df)
    IBM_df = time_split_relocate(IBM_df)
    NVDA_df = time_split_relocate(NVDA_df)
    AVGO_df = time_split_relocate(AVGO_df)
    VIX_df = time_split_relocate(VIX_df)
    unseen_df = merging_data(AMZN_df, TSLA_df, META_df, ORCL_df, INTC_df, IBM_df, NVDA_df, AVGO_df)
    cols_26 = unseen_df.columns.tolist()
    processed_unseen = pd.DataFrame()   
    for a in cols_26:
        column_name = str(a).replace("(", "").replace(")","").replace("'","").replace(",","-").replace(" ","").replace("'", "").replace("Month-","Month").replace("Year-","Year").replace("Quarter-","Quarter")
        processed_unseen[column_name] = unseen_df[a]
    daily_returns(processed_unseen, SP500_df)
    processed_unseen = processed_unseen.fillna(0)
    pairing_2026 = pairing(processed_unseen)
    Pairing_names_26 = []
    for o in pairing_2026:
        name = (f"pairing_"+str(str(o.columns[4]).split("-")[1].split("'")[0])+"_"+str(str(o.columns[9]).split("-")[1].split("'")[0]))
        Pairing_names_26.append(name)
    pairing_dict_26 = {
        name: df
        for name, df in zip(Pairing_names_26, pairing_2026)
    }
    for a in Pairing_names_26:
        coin = []
        coin = [a for a in Pairing_names_26 if a not in cointegrated_pairs]
    for d in coin:
        pairing_dict_26.pop(d,None)
    pairings_means = [pair1_mean_resid, pair2_mean_resid, pair3_mean_resid, pair4_mean_resid, pair5_mean_resid]
    pairings_stds = [pair1_std_resid, pair2_std_resid, pair3_std_resid, pair4_std_resid, pair5_std_resid]
    pairing_dict = residuals_zscores_features(pairing_dict, cointegrated_pairs,'2020', '2025', pairings_means, pairings_stds)
    pairing_dict_26 = residuals_zscores_features(pairing_dict_26, cointegrated_pairs,'2026', '2026', pairings_means, pairings_stds)
    trade_signals(pairing_dict_26[cointegrated_pairs[0]], 2)
    trade_signals(pairing_dict_26[cointegrated_pairs[1]], 2)
    trade_signals(pairing_dict_26[cointegrated_pairs[2]], 2)
    trade_signals(pairing_dict_26[cointegrated_pairs[3]], 2)
    trade_signals(pairing_dict_26[cointegrated_pairs[4]], 2)
    pairing_dict_26 = time_series_features(pairing_dict_26, cointegrated_pairs)
    pairing_dict = time_series_features(pairing_dict, cointegrated_pairs)
    seq_len = 20  # number of past time steps (days) to use
    forecast = 5
    training_tensors = {}
    for pair in cointegrated_pairs:
        X_tensor, Y_tensor = prepare_training_data(pair, pairing_dict, seq_len, forecast)

        training_tensors[pair] = {
            'X': X_tensor,
            'Y': Y_tensor
        }
    models = {}
    for pair in cointegrated_pairs:

        lstm, fc, loss, optimizer = initialise_lstm()

        models[pair] = {
            'lstm': lstm,
            'fc': fc,
            'loss': loss,
            'optimizer': optimizer
        }
    X_test_tensors = {}
    Y_test_tensors = {}
    features = ['Z-score', 'lag1', 'lag2', 'lag3', 'rolling_std_20']
    target = ['Z-score']
    for pair in cointegrated_pairs:

        X_array = pairing_dict_26[pair][features].fillna(0).values
        Y_array = pairing_dict_26[pair][target].values

        X_sequences, Y_sequences = create_sequences(X_array, Y_array, seq_len, forecast)

        X_test_tensors[pair] = torch.tensor(X_sequences, dtype=torch.float32)
        Y_test_tensors[pair] = torch.tensor(Y_sequences, dtype=torch.float32)
    pairing_test_data = {}
    for i, pair in enumerate(cointegrated_pairs):

        pairing_test_data[f"pair{i+1}"] = {
            "X_test": X_test_tensors[pair],
            "Y_test": Y_test_tensors[pair],
            "lstm": models[pair]["lstm"],
            "fc": models[pair]["fc"],
            "loss": models[pair]["loss"],
            "name": pair
        }
    test_results = test_evaluation(pairing_test_data)
    test_input_dfs = {}
    for pairing_name in cointegrated_pairs:

        test_input_df = pd.concat([
                pairing_dict[pairing_name].tail(20),
                pairing_dict_26[pairing_name]
            ])

        test_input_dfs[pairing_name] = test_input_df
    forecast_data, actual_data = prepare_forecast_df(
        test_input_dfs, test_results
    )
    statistical_results = {}
    lstm_results = {}
    statistical_sensitivity = {}
    for pairing_name, df in forecast_data.items():

        statistical_results[pairing_name] = trade_signals(
            df,
            threshold=2
        )

        lstm_results[pairing_name] = lstm_trade_signals(
            df,
            threshold=2
        )

        statistical_sensitivity[pairing_name] = trade_signals(
            df,
            threshold=1
        )
    beta_dict = {}
    for pairing_name, pair_df in pairing_dict_26.items():

        stock_a = pairing_name.replace('pairing_', '').split('_')[0]
        stock_b = pairing_name.replace('pairing_', '').split('_')[1]

        X = np.log(pair_df[f'Close-{stock_a}'])
        y = np.log(pair_df[f'Close-{stock_b}'])

        X = sm.add_constant(X)

        ols_model = sm.OLS(y, X).fit()

        beta_dict[pairing_name] = ols_model.params.iloc[1]
    strat_eval_mean_reversion = {}
    for name in cointegrated_pairs:
        strat_eval_mean_reversion[name] = evaluate_strategy(statistical_results[name],
                                             pairing_dict_26[name], 
                                             beta_dict[name],
                                             pairing_dict_26[name].columns[3][-3:] if pairing_dict_26[name].columns[3][-4:].endswith('IBM') else pairing_dict_26[name].columns[3][-4:],
                                             pairing_dict_26[name].columns[8][-3:] if pairing_dict_26[name].columns[8][-4:].endswith('IBM') else pairing_dict_26[name].columns[8][-4:],)
    strat_eval_lstm = {}
    for name in cointegrated_pairs:
        strat_eval_lstm[name] = evaluate_strategy(lstm_results[name],
                                             pairing_dict_26[name], 
                                             beta_dict[name],
                                             pairing_dict_26[name].columns[3][-3:] if pairing_dict_26[name].columns[3][-4:].endswith('IBM') else pairing_dict_26[name].columns[3][-4:],
                                             pairing_dict_26[name].columns[8][-3:] if pairing_dict_26[name].columns[8][-4:].endswith('IBM') else pairing_dict_26[name].columns[8][-4:],)
    strat_eval_sensitive = {}
    for name in cointegrated_pairs:
        strat_eval_sensitive[name] = evaluate_strategy(statistical_sensitivity[name],
                                             pairing_dict_26[name], 
                                             beta_dict[name],
                                             pairing_dict_26[name].columns[3][-3:] if pairing_dict_26[name].columns[3][-4:].endswith('IBM') else pairing_dict_26[name].columns[3][-4:],
                                             pairing_dict_26[name].columns[8][-3:] if pairing_dict_26[name].columns[8][-4:].endswith('IBM') else pairing_dict_26[name].columns[8][-4:],)
    comparison_df = compare_strategies(
        ('Statistical', strat_eval_mean_reversion),
        ('LSTM', strat_eval_lstm),
        ('Sensitivity', strat_eval_sensitive)
    )
    comparison_df.sort_values(by='Total Return', ascending=False)


if __name__ == "__main__":
    main()
