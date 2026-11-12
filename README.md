# Statistical Arbitrage & Pair Trading Strategy

## Project Overview

This project explores the application of statistical arbitrage techniques and machine learning to develop a quantitative pair trading strategy using publicly traded technology stocks.

The objective is to identify pairs of stocks that exhibit a long-term statistical relationship, model deviations from that relationship, and generate trading signals based on the expectation that those deviations will revert towards their historical equilibrium.

The project combines financial data analysis, econometrics, statistical hypothesis testing, time-series feature engineering, machine learning and API development.

The final deliverable includes a Python-based research pipeline and a deployed FastAPI application that allows users to evaluate individual stock pairs and generate pair trading signals.

**Live API:** https://capstone-project-pair-trading.onrender.com

---

## Project Objectives

The project was developed around the following objectives:

- Identify potential pairs of technology stocks with historically related price movements.
- Apply cointegration testing to establish whether stock pairs exhibit a statistically significant long-term relationship.
- Estimate the equilibrium relationship between each pair using Ordinary Least Squares (OLS) regression.
- Calculate the spread and standardised Z-scores to identify deviations from equilibrium.
- Develop a mean-reversion trading strategy using predefined entry and exit thresholds.
- Explore whether Long Short-Term Memory (LSTM) neural networks can improve trading decisions through time-series forecasting.
- Evaluate the performance of the statistical and machine learning-enhanced strategies.
- Develop and deploy a FastAPI application to make the pair trading methodology accessible through a web interface.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Financial data | yfinance |
| Statistical analysis | Statsmodels, SciPy |
| Machine learning | Scikit-learn, TensorFlow / Keras |
| Visualisation | Matplotlib |
| API development | FastAPI |
| API server | Uvicorn |
| Deployment | Render |
| Development environment | Jupyter Notebook |

---

## 1. Financial Data Collection and Preparation

Historical stock price data is retrieved using the Yahoo Finance API through the `yfinance` Python library.

The initial research focused on eight technology stocks:

- Amazon (AMZN)
- Tesla (TSLA)
- Meta Platforms (META)
- Oracle (ORCL)
- Intel (INTC)
- IBM (IBM)
- NVIDIA (NVDA)
- Broadcom (AVGO)

Historical OHLCV (Open, High, Low, Close and Volume) data is collected and consolidated into a single dataset.

### Data preparation

The preprocessing stage includes:

1. Downloading historical financial data.
2. Converting the date column into a datetime index.
3. Extracting additional time-based variables, including month, year and quarter.
4. Consolidating stock datasets into a common DataFrame.
5. Standardising column names to identify individual stock price variables.
6. Calculating daily returns where required for exploratory analysis.

The resulting dataset provides a consistent structure for subsequent statistical analysis.

---

## 2. Stock Pair Generation

All possible combinations of the selected technology stocks are generated to identify potential relationships.

With eight stocks, this produces 28 unique stock pairs.

Each pair is extracted into its own DataFrame, allowing the individual relationships to be evaluated independently.

This approach avoids arbitrarily selecting pairs based on correlation alone and provides a systematic starting point for cointegration analysis.

---

## 3. Statistical Analysis and Cointegration Testing

A key component of the project is distinguishing between correlation and cointegration.

Two stocks can exhibit correlated price movements without maintaining a stable long-term equilibrium relationship. Cointegration testing helps identify whether a linear combination of their non-stationary price series is stationary.

### Ordinary Least Squares regression

For each candidate pair, the natural logarithm of closing prices is used to estimate the long-term relationship.

The forward regression is defined as:

\[
\log(P_B) = \alpha + \beta\log(P_A) + \epsilon
\]

Where:

- \(P_A\) is the closing price of stock A.
- \(P_B\) is the closing price of stock B.
- \(\alpha\) is the intercept.
- \(\beta\) is the estimated regression coefficient.
- \(\epsilon\) is the regression residual.

The residuals represent the deviation of the observed relationship from the estimated equilibrium.

A backward regression is also estimated, reversing the dependent and independent variables.

### Statistical tests

Three statistical approaches are incorporated into the original pair-selection methodology.

**Augmented Dickey-Fuller (ADF) test**

The ADF test is applied to the regression residuals to assess whether the estimated spread is stationary.

The null hypothesis is that the series contains a unit root.

A p-value below 0.05 provides evidence against the null hypothesis.

**Engle-Granger cointegration test**

The Engle-Granger test evaluates whether two non-stationary price series are cointegrated.

The null hypothesis is that the series are not cointegrated.

A p-value below 0.05 provides evidence of cointegration.

**Johansen cointegration test**

The Johansen test provides an additional approach to evaluating cointegration and estimating the number of cointegrating relationships.

The trace statistic is compared against its 5% critical value.

A trace statistic greater than the critical value provides evidence against the null hypothesis.

### Pair selection

The results of the statistical tests are consolidated into a results DataFrame.

Pairs satisfying the statistical selection criteria are retained for further analysis.

This process reduces the original 28 combinations to a smaller group of candidate pairs with evidence of a stable long-term relationship.

---

## 4. Spread Calculation and Z-Score Standardisation

Once suitable pairs have been identified, the regression residuals are used to construct the trading spread.

The spread is calculated as:

\[
Spread_t = \log(P_{B,t}) -
(\alpha + \beta\log(P_{A,t}))
\]

The spread measures the difference between the actual log price of stock B and the price predicted by the regression model.

### Z-score calculation

To identify unusually large deviations from the estimated equilibrium, the spread is standardised using its historical mean and standard deviation.

\[
Z_t = \frac{Spread_t-\mu}{\sigma}
\]

Where:

- \(Spread_t\) is the current residual.
- \(\mu\) is the historical mean of the spread.
- \(\sigma\) is the historical standard deviation.

A Z-score close to zero indicates that the relationship is near its historical equilibrium.

Large positive or negative values indicate that the relationship has deviated significantly from its historical mean.

### Avoiding data leakage

The original research pipeline separates the historical dataset into training and testing periods.

- Training period: 2020–2025.
- Testing period: 2026.

OLS coefficients, residual means and standard deviations are estimated using the training data.

These parameters are then held constant when calculating test-period Z-scores.

This ensures that the test-period calculations do not use future information when estimating the historical equilibrium.

---

## 5. Statistical Mean-Reversion Trading Strategy

The trading strategy is based on the hypothesis that deviations from a cointegrated relationship will eventually revert towards their historical equilibrium.

The primary entry threshold is a Z-score of ±2, with ±1 also explored as a sensitivity analysis.

### Trading rules

| Condition | Trading signal |
|---|---|
| Z-score falls below -2 | Enter a long spread position |
| Z-score rises above +2 | Enter a short spread position |
| Z-score returns to zero | Exit the position |

The direction of the spread position determines the relative exposure to the two stocks.

For the forward regression, where stock B is modelled using stock A:

- A long spread position corresponds to buying B and shorting the appropriately weighted A exposure.
- A short spread position corresponds to shorting B and buying the appropriately weighted A exposure.

The regression coefficient is used to determine the relative hedge exposure.

### Strategy evaluation

The strategy is evaluated using historical spread movements and the corresponding stock returns.

Performance measures include:

- Total strategy return.
- Number of completed trades.
- Win rate.
- Trading signal frequency.
- Cumulative strategy performance.

Trading positions are shifted to the following trading day after a signal is generated, avoiding the assumption that a position can be entered retrospectively at the same closing price used to calculate the signal.

---

## 6. Time-Series Feature Engineering and LSTM Modelling

An additional machine learning component explores whether an LSTM neural network can forecast future movements in the spread's Z-score.

LSTMs are designed to process sequential data and capture temporal relationships that may not be adequately represented by traditional statistical approaches.

### Feature engineering

The following features are used:

| Feature | Description |
|---|---|
| Z-score | Current standardised spread |
| lag1 | Previous day's Z-score |
| lag2 | Z-score from two trading days earlier |
| lag3 | Z-score from three trading days earlier |
| rolling_std_20 | Rolling 20-day standard deviation |

These features provide the model with information about the current spread position, recent movements and changing volatility.

### Sequence preparation

The time-series data is transformed into sequential samples.

- Sequence length: 20 trading days.
- Forecast horizon: 5 trading days.
- Input features: 5.

Each training sample therefore contains 20 consecutive observations of the engineered features, which are used to forecast the subsequent five Z-score values.

### LSTM architecture

The model uses the following architecture:

- Input layer: 5 features.
- LSTM layer: 8 hidden units.
- Dense output layer: 5 forecast values.

Separate models and preprocessing scalers are maintained for the individual stock pairs.

The forecast outputs are subsequently evaluated to explore whether predicted Z-score movements can help filter statistical trading signals.

### Machine learning-enhanced strategy

The LSTM-enhanced strategy combines the original statistical entry conditions with the predicted direction of the Z-score.

The intention is to reduce potentially unfavourable entries by requiring the forecast to provide additional support for the expected mean-reversion movement.

The statistical strategy and LSTM-enhanced strategy are then compared using the same underlying evaluation framework.

---

## 7. FastAPI Application

The final development phase converts the statistical pair trading methodology into a web-accessible application using FastAPI.

Rather than requiring users to work directly with the original research notebook and predefined stock pairs, the application allows them to submit their own ticker combinations.

### Application workflow

1. The user enters two stock ticker symbols.
2. The application downloads the corresponding historical price data.
3. The data is prepared and transformed into the format required by the analysis functions.
4. An OLS regression is fitted to estimate the relationship between the two stocks.
5. Statistical tests are performed to evaluate cointegration.
6. Where sufficient evidence of cointegration is found, the regression residuals are used to calculate the spread and Z-scores.
7. Trading signals are generated using the defined mean-reversion thresholds.
8. The results are returned through the API and displayed by the application.

### Cointegration assessment

The application incorporates statistical checks to classify the strength of the evidence for cointegration.

The checks include:

- ADF residual test.
- Engle-Granger cointegration test.
- Johansen trace statistic.

The results are presented alongside the relevant statistics and thresholds.

The application uses the combined results to provide an overall assessment:

- **Strong Cointegration:** All three checks are satisfied.
- **Moderate Support:** One or two checks are satisfied.
- **Weak / Inconclusive:** No checks are satisfied.

This classification is a project-specific decision rule rather than a formal statistical test in its own right.

### API endpoint

The application exposes a prediction endpoint that accepts two ticker symbols.

Example:

`/predict?ticker1=TSLA&ticker2=AMZN`

The API returns the analysis results for the selected pair.

The root endpoint is also available as a basic health check:

`/`

---

## 8. Deployment

The FastAPI application is deployed using Render.

The application runs using Uvicorn, with the host and port configured for Render's deployment environment.

The deployment configuration uses:

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The deployed application can be accessed at:

https://capstone-project-pair-trading.onrender.com

FastAPI also provides interactive API documentation through:

`/docs`

This allows users to inspect the available endpoints and submit requests directly through the browser.

---

## 9. Key Findings and Evaluation

The original statistical strategy was evaluated across five selected stock pairs using the ±2 Z-score entry threshold.

The historical results were:

| Stock pair | Completed trades | Total strategy return | Win rate |
|---|---:|---:|---:|
| ORCL–IBM | 4 | 75.04% | 100% |
| ORCL–NVDA | 5 | 50.93% | 80% |
| ORCL–AVGO | 3 | 29.96% | 66.67% |
| IBM–NVDA | 3 | 51.01% | 100% |
| IBM–AVGO | 2 | 31.51% | 100% |

These results demonstrate how statistical relationships between individual stocks can be translated into a systematic trading strategy.

However, the results should be interpreted with caution.

The strategy uses a simplified spread-return calculation rather than a fully capital-weighted portfolio simulation. The historical results do not automatically account for transaction costs, borrowing costs, slippage, market impact or all practical short-selling constraints.

The number of completed trades is also relatively small, meaning that the observed win rates should not be interpreted as evidence of guaranteed future performance.

The LSTM component provides an additional avenue for evaluating whether time-series forecasts can improve entry decisions, rather than assuming that machine learning will necessarily outperform the statistical baseline.

---

## 10. Project Structure

The repository can be organised as follows:

```text
pair-trading-project/
│
├── main.py
│   └── FastAPI application and API endpoints
│
├── main_script_ready.py
│   └── Core data preparation, statistical analysis,
│       feature engineering and strategy functions
│
├── requirements.txt
│   └── Project dependencies
│
├── notebooks/
│   └── Pair trading research and model development
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Excluded files and local environment settings
```

The structure is intended to separate the research and development workflow from the deployed API.

---

## 11. Future Improvements

There are several opportunities to extend the project:

- Incorporate transaction costs, borrowing fees and slippage into strategy evaluation.
- Develop a more realistic portfolio-level backtesting framework.
- Evaluate hedge-ratio stability and implement rolling or dynamic regression models.
- Explore alternative cointegration methods and more robust pair-selection criteria.
- Improve LSTM architecture and compare it with alternative forecasting models.
- Introduce additional risk management rules, including stop-losses and position sizing.
- Expand the application to support multiple pairs and comparative performance analysis.
- Introduce automated monitoring and scheduled data updates.

---

## Disclaimer

This project is an educational and research-based exploration of quantitative trading strategies.

Statistical cointegration does not guarantee that a relationship will remain stable or revert to its historical mean. Historical backtesting results are not indicative of future performance.

The application is not intended to provide financial advice or automated investment recommendations.

---

## Author

Developed as part of a postgraduate programme in Artificial Intelligence and Machine Learning.

The project demonstrates the application of Python, econometrics, statistical arbitrage, time-series modelling and machine learning to a practical financial data science problem.
