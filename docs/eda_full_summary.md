# EDA Summary

## What we did

We downloaded 10 years of daily price data for 16 Vanguard ETFs from Yahoo Finance using yfinance, covering January 2015 to May 2026. The data came in clean -- all 16 tickers loaded with exactly 2,854 trading days each, no missing prices, and only one zero-volume day in BND which was forward-filled.

We stacked the 16 individual DataFrames into a single panel of 45,664 rows indexed by date and ticker. On top of the 6 raw price and volume columns we then engineered 41 features following the data dictionary -- log returns at 5 different horizons, rolling volatility, drawdown, moving averages, MACD, RSI, ATR, Bollinger Bands, volume z-score and calendar flags.

With the full feature set built we ran individual EDA for every ETF -- a 7-panel chart covering price, drawdown, return distribution, rolling volatility, MACD, RSI and volume, plus a summary statistics table per asset. We then ran cross-asset EDA comparing all 16 ETFs together through correlation, risk-return, cumulative returns, rolling correlation against VOO, volatility and drawdown.

Finally we ran the statistical checks -- stationarity, normality, autocorrelation, a NaN audit, Sharpe and Calmar ratios, and VaR and CVaR at 95 and 99 percent confidence. After dropping the warm-up rows we are left with 41,632 clean rows representing 91.2 percent of the full dataset.

---

## What the analysis tells us

Returns are stationary but not normal. Every single ETF passed the ADF stationarity test on daily returns. Every single one also failed the Jarque-Bera normality test. The return distributions all have fat tails and negative skew. Any model that assumes Gaussian returns will underestimate tail risk for all 16 assets.

Almost everything is autocorrelated. 15 of the 16 ETFs showed statistically significant return autocorrelation under the Ljung-Box test. BNDX was the only exception. This means returns are not purely random day to day and simple models that ignore this structure will leave predictable patterns on the table.

The asset classes behave very differently from each other. VGT had a Sharpe of 0.92 and a Calmar of 0.63, by far the best in the universe. The broad US market ETFs VOO, VTI and VUG clustered between Sharpe 0.74 and 0.78. At the bottom VDE came in at Sharpe 0.22 and Calmar 0.10, entirely explained by its 69 percent drawdown during the 2020 oil crash. Bond ETFs sit in their own low-volatility, low-return world with BND losing 18.6 percent at its worst during the 2022 rate hike cycle.

Volatility is not constant -- it clusters. Every volatility chart shows the same pattern. Markets were relatively calm from 2015 to 2019, then March 2020 produced a spike that dwarfs everything before or after it. After 2020 volatility settled at a structurally higher level than pre-COVID. This clustering means GARCH-type models are more appropriate than fixed-volatility assumptions.

Correlations rise in a crisis. Under normal conditions the diversification across asset classes looks reasonable. Bonds, international equities and domestic equities do not all move together. But during March 2020 the rolling correlations against VOO converged sharply upward across almost every ticker. Diversification tends to fail exactly when you need it most.

Tail risk is proportional to asset class risk. BND has a 99 percent CVaR of minus 1.32 percent, meaning on its worst days you lose about 1.3 percent. VDE has a 99 percent CVaR of minus 7.66 percent. The equity ETFs sit between minus 4.6 and minus 5.9 percent at the 99 percent level. This hierarchy is consistent and logical.

The NaN situation is clean. The 4,032 rows dropped represent the warm-up period at the start of each ticker where the longest rolling windows cannot yet produce a value. This is expected. The remaining 41,632 rows in panel_clean are fully usable for modelling.

---

## Technical challenges

The panel has a two-level index of date and ticker. When we tried to compute calendar features inside the feature engineering function, Python could not call date methods directly on a MultiIndex. We had to extract the date level separately before applying day of week, month and month-end calculations.

Feature engineering had to be applied within each ticker separately, not across the whole panel at once. Rolling windows and cumulative calculations like drawdown would have bled across tickers if we had not used groupby to isolate each asset before computing.

The NaN audit initially reported every row as unusable. This was because we were checking whether any column in the entire row had a NaN, which is too strict. Columns like downside volatility are legitimately empty on windows where every daily return was positive. We fixed this by only checking the columns we actually need for modelling, which gave us a clean and accurate picture of what data is usable.

BND had one day where Yahoo Finance recorded valid prices but left volume as zero. A zero in that column would have caused a division by zero later when computing the volume z-score. We caught it in the sanity check and forward-filled it before building the panel.

The cumulative returns comparison chart used log returns to build the growth index but the cross-asset chart used simple returns with cumprod. These two approaches give slightly different numbers and we had to make sure we were consistent across the individual and cross-asset EDA so comparisons were valid.

---

## Where we stand

The data is clean, the features are built, the distributions are understood, the risk metrics are documented and the modelling dataset is ready. The next step is baseline analysis.
