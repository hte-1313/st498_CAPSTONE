# EDA Sufficiency Review

---

## What we have done

The individual ETF EDA looks good. Each asset has a 7-panel chart showing price, drawdown, return distribution, volatility, MACD, RSI and volume. We also have summary stats with skew and kurtosis per ETF and all 41 features from the data dictionary have been built correctly.

The cross-asset work is also solid — correlation heatmap, risk-return scatter, cumulative returns, rolling correlation against VOO, vol comparison and drawdown comparison are all done.

---

## What we still need before moving on

**Stationarity.** We imported `adfuller` but never actually ran it. We need to formally test whether returns are stationary before building any model on top of them. This is not optional.

**Normality test.** Same situation — `jarque_bera` is sitting in the imports unused. Skew and kurtosis give a hint visually but we need the actual test result per ETF.

**Autocorrelation.** We imported `acf`, `pacf` and the plot functions and never touched them. Whether returns are autocorrelated changes what kind of model makes sense. We need at least a quick check here.

**NaN audit.** After feature engineering we never checked how many NaN rows we actually have. The 200-day SMA and 252-day return alone wipe out the first year of data per ticker. We should know exactly what we are dropping before we model anything.

**Sharpe and Calmar.** These are two numbers every analyst expects to see before baseline modelling. We have all the data to compute them, we just have not done it yet.

**VaR and CVaR.** We showed the 5th percentile line on the return distribution chart but never put a number to it. A simple table with 95% and 99% VaR per ETF wraps up the tail risk story properly.

---

## Where we are

Visually the EDA is done. Statistically it is not. The three things that actually block us from moving forward are stationarity, autocorrelation and the NaN audit. The rest are worth doing but will not stop us.
