# stats/statistical_checks.py
# Formal statistical tests run before baseline modelling.
# Covers: ADF stationarity, Jarque-Bera normality, Ljung-Box autocorrelation,
#         NaN audit, Sharpe ratio, Calmar ratio, VaR and CVaR.

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera


def run_statistical_checks(panel):
    ret_panel = panel["ret_1d"].unstack("ticker")

    print("Stationarity, ADF Test on Daily Returns")
    print("-" * 45)
    for ticker in ret_panel.columns:
        series = ret_panel[ticker].dropna()
        _, p_value, _, _, _, _ = adfuller(series)
        result = "stationary" if p_value < 0.05 else "non-stationary"
        print(f"  {ticker:<6}  p={p_value:.4f}  {result}")

    print("\nNormality, Jarque-Bera Test on Daily Returns")
    print("-" * 45)
    for ticker in ret_panel.columns:
        series = ret_panel[ticker].dropna()
        _, p_value = jarque_bera(series)
        result = "normal" if p_value > 0.05 else "not normal"
        print(f"  {ticker:<6}  p={p_value:.4f}  {result}")

    print("\nAutocorrelation, Ljung-Box Test on Daily Returns (lag 10)")
    print("-" * 55)
    for ticker in ret_panel.columns:
        series    = ret_panel[ticker].dropna()
        lb_result = sm.stats.acorr_ljungbox(series, lags = [10], return_df = True)
        p_value   = lb_result["lb_pvalue"].values[0]
        result    = "autocorrelated" if p_value < 0.05 else "no autocorrelation"
        print(f"  {ticker:<6}  p={p_value:.4f}  {result}")

    print("\nNaN Audit, Post Feature Engineering")
    print("-" * 45)
    for ticker in panel.index.get_level_values("ticker").unique():
        df       = panel.xs(ticker, level = "ticker")
        total    = len(df)
        nan_rows = df.isnull().any(axis = 1).sum()
        clean    = total - nan_rows
        print(f"  {ticker:<6}  total={total}  nan_rows={nan_rows}  usable={clean}")

    print("\nRisk-Adjusted Returns, Sharpe and Calmar")
    print("-" * 45)
    for ticker in ret_panel.columns:
        series  = ret_panel[ticker].dropna()
        ann_ret = np.expm1(series.mean() * 252)
        ann_vol = series.std() * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        max_dd  = panel.xs(ticker, level = "ticker")["drawdown"].min()
        calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        print(f"  {ticker:<6}  sharpe={sharpe:.2f}  calmar={calmar:.2f}")

    print("\nTail Risk, VaR and CVaR at 95 and 99 percent")
    print("-" * 50)
    for ticker in ret_panel.columns:
        series  = ret_panel[ticker].dropna()
        var_95  = series.quantile(0.05)
        var_99  = series.quantile(0.01)
        cvar_95 = series[series <= var_95].mean()
        cvar_99 = series[series <= var_99].mean()
        print(f"  {ticker:<6}  VaR95={var_95:.4f}  CVaR95={cvar_95:.4f}  VaR99={var_99:.4f}  CVaR99={cvar_99:.4f}")


if __name__ == "__main__":
    from data.load_data import load_all
    from data.sanity_check import fix_zero_volume
    from data.build_panel import build_panel
    from features.build_features import apply_features
    raw   = load_all()
    raw   = fix_zero_volume(raw)
    panel = build_panel(raw)
    panel = apply_features(panel)
    run_statistical_checks(panel)
