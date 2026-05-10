# eda/eda_individual.py
# Produces a 7-panel chart and summary statistics table for each ETF.
# Charts saved to output/plots/{ticker}_eda.png
# Panels: price + SMAs + Bollinger, drawdown, return distribution,
#         rolling volatility, MACD, RSI, volume z-score.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from config.settings import PLOT_DIR


def eda_individual(panel, ticker):
    df = panel.xs(ticker, level = "ticker").copy()

    fig, axes = plt.subplots(7, 1, figsize = (14, 28))
    fig.suptitle(f"{ticker}  Individual EDA  {df.index[0].date()} to {df.index[-1].date()}",
                 fontsize = 14, fontweight = "bold", y = 1.01)
    fig.tight_layout(pad = 3.0)

    ax = axes[0]
    ax.plot(df.index, df["adj_close"], lw = 1.2, color = "#1f77b4", label = "Adj Close")
    ax.plot(df.index, df["sma_20"],    lw = 0.8, color = "#ff7f0e", alpha = 0.8, label = "SMA 20")
    ax.plot(df.index, df["sma_60"],    lw = 0.8, color = "#2ca02c", alpha = 0.8, label = "SMA 60")
    ax.plot(df.index, df["sma_200"],   lw = 0.8, color = "#d62728", alpha = 0.8, label = "SMA 200")
    ax.fill_between(df.index, df["bb_low"], df["bb_up"], alpha = 0.08, color = "#1f77b4", label = "BB 2sd")
    ax.set_title("Price, Moving Averages, Bollinger Bands")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc = "upper left", ncol = 5)

    ax = axes[1]
    ax.fill_between(df.index, df["drawdown"], 0, color = "#d62728", alpha = 0.5, label = "Drawdown")
    ax.plot(df.index, df["max_drawdown_252"], lw = 0.8, color = "#8c0c0c", ls = "--", label = "Max DD 252d")
    ax.set_title("Drawdown from Peak")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.legend(loc = "lower left")

    ax  = axes[2]
    ret = df["ret_1d"].dropna()
    ax.hist(ret, bins = 80, color = "#1f77b4", edgecolor = "white", linewidth = 0.2, density = True, alpha = 0.7)
    x   = np.linspace(ret.min(), ret.max(), 300)
    ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), color = "#d62728", lw = 1.2, label = "Normal fit")
    ax.axvline(ret.mean(),         color = "#ff7f0e", lw = 1.2, ls = "--", label = f"Mean {ret.mean():.4f}")
    ax.axvline(ret.quantile(0.05), color = "#2ca02c", lw = 1.0, ls = ":",  label = f"5th pct {ret.quantile(0.05):.4f}")
    ax.set_title(f"Daily Return Distribution  skew={ret.skew():.2f}  kurt={ret.kurt():.2f}")
    ax.set_xlabel("Log Return")
    ax.legend()

    ax = axes[3]
    ax.plot(df.index, df["vol_20"],          lw = 0.9, color = "#ff7f0e", label = "Vol 20d")
    ax.plot(df.index, df["vol_60"],          lw = 0.9, color = "#2ca02c", label = "Vol 60d")
    ax.plot(df.index, df["downside_vol_20"], lw = 0.9, color = "#d62728", ls = "--", label = "Downside Vol 20d")
    ax.set_title("Rolling Annualized Volatility")
    ax.set_ylabel("Ann. Vol")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.legend(loc = "upper left", ncol = 3)

    ax = axes[4]
    ax.plot(df.index, df["macd"],        lw = 0.9, color = "#1f77b4", label = "MACD")
    ax.plot(df.index, df["macd_signal"], lw = 0.9, color = "#ff7f0e", label = "Signal")
    ax.bar(df.index, df["macd_hist"], color = np.where(df["macd_hist"] >= 0, "#2ca02c", "#d62728"), alpha = 0.4, width = 1)
    ax.axhline(0, color = "black", lw = 0.6)
    ax.set_title("MACD")
    ax.legend(loc = "upper left")

    ax = axes[5]
    ax.plot(df.index, df["rsi_14"], lw = 0.9, color = "#9467bd")
    ax.axhline(70, color = "#d62728", lw = 0.8, ls = "--", label = "Overbought 70")
    ax.axhline(30, color = "#2ca02c", lw = 0.8, ls = "--", label = "Oversold 30")
    ax.fill_between(df.index, df["rsi_14"], 70, where = df["rsi_14"] >= 70, color = "#d62728", alpha = 0.15)
    ax.fill_between(df.index, df["rsi_14"], 30, where = df["rsi_14"] <= 30, color = "#2ca02c", alpha = 0.15)
    ax.set_ylim(0, 100)
    ax.set_title("RSI 14-day")
    ax.legend(loc = "upper left", ncol = 2)

    ax     = axes[6]
    vz     = df["vol_z_60"].fillna(0)
    colors = np.where(vz > 2, "#d62728", "#aec7e8")
    ax.bar(df.index, vz, color = colors, alpha = 0.7, width = 1)
    ax.axhline(2,  color = "#d62728", lw = 0.8, ls = "--", label = "+2sd abnormal")
    ax.axhline(-2, color = "#d62728", lw = 0.8, ls = "--")
    ax.set_title("Volume Z-Score 60d")
    ax.set_ylabel("Z-score")
    ax.legend(loc = "upper left")

    plt.savefig(PLOT_DIR / f"{ticker}_eda.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    cols   = ["adj_close", "ret_1d", "vol_20", "drawdown", "rsi_14", "bb_pctb", "macd", "vol_z_60"]
    stats_ = df[cols].describe(percentiles = [.05, .25, .5, .75, .95]).T
    stats_["skew"] = df[cols].skew()
    stats_["kurt"] = df[cols].kurt()
    print(f"\n{ticker}  Summary Stats")
    print(stats_.to_string())
    print()


def run_all(panel):
    for ticker in panel.index.get_level_values("ticker").unique():
        eda_individual(panel, ticker)


if __name__ == "__main__":
    from data.load_data import load_all
    from data.sanity_check import fix_zero_volume
    from data.build_panel import build_panel
    from features.build_features import apply_features
    raw   = load_all()
    raw   = fix_zero_volume(raw)
    panel = build_panel(raw)
    panel = apply_features(panel)
    run_all(panel)
