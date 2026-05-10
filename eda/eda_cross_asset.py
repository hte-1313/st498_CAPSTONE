# eda/eda_cross_asset.py
# Six cross-asset comparison charts across all 16 ETFs.
# Charts saved to output/plots/
# Covers: correlation, risk-return, cumulative returns,
#         rolling correlation vs VOO, volatility, drawdown.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from config.settings import PLOT_DIR


def run_cross_asset_eda(panel):
    ret_panel   = panel["ret_1d"].unstack("ticker")
    vol_panel   = panel["vol_20"].unstack("ticker")
    dd_panel    = panel["drawdown"].unstack("ticker")

    corr    = ret_panel.corr()
    ann_ret = np.expm1(ret_panel.mean() * 252)
    ann_vol = ret_panel.std() * np.sqrt(252)
    cum_ret = ret_panel.fillna(0).add(1).cumprod()

    fig, ax = plt.subplots(figsize = (14, 10))
    mask    = np.triu(np.ones_like(corr, dtype = bool))
    sns.heatmap(corr, mask = mask, annot = True, fmt = ".2f", cmap = "RdYlGn",
                vmin = -1, vmax = 1, linewidths = 0.5, ax = ax)
    ax.set_title("Pairwise Return Correlation, All ETFs", fontsize = 13)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cross_correlation.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize = (11, 7))
    sc = ax.scatter(ann_vol, ann_ret, s = 120, c = ann_ret, cmap = "RdYlGn",
                    edgecolors = "white", linewidth = 0.8, zorder = 3)
    for ticker in ann_ret.index:
        ax.annotate(ticker, (ann_vol[ticker], ann_ret[ticker]),
                    fontsize = 8, xytext = (5, 4), textcoords = "offset points")
    ax.axhline(0, lw = 0.8, color = "grey", ls = "--")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title("Risk-Return Landscape, All ETFs")
    plt.colorbar(sc, ax = ax, label = "Ann. Return")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "risk_return.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize = (14, 6))
    cum_ret.plot(ax = ax, lw = 1.0, alpha = 0.85)
    ax.set_title("Cumulative Returns, All ETFs (2015 = 1.0)")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc = "upper left", ncol = 4, fontsize = 7)
    ax.axhline(1, lw = 0.7, color = "black", ls = "--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cumulative_returns.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    roll_corr_voo = ret_panel.rolling(126).corr(ret_panel["VOO"]).drop("VOO", axis = 1)
    fig, ax = plt.subplots(figsize = (14, 6))
    roll_corr_voo.plot(ax = ax, lw = 0.9, alpha = 0.8)
    ax.set_title("126-day Rolling Correlation vs VOO")
    ax.set_ylabel("Correlation")
    ax.axhline(0, lw = 0.7, color = "black", ls = "--")
    ax.legend(loc = "lower left", ncol = 4, fontsize = 7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rolling_correlation_voo.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize = (14, 6))
    vol_panel.plot(ax = ax, lw = 0.9, alpha = 0.8)
    ax.set_title("20-day Rolling Annualised Volatility, All ETFs")
    ax.set_ylabel("Ann. Vol")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.legend(loc = "upper left", ncol = 4, fontsize = 7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "volatility_comparison.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize = (14, 6))
    dd_panel.plot(ax = ax, lw = 0.9, alpha = 0.8)
    ax.set_title("Drawdown from Peak, All ETFs")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax = 1))
    ax.legend(loc = "lower left", ncol = 4, fontsize = 7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_comparison.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

    print("Cross-asset EDA complete. 6 charts saved.")


if __name__ == "__main__":
    from data.load_data import load_all
    from data.sanity_check import fix_zero_volume
    from data.build_panel import build_panel
    from features.build_features import apply_features
    raw   = load_all()
    raw   = fix_zero_volume(raw)
    panel = build_panel(raw)
    panel = apply_features(panel)
    run_cross_asset_eda(panel)
