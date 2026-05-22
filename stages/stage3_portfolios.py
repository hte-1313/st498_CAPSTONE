import numpy as np
import pandas as pd
from pathlib import Path

FACTOR_DIR = Path("data") / "processed" / "factors"
DATA_PROC  = Path("data") / "processed"
ART_TAB    = Path("artifacts") / "tables"
ART_FIG    = Path("artifacts") / "figures"

FACTOR_COLS = [f"PC{i+1}" for i in range(10)]


def load_merged_panel(factors_path, panel_path, split_month):
    factors = pd.read_parquet(factors_path)
    panel   = pd.read_parquet(panel_path)

    factors["month"] = factors["date"].dt.to_period("M").astype(str)
    panel["month"]   = panel["month"].astype(str)

    merged = panel.merge(
        factors[["ticker","month"] + FACTOR_COLS],
        on=["ticker","month"],
        how="inner",
    )
    merged["split"] = merged["month"].apply(
        lambda m: "train" if m < split_month else "test")
    return merged


def build_ls_portfolio(df, factor_col, split="test"):
    rows = []
    for month, g in df[df["split"] == split].groupby("month"):
        g = g.dropna(subset=[factor_col, "ret"])
        if len(g) < 4:
            continue
        g = g.copy()
        g["rank"]  = g[factor_col].rank(pct=True)
        long_ret   = g[g["rank"] >= 0.67]["ret"].mean()
        short_ret  = g[g["rank"] <= 0.33]["ret"].mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            continue
        rows.append({"month": month, "ls_ret": long_ret - short_ret})
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def portfolio_metrics(port_df, ann=12):
    r       = port_df["ls_ret"].dropna()
    ret_ann = r.mean() * ann
    vol_ann = r.std()  * np.sqrt(ann)
    sharpe  = ret_ann / vol_ann if vol_ann > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ret_ann / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "ret_ann_pct": round(ret_ann * 100, 3),
        "vol_ann_pct": round(vol_ann * 100, 3),
        "sharpe":      round(sharpe, 3),
        "max_dd_pct":  round(max_dd * 100, 3),
        "calmar":      round(calmar, 3),
        "hit_rate":    round((r > 0).mean(), 3),
        "n_months":    len(r),
    }


def run_portfolios(merged_clip, merged_pixel):
    portfolio_results_clip  = {pc: build_ls_portfolio(merged_clip,  pc) for pc in FACTOR_COLS}
    portfolio_results_pixel = {pc: build_ls_portfolio(merged_pixel, pc) for pc in FACTOR_COLS}

    rows = []
    for pc in FACTOR_COLS:
        m = portfolio_metrics(portfolio_results_clip[pc])
        m["factor"] = pc
        m["type"]   = "CLIP"
        rows.append(m)

        m = portfolio_metrics(portfolio_results_pixel[pc])
        m["factor"] = pc
        m["type"]   = "Pixel"
        rows.append(m)

    perf_df = pd.DataFrame(rows)[
        ["type","factor","ret_ann_pct","vol_ann_pct","sharpe",
         "max_dd_pct","calmar","hit_rate","n_months"]
    ]
    perf_df.to_csv(ART_TAB / "stage3_portfolio_performance.csv", index=False)
    print("Portfolio performance saved.")
    print(perf_df.to_string(index=False))

    return portfolio_results_clip, portfolio_results_pixel, perf_df


if __name__ == "__main__":
    import joblib

    factors_clip  = pd.read_parquet(FACTOR_DIR / "clip_pca_factors_20d_month.parquet")
    split_cutoff  = factors_clip[factors_clip["split"] == "test"]["date"].min()
    split_month   = pd.Timestamp(split_cutoff).to_period("M").strftime("%Y-%m")

    panel_m = pd.read_parquet(DATA_PROC / "stage0_panel_ff_monthly.parquet")

    merged_clip = load_merged_panel(
        FACTOR_DIR / "clip_pca_factors_20d_month.parquet",
        DATA_PROC  / "stage0_panel_ff_monthly.parquet",
        split_month,
    )
    merged_pixel = load_merged_panel(
        FACTOR_DIR / "pixel_pca_factors_20d_month.parquet",
        DATA_PROC  / "stage0_panel_ff_monthly.parquet",
        split_month,
    )

    portfolio_results_clip, portfolio_results_pixel, perf_df = run_portfolios(merged_clip, merged_pixel)
    print("\nStage 3 complete.")
