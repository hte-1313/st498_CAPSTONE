import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

FACTOR_DIR = Path("data") / "processed" / "factors"
DATA_PROC  = Path("data") / "processed"
ART_TAB    = Path("artifacts") / "tables"

FACTOR_COLS = [f"PC{i+1}" for i in range(10)]
FF_FACTORS  = ["mkt_rf","smb","hml","rmw","cma","mom"]

EVAL_MODELS = {
    "Unconditional": [],
    "CAPM":          ["mkt_rf"],
    "FF3":           ["mkt_rf","smb","hml"],
    "FF5":           ["mkt_rf","smb","hml","rmw","cma"],
    "FF5+MOM":       ["mkt_rf","smb","hml","rmw","cma","mom"],
}


def find_best_factor(perf_df):
    return perf_df[perf_df["type"] == "CLIP"].sort_values(
        "sharpe", ascending=False).iloc[0]["factor"]


def build_eval_df(portfolio_results_clip, panel_m, best_pc, split_month):
    test_months    = sorted([
        m for m in panel_m["month"].unique()
        if str(m) >= split_month
    ])
    ff_monthly_avg = (
        panel_m[panel_m["month"].isin(test_months)]
        .groupby("month")[FF_FACTORS + ["rf"]]
        .mean()
        .reset_index()
    )
    best_port = portfolio_results_clip[best_pc][["month","ls_ret"]].copy()
    eval_df   = best_port.merge(ff_monthly_avg, on="month", how="inner")
    eval_df["excess_ls"] = eval_df["ls_ret"] - eval_df["rf"]
    return eval_df


def run_alpha_regressions(eval_df, best_pc):
    rows = []
    for model_name, facs in EVAL_MODELS.items():
        y  = eval_df["excess_ls"]
        Xc = sm.add_constant(
            eval_df[facs] if facs else np.ones(len(y)),
            has_constant="add",
        )
        res = sm.OLS(y, Xc, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append({
            "model":         model_name,
            "alpha_ann_pct": float(res.params["const"]) * 12 * 100,
            "t_alpha_HAC":   float(res.tvalues["const"]),
            "r2":            float(res.rsquared),
            "n":             int(res.nobs),
            "factor":        best_pc,
            "embedding":     "CLIP",
        })
    return pd.DataFrame(rows)


def print_scoring_summary(eval_results, perf_df, best_pc):
    ff5mom  = eval_results[eval_results["model"] == "FF5+MOM"].iloc[0]
    best    = perf_df[(perf_df["type"] == "CLIP") & (perf_df["factor"] == best_pc)].iloc[0]
    sig     = (
        "YES p<0.05"   if abs(ff5mom["t_alpha_HAC"]) >= 1.96  else
        "MARGINAL p<0.10" if abs(ff5mom["t_alpha_HAC"]) >= 1.645 else
        "NO"
    )

    print("=" * 55)
    print("SCORING SUMMARY")
    print("=" * 55)
    print(f"Best visual factor    : {best_pc}")
    print(f"Annualised return     : {best['ret_ann_pct']:.2f}%")
    print(f"Annualised vol        : {best['vol_ann_pct']:.2f}%")
    print(f"Sharpe ratio          : {best['sharpe']:.3f}")
    print(f"Max drawdown          : {best['max_dd_pct']:.2f}%")
    print(f"Calmar ratio          : {best['calmar']:.3f}")
    print(f"Hit rate              : {best['hit_rate']:.3f}")
    print(f"Alpha vs FF5+MOM      : {ff5mom['alpha_ann_pct']:.2f}%")
    print(f"t-stat (HAC)          : {ff5mom['t_alpha_HAC']:.3f}")
    print(f"R squared             : {ff5mom['r2']:.3f}")
    print(f"Significant           : {sig}")
    print("=" * 55)


if __name__ == "__main__":
    from stage3_portfolios import (
        load_merged_panel,
        build_ls_portfolio,
        portfolio_metrics,
        FACTOR_COLS,
    )

    factors_clip = pd.read_parquet(FACTOR_DIR / "clip_pca_factors_20d_month.parquet")
    split_cutoff = factors_clip[factors_clip["split"] == "test"]["date"].min()
    split_month  = pd.Timestamp(split_cutoff).to_period("M").strftime("%Y-%m")

    panel_m = pd.read_parquet(DATA_PROC / "stage0_panel_ff_monthly.parquet")
    panel_m["month"] = panel_m["month"].astype(str)

    merged_clip = load_merged_panel(
        FACTOR_DIR / "clip_pca_factors_20d_month.parquet",
        DATA_PROC  / "stage0_panel_ff_monthly.parquet",
        split_month,
    )

    portfolio_results_clip = {pc: build_ls_portfolio(merged_clip, pc) for pc in FACTOR_COLS}

    rows = []
    for pc in FACTOR_COLS:
        m = portfolio_metrics(portfolio_results_clip[pc])
        m["factor"] = pc
        m["type"]   = "CLIP"
        rows.append(m)

    perf_df = pd.DataFrame(rows)

    best_pc      = find_best_factor(perf_df)
    eval_df      = build_eval_df(portfolio_results_clip, panel_m, best_pc, split_month)
    eval_results = run_alpha_regressions(eval_df, best_pc)

    eval_results.to_csv(ART_TAB / "stage4_visual_factor_alpha.csv", index=False)
    print(eval_results.to_string(index=False))

    print_scoring_summary(eval_results, perf_df, best_pc)
    print("\nStage 4 complete.")
