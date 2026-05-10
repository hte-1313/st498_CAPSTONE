# Statistical Checks — Results

---

## Stationarity

All 16 ETFs came back stationary on daily returns with p=0.0000 across the board. This is exactly what we wanted to see. Returns are safe to model.

---

## Normality

All 16 ETFs failed the Jarque-Bera test with p=0.0000. None of the return series are normally distributed. This confirms what the histograms showed — fat tails are real across every asset class in this universe. Any model that assumes normality will be wrong.

---

## Autocorrelation

15 out of 16 ETFs showed statistically significant autocorrelation. The one exception was BNDX at p=0.0714, which is the only ETF where we cannot reject the null of no autocorrelation. This matters for modelling — most of these assets have return patterns that are not purely random, which means simple OLS may not be the right baseline without accounting for this.

---

## NaN Audit

This is the result that needs attention. Every ETF shows nan_rows=2854 and usable=0. That does not mean all data is unusable — it means at least one column has a NaN somewhere in every row. The culprit is most likely `downside_vol_20`, which produces NaN on any 20-day window where all returns were positive. Before modelling we need to decide which columns are actually required and drop NaN only on those, not across all 46 columns at once.

---

## Sharpe and Calmar

VGT is the clear winner on both metrics — Sharpe of 0.92 and Calmar of 0.63. The broad US market ETFs VOO, VTI and VUG all cluster around Sharpe 0.74-0.78. At the bottom are VDE and BND, both at Sharpe 0.35 and Calmar 0.10, reflecting energy's crash in 2020 and bonds' terrible 2022.

| Best | Sharpe | Calmar |
|---|---|---|
| VGT | 0.92 | 0.63 |
| VOO | 0.78 | 0.41 |
| VUG | 0.78 | 0.46 |

| Worst | Sharpe | Calmar |
|---|---|---|
| VDE | 0.22 | 0.10 |
| BND | 0.35 | 0.10 |
| VNQ | 0.27 | 0.13 |

---

## VaR and CVaR

BND has the tightest tail risk — on a bad day at 95% confidence you lose 0.48%, and the average loss beyond that is 0.75%. VDE has the widest — 2.83% at VaR95 and 7.66% at CVaR99. The bond ETFs and equity ETFs are clearly living in different risk regimes, which is useful context before we run any cross-asset model.

---

## Where we are now

Stationary — yes. Not normal — confirmed. Autocorrelated — mostly yes. NaN audit — needs one fix before modelling. EDA is done. We are ready for baseline analysis.
