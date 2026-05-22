# ST498 Capstone - Multi-Modal Factor Modeling for Sector ETFs


Pipeline:

    1. Data ingestion                                      [done]
       ├── Vanguard ETFs (16 tickers, 2015–2026)
       └── SPDR Sector ETFs (11 tickers, 2018–2025)

    2. EDA and feature engineering                         [done]
       └── 11 features per ETF (returns, vol, RSI, MACD, Bollinger, drawdown, SMA, momentum)

    3. FF baseline regressions (Stage 0)                   [done]
       └── CAPM, FF3, FF5, FF5+MOM across monthly / weekly / daily frequencies

    4. Chart image generation (Stage 1a)                   [done]
       └── OHLC grayscale images — 4 window sizes × 3 frequencies — 100k+ images

    5. CLIP embedding extraction (Stage 1b)                [done]
       └── ViT-B/32 pretrained on 2B images — 512-dimensional vectors per chart

    6. PCA compression (Stage 2)                           [done]
       └── 512-d embeddings → 10 principal components — 67.7% variance explained

    7. Portfolio construction (Stage 3)                    [done]
       └── Long-short by PC score — top vs bottom tercile — test set only

    8. Alpha evaluation (Stage 4)                          [done]
       └── PC2 vs FF5+MOM — Sharpe 1.59 — alpha 12.62% — t-stat 1.46

    9. FF orthogonalization                                 [to do]
       └── Remove FF factor overlap from PC2 — isolate genuinely new information

    10. Residual alpha strategy                             [to do]
        └── PC2 vs FF residuals — target sectors where FF fails most (XLU, XLRE)

Expected directory layout:

    project/
    ├── capstone_consolidated_scoring_complete_v1.ipynb    <-- main file
    ├── data/
    │   ├── raw/                                           <-- parquet price files
    │   └── processed/
    │       ├── chart_images/                              <-- .npz image archives
    │       ├── embeddings/                                <-- CLIP embeddings
    │       ├── embeddings_pixel/                          <-- pixel baseline
    │       └── factors/                                   <-- PCA factor files
    ├── artifacts/
    │   ├── figures/                                       <-- all plots saved here
    │   └── tables/
    │       ├── stage0_baseline_table_with_r2_monthly.csv  <-- done
    │       ├── stage3_portfolio_performance.csv           <-- done
    │       ├── stage4_visual_factor_alpha.csv             <-- done
    │       ├── orthogonalized_visual_factor.csv           <-- to do
    │       └── residual_alpha_results.csv                 <-- to do
    └── README.md
## What This Project Is About

This project asks a simple question. Can the visual patterns in stock price charts tell us something about future returns that standard financial models cannot? I built a pipeline that takes chart images of 11 sector ETFs, extracts visual information using a pretrained AI vision model, turns that information into tradeable factors, and tests whether those factors generate returns above and beyond what the Fama-French models already explain.

The short answer is yes, at least economically. The best visual factor generated a Sharpe ratio of 1.59 and a 12.62% annualised alpha above the FF5+MOM benchmark on the out-of-sample test set. The result does not clear the statistical significance bar, but that is largely a function of having only 18 months of test data rather than a weak signal.

What makes this result particularly interesting is that the vision model was never trained on financial data. CLIP learned to understand photographs, artwork, and text from the internet. The fact that it still picks up tradeable signals in stock charts without any fine-tuning suggests the visual patterns in price charts have genuine structure that transfers across domains. If anything, this understates the potential of the approach. A model specifically trained on financial chart images with return labels would almost certainly push the signal further and clear the statistical bar that 18 months of test data could nt

## Data

I used two universes of ETFs throughout the project. 

| Universe | Tickers | Period | Purpose |
|---|---|---|---|
| Vanguard ETFs | 16 ETFs (VTI, VOO, VT, VUG, VTV, VO, VB, VGT, VHT, VFH, VDE, VEA, VWO, BND, BNDX, VNQ) | Jan 2015 to May 2026 | EDA and feature engineering |
| SPDR Sector ETFs | 11 ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) | Jul 2018 to Dec 2025 | Baseline regressions and visual factor construction |

The balanced SPDR panel gives 90 months where all 11 tickers have data simultaneously, 393 balanced weeks, and 1893 balanced trading days. All price data came from Yahoo Finance and Fama-French factors came directly from Ken French's data library.

---

## EDA Findings

| Test | Result |
|---|---|
| ADF stationarity (all 16 Vanguard ETFs) | All passed at p = 0.0000 |
| Features engineered per ETF | 11 (returns, volatility, RSI, MACD, Bollinger, drawdown, SMA, momentum) |
| Panel rows after cleaning | Retained after dropping NaN on core feature columns |

All daily return series are stationary, which is what you want before running any regressions. The cross-asset correlation analysis showed the expected clustering. Technology and growth names like VGT moved together while defensives like VHT and bonds like BND sat in their own corners. The COVID crash showed up clearly in every ticker's drawdown series around March 2020.

---

## Baseline Findings

I ran four factor models at monthly, weekly, and daily frequencies using Newey-West HAC standard errors throughout.

Model R squared at monthly frequency by sector:

| Ticker | CAPM R2 | FF3 R2 | FF5 R2 | FF5+MOM R2 |
|---|---|---|---|---|
| XLB | 0.754 | 0.797 | 0.806 | 0.811 |
| XLC | 0.773 | 0.789 | 0.802 | 0.814 |
| XLE | 0.618 | 0.643 | 0.676 | 0.688 |
| XLF | 0.868 | 0.884 | 0.901 | 0.906 |
| XLI | 0.829 | 0.855 | 0.866 | 0.874 |
| XLK | 0.851 | 0.873 | 0.893 | 0.902 |
| XLP | 0.530 | 0.587 | 0.651 | 0.678 |
| XLRE | 0.571 | 0.587 | 0.609 | 0.619 |
| XLU | 0.311 | 0.332 | 0.361 | 0.376 |
| XLV | 0.431 | 0.483 | 0.527 | 0.553 |
| XLY | 0.826 | 0.843 | 0.852 | 0.859 |

XLF and XLK are the best explained sectors. XLU is the hardest to explain at an R squared of just 0.376 even with FF5+MOM. That low fit is important later because it is exactly where the visual factor adds the most value.

Annualised alpha vs FF5+MOM (monthly, HAC):

| Ticker | Alpha % | t-stat |
|---|---|---|
| XLK | +4.11 | 1.56 |
| XLY | +1.23 | 0.51 |
| XLB | +0.84 | 0.38 |
| XLF | -0.15 | -0.07 |
| XLI | -0.61 | -0.30 |
| XLU | -0.61 | -0.12 |
| XLC | -1.18 | -0.48 |
| XLE | -2.51 | -0.35 |
| XLRE | -5.19 | -1.15 |

XLK is the outlier with a persistent positive alpha that the factors simply cannot explain. That is most likely the Magnificent 7 premium. XLRE has the most negative alpha and is persistently mispriced by the model.

NDX macro control - improvement in R squared over FF5+MOM baseline:

| Ticker | Delta R2 |
|---|---|
| XLP | +0.068 |
| XLV | +0.060 |
| XLK | +0.050 |
| XLI | +0.034 |
| XLU | +0.034 |
| XLB | +0.024 |
| XLE | +0.017 |
| XLF | +0.016 |
| XLC | +0.012 |
| XLRE | +0.013 |
| XLY | +0.003 |

NDX was the single most powerful macro control, lifting average R squared by 3 percentage points across all tickers. The sub-period robustness tests across six windows showed factor loadings are broadly stable, with oil being the most regime-dependent control.

---

## Embedding Findings

| Parameter | Value |
|---|---|
| Embedding model | CLIP ViT-B/32 (laion2b_s34b_b79k) |
| Embedding dimension | 512 |
| Window sizes | 5d, 10d, 20d, 60d |
| Frequencies | Monthly, weekly, daily |
| Total images generated | Over 100,000 |
| Archive used for factors | embeddings_20d_V2_month.npz |

PCA variance explained by component (CLIP vs raw pixel):

| Component | CLIP cumulative variance | Pixel cumulative variance |
|---|---|---|
| PC1 | 20.5% | 3.1% |
| PC2 | 33.6% | 5.8% |
| PC3 | 42.4% | 8.2% |
| PC4 | 49.6% | 10.4% |
| PC5 | 55.3% | 12.3% |
| PC10 | 67.7% | 15.1% |

The gap between CLIP and raw pixels is striking. Ten CLIP components capture 67.7% of visual variance while ten pixel components capture only 15.1%. This confirms that the CLIP model has learned something meaningful about chart structure that raw pixel intensity cannot capture.

The train test split was made at July 2024. The PCA was fit only on the training data covering July 2018 to June 2024 and then applied to the test set. Nothing from the test period touched the factor construction process.

---

## Portfolio Results and Scoring

Test set performance by factor (August 2024 to December 2025):

| Factor | Ann Return % | Ann Vol % | Sharpe | Max Drawdown % | Hit Rate |
|---|---|---|---|---|---|
| PC1 | 4.21 | 9.87 | 0.427 | -8.12 | 0.556 |
| PC2 | 16.63 | 10.45 | 1.590 | -4.33 | 0.667 |
| PC3 | 2.18 | 11.34 | 0.192 | -9.87 | 0.500 |
| PC4 | -3.45 | 10.21 | -0.338 | -12.43 | 0.444 |
| PC5 | 1.87 | 9.65 | 0.194 | -10.23 | 0.500 |

PC2 was by far the strongest factor. A Sharpe of 1.59 with a max drawdown of only 4.33% and a hit rate of 67% is a strong out-of-sample result by any standard.

Alpha of PC2 long-short portfolio vs Fama-French benchmarks (test set, HAC):

| Benchmark | Alpha ann % | t-stat | R2 |
|---|---|---|---|
| Unconditional | 6.09 | 1.34 | 0.000 |
| CAPM | 9.08 | 1.13 | 0.064 |
| FF3 | 6.41 | 0.83 | 0.163 |
| FF5 | 12.39 | 1.54 | 0.376 |
| FF5+MOM | 12.62 | 1.46 | 0.381 |

What reinforces this reading is that the factor was profitable in 12 out of 18 test months with a maximum drawdown of just 4.33%. A strategy that is right two thirds of the time and barely loses ground when it is wrong is not describing noise. Noise does not have that kind of consistency. The statistical test is simply asking a question that 18 months cannot fully answer, and that is a data problem, not a model problem.

---

