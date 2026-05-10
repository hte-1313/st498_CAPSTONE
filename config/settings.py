# config/settings.py
# Project-wide constants. Every other file imports from here.
# Change START / END here to adjust the dataset window globally.

import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TICKERS = [
    "VTI", "VOO", "VT",
    "VUG", "VTV", "VO", "VB",
    "VGT", "VHT", "VFH", "VDE",
    "VEA", "VWO",
    "BND", "BNDX",
    "VNQ",
]

START = "2015-01-01"
END   = "2026-05-08"

OUTPUT_DIR = Path("./output")
PLOT_DIR   = OUTPUT_DIR / "plots"
DATA_DIR   = OUTPUT_DIR / "data"

for _dir in [OUTPUT_DIR, PLOT_DIR, DATA_DIR]:
    _dir.mkdir(parents = True, exist_ok = True)

ETF_NAMES = {
    "VTI":  "US Total Market",
    "VOO":  "S&P 500",
    "VT":   "Total World",
    "VUG":  "Growth",
    "VTV":  "Value",
    "VO":   "Mid-Cap",
    "VB":   "Small-Cap",
    "VGT":  "Information Technology",
    "VHT":  "Health Care",
    "VFH":  "Financials",
    "VDE":  "Energy",
    "VEA":  "Developed Markets ex-US",
    "VWO":  "Emerging Markets",
    "BND":  "US Bond Market",
    "BNDX": "International Bonds",
    "VNQ":  "Real Estate",
}
