import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA_RAW  = Path("data") / "raw"
DATA_PROC = Path("data") / "processed"
IMAGE_DIR = DATA_PROC / "chart_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

SPDR_TICKERS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
WINDOW_SIZES = [5, 10, 20, 60]
VERSION      = "V2"
FREQUENCIES  = ["month", "week", "day"]
BAL_START    = "2018-07-01"
BAL_END      = "2025-12-31"

IMAGE_HEIGHT = {5: 32, 10: 48, 20: 64, 60: 96}
IMAGE_WIDTH  = {5: 15, 10: 30, 20: 60, 60: 180}


def load_daily_prices(tickers, data_raw):
    dfs = []
    for t in tickers:
        df = pd.read_parquet(data_raw / f"{t}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values(["ticker","date"]).reset_index(drop=True)


def get_balanced_period_dates(daily, tickers, freq, bal_start, bal_end):
    d = daily[daily["ticker"].isin(tickers)].copy()
    d = d[(d["date"] >= bal_start) & (d["date"] <= bal_end)]

    if freq == "month":
        d["period"] = d["date"].dt.to_period("M")
    elif freq == "week":
        d["period"] = d["date"].dt.to_period("W-FRI")
    elif freq == "day":
        d["period"] = d["date"].dt.to_period("D")

    period_ends = (
        d.groupby(["ticker","period"])["date"]
        .max()
        .reset_index()
    )
    counts    = period_ends.groupby("period")["ticker"].nunique()
    good      = counts[counts == len(tickers)].index
    period_ends = period_ends[period_ends["period"].isin(good)].copy()
    return period_ends.sort_values(["period","ticker"]).reset_index(drop=True)[["ticker","date","period"]]


def make_ohlc_image(price_window, height, width):
    img = np.zeros((height, width), dtype=np.uint8)
    price_min = float(price_window["Low"].min())
    price_max = float(price_window["High"].max())

    if price_max <= price_min:
        return img.flatten()

    price_range = price_max - price_min

    def to_row(price):
        r = int(round((price_max - float(price)) / price_range * (height - 1)))
        return int(np.clip(r, 0, height - 1))

    for i, (_, row) in enumerate(price_window.iterrows()):
        col  = i * 3
        r_o  = to_row(row["Open"])
        r_h  = to_row(row["High"])
        r_l  = to_row(row["Low"])
        r_c  = to_row(row["Close"])

        img[r_o, col] = 255
        img[min(r_h,r_l):max(r_h,r_l)+1, col+1] = 255
        img[r_c, col+2] = 255

    return img.flatten()


def build_price_lookup(daily, tickers):
    lookup = {}
    for ticker in tickers:
        df = (
            daily[daily["ticker"] == ticker]
            [["date","Open","High","Low","Close"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
        lookup[ticker] = df
    return lookup


def generate_all_archives(daily, tickers):
    price_lookup = build_price_lookup(daily, tickers)

    for freq in FREQUENCIES:
        periods_df = get_balanced_period_dates(daily, tickers, freq, BAL_START, BAL_END)

        for ws in WINDOW_SIZES:
            h     = IMAGE_HEIGHT[ws]
            w     = IMAGE_WIDTH[ws]
            fname = f"images_{ws}d_{VERSION}_{freq}"
            out_npz = IMAGE_DIR / f"{fname}.npz"
            out_csv = IMAGE_DIR / f"{fname}_meta.csv"

            if out_npz.exists():
                n = len(np.load(out_npz)["images"])
                print(f"  {fname}: cached ({n:,} images)")
                continue

            print(f"  Generating {fname}...")
            t0 = time.time()

            images_list = []
            meta_rows   = []
            skipped     = 0

            for _, prow in periods_df.iterrows():
                ticker      = prow["ticker"]
                period_date = prow["date"]
                prices      = price_lookup[ticker]
                available   = prices[prices["date"] <= period_date]

                if len(available) < ws:
                    skipped += 1
                    continue

                window   = available.tail(ws).reset_index(drop=True)
                img_flat = make_ohlc_image(window, h, w)
                images_list.append(img_flat)
                meta_rows.append({
                    "ticker":  ticker,
                    "date":    pd.Timestamp(period_date).strftime("%Y-%m-%d"),
                    "window":  ws,
                    "freq":    freq,
                    "version": VERSION,
                })

            images_array = np.array(images_list, dtype=np.uint8)
            meta_df      = pd.DataFrame(meta_rows)

            np.savez_compressed(out_npz, images=images_array)
            meta_df.to_csv(out_csv, index=False)

            elapsed = time.time() - t0
            print(f"    {len(images_list):,} images in {elapsed:.1f}s  ({skipped} skipped)")

    print("Stage 1a complete.")


if __name__ == "__main__":
    daily = load_daily_prices(SPDR_TICKERS, DATA_RAW)
    print(f"Loaded {len(daily):,} rows")
    generate_all_archives(daily, SPDR_TICKERS)
