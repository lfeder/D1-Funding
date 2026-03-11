"""Build bidask.json from L2 orderbook data.

Reads raw/l2_orderbook_1min_0309.parquet and computes per-exchange, per-coin
bid/ask at $10k impact, aggregated to 30-minute buckets.
Three views: median, max, and 8h25% (avg of tightest 25% over forward 8h).
Only includes coins present on 2+ exchanges.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "raw", "l2_orderbook_1min_0309.parquet")
OUT = os.path.join(BASE, "bidask", "bidask.json")

EXCH_MAP = {
    "aster": "Aster",
    "binance": "Binance",
    "bybit": "Bybit",
    "hyperliquid": "Hyperliquid",
    "lighter": "Lighter",
    "okx": "OKX",
}

os.makedirs(os.path.join(BASE, "bidask"), exist_ok=True)

# ── Load & normalize ────────────────────────────────────────────────────────
print("Reading L2 parquet...")
df = pd.read_parquet(
    PARQUET,
    columns=["exchange", "symbol", "minute_utc",
             "slippage_10k_buy_bps", "slippage_10k_sell_bps"],
)
print(f"  {len(df):,} rows")

df["exchange"] = df["exchange"].map(EXCH_MAP)
df["coin"] = df["symbol"]
df["minute_utc"] = pd.to_datetime(df["minute_utc"])

# Round-trip bidask in bps
df["bidask_bps"] = df["slippage_10k_buy_bps"] + df["slippage_10k_sell_bps"]

# ── Filter to coins on 2+ exchanges ─────────────────────────────────────────
coin_exch_count = df.groupby("coin")["exchange"].nunique()
multi_exch_coins = set(coin_exch_count[coin_exch_count >= 2].index)
df = df[df["coin"].isin(multi_exch_coins)]
print(f"  {len(multi_exch_coins)} coins on 2+ exchanges, {len(df):,} rows after filter")

# Global data range
data_start = df["minute_utc"].min()
data_end = df["minute_utc"].max()
print(f"  Data range: {data_start} to {data_end}")

# ── Build 30-minute bucket index ────────────────────────────────────────────
df["bucket"] = df["minute_utc"].dt.floor("30min")

all_buckets = pd.date_range(data_start.floor("30min"), data_end.floor("30min"), freq="30min")
n_buckets = len(all_buckets)
bucket_to_idx = {b: i for i, b in enumerate(all_buckets)}
print(f"  {n_buckets} half-hour buckets")

bucket_timestamps = [int(b.timestamp() * 1000) for b in all_buckets]

# ── Aggregate median/max per exchange+coin+bucket ───────────────────────────
print("Aggregating bidask (median/max)...")
agg = df.groupby(["exchange", "coin", "bucket"])["bidask_bps"].agg(["median", "max"])
agg.columns = ["median", "max"]
agg = agg.reset_index()

# ── Compute 4h25%: for each bucket, forward 4h window at minute level ──────
print("Computing 8h25% (this takes a while)...")
FORWARD_MINUTES = 480  # 8 hours
FORWARD_BUCKETS = FORWARD_MINUTES // 30  # 16 buckets

# Pre-index minute-level data for fast lookup
# For each (exchange, coin), build a Series indexed by minute_utc
df_sorted = df[["exchange", "coin", "minute_utc", "bidask_bps"]].sort_values("minute_utc")

exchanges = sorted(df["exchange"].unique())
bidask_data = {exch: {} for exch in exchanges}

ec_groups = df_sorted.groupby(["exchange", "coin"])
total_groups = len(ec_groups)
print(f"  Processing {total_groups} exchange+coin groups...")

# Pre-build agg lookup
agg_lookup = {}
for _, row in agg.iterrows():
    key = (row["exchange"], row["coin"], row["bucket"])
    agg_lookup[key] = (round(float(row["median"]), 1), round(float(row["max"]), 1))

for gi, ((exch, coin), grp) in enumerate(ec_groups):
    if (gi + 1) % 50 == 0 or (gi + 1) == total_groups:
        print(f"    {gi+1}/{total_groups}...")

    median_arr = [None] * n_buckets
    max_arr = [None] * n_buckets
    p25_arr = [None] * n_buckets

    # Fill median/max from agg lookup
    overall_medians = []
    for bi, bucket in enumerate(all_buckets):
        key = (exch, coin, bucket)
        if key in agg_lookup:
            median_arr[bi], max_arr[bi] = agg_lookup[key]
            overall_medians.append(median_arr[bi])

    overall_median = round(float(np.median(overall_medians)), 1) if overall_medians else None

    # For 4h25%: get minute-level values as sorted array by minute
    minute_vals = grp[["minute_utc", "bidask_bps"]].copy()
    minute_vals = minute_vals.set_index("minute_utc").sort_index()

    for bi, bucket in enumerate(all_buckets):
        window_start = bucket
        window_end = bucket + pd.Timedelta(hours=8)
        window = minute_vals.loc[window_start:window_end - pd.Timedelta(minutes=1), "bidask_bps"]
        if len(window) == 0:
            continue
        # p25 threshold, then average values at or below it
        p25_val = np.percentile(window.values, 25)
        tight = window.values[window.values <= p25_val]
        p25_arr[bi] = round(float(tight.mean()), 1)

    overall_max = round(float(np.median([v for v in max_arr if v is not None])), 1) if any(v is not None for v in max_arr) else None
    overall_p25 = round(float(np.median([v for v in p25_arr if v is not None])), 1) if any(v is not None for v in p25_arr) else None

    bidask_data[exch][coin] = {
        "median_bps": overall_median,
        "max_bps": overall_max,
        "p25_8h_bps": overall_p25,
        "median": median_arr,
        "max": max_arr,
        "p25_8h": p25_arr,
    }

# ── Exchange-level summary arrays ───────────────────────────────────────────
print("Computing exchange summaries...")
exch_summary = {}
for exch in exchanges:
    coins = bidask_data[exch]
    n_coins = len(coins)
    median_summary = [None] * n_buckets
    max_summary = [None] * n_buckets
    p25_summary = [None] * n_buckets
    for bi in range(n_buckets):
        med_vals = [c["median"][bi] for c in coins.values() if c["median"][bi] is not None]
        max_vals = [c["max"][bi] for c in coins.values() if c["max"][bi] is not None]
        p25_vals = [c["p25_8h"][bi] for c in coins.values() if c["p25_8h"][bi] is not None]
        if med_vals:
            median_summary[bi] = round(float(np.median(med_vals)), 1)
        if max_vals:
            max_summary[bi] = round(float(np.median(max_vals)), 1)
        if p25_vals:
            p25_summary[bi] = round(float(np.median(p25_vals)), 1)

    overall_med = round(float(np.median([c["median_bps"] for c in coins.values() if c["median_bps"] is not None])), 1)
    overall_max = round(float(np.median([c["max_bps"] for c in coins.values() if c["max_bps"] is not None])), 1)
    overall_p25 = round(float(np.median([c["p25_8h_bps"] for c in coins.values() if c["p25_8h_bps"] is not None])), 1)
    exch_summary[exch] = {
        "n_coins": n_coins,
        "median_bps": overall_med,
        "max_bps": overall_max,
        "p25_8h_bps": overall_p25,
        "median": median_summary,
        "max": max_summary,
        "p25_8h": p25_summary,
    }

# ── Write output ─────────────────────────────────────────────────────────────
output = {
    "exchanges": exchanges,
    "data_start": data_start.isoformat() + "Z",
    "data_end": data_end.isoformat() + "Z",
    "n_buckets": n_buckets,
    "bucket_ms": bucket_timestamps,
    "exchange_summary": exch_summary,
    "bidask": bidask_data,
}

with open(OUT, "w") as f:
    json.dump(output, f, separators=(",", ":"))

fsize = os.path.getsize(OUT)
print(f"\nWrote {OUT} ({fsize / 1024 / 1024:.1f} MB)")
for exch in exchanges:
    s = exch_summary[exch]
    print(f"  {exch}: {s['n_coins']} coins, median bidask {s['median_bps']} bps")
print("Done.")
