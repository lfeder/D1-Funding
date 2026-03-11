"""Build spreads.json from L2 orderbook data.

Reads raw/l2_orderbook_1min_0309.parquet and computes per-exchange, per-coin
bid/offer spread at $10k impact, aggregated to 30-minute buckets.
Only includes coins present on 2+ exchanges.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "raw", "l2_orderbook_1min_0309.parquet")
OUT = os.path.join(BASE, "spreads", "spreads.json")

EXCH_MAP = {
    "aster": "Aster",
    "binance": "Binance",
    "bybit": "Bybit",
    "hyperliquid": "Hyperliquid",
    "lighter": "Lighter",
    "okx": "OKX",
}

os.makedirs(os.path.join(BASE, "spreads"), exist_ok=True)

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

# Round-trip spread in bps
df["spread_bps"] = df["slippage_10k_buy_bps"] + df["slippage_10k_sell_bps"]

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
# Floor to 30-min boundaries
df["bucket"] = df["minute_utc"].dt.floor("30min")

# All 30-min buckets in the global range
all_buckets = pd.date_range(data_start.floor("30min"), data_end.floor("30min"), freq="30min")
n_buckets = len(all_buckets)
bucket_to_idx = {b: i for i, b in enumerate(all_buckets)}
print(f"  {n_buckets} half-hour buckets")

# Timestamps for the viewer (ms since epoch)
bucket_timestamps = [int(b.timestamp() * 1000) for b in all_buckets]

# ── Aggregate per exchange+coin+bucket ──────────────────────────────────────
print("Aggregating spreads...")
agg = df.groupby(["exchange", "coin", "bucket"])["spread_bps"].agg(["median", "max"])
agg.columns = ["median", "max"]
agg = agg.reset_index()

exchanges = sorted(df["exchange"].unique())
spreads_data = {exch: {} for exch in exchanges}

groups = agg.groupby(["exchange", "coin"])
total_groups = len(groups)
print(f"  Processing {total_groups} exchange+coin groups...")

for i, ((exch, coin), grp) in enumerate(groups):
    if (i + 1) % 100 == 0 or (i + 1) == total_groups:
        print(f"    {i+1}/{total_groups}...")

    # Initialize with null (None → null in JSON)
    median_arr = [None] * n_buckets
    max_arr = [None] * n_buckets

    overall_median = round(float(grp["median"].median()), 1)

    for _, row in grp.iterrows():
        idx = bucket_to_idx.get(row["bucket"])
        if idx is not None:
            median_arr[idx] = round(float(row["median"]), 1)
            max_arr[idx] = round(float(row["max"]), 1)

    spreads_data[exch][coin] = {
        "median_bps": overall_median,
        "median": median_arr,
        "max": max_arr,
    }

# ── Exchange-level summary arrays ───────────────────────────────────────────
print("Computing exchange summaries...")
exch_summary = {}
for exch in exchanges:
    coins = spreads_data[exch]
    n_coins = len(coins)
    # For each bucket, take median/max across all coins on that exchange
    median_summary = [None] * n_buckets
    max_summary = [None] * n_buckets
    for bi in range(n_buckets):
        med_vals = [c["median"][bi] for c in coins.values() if c["median"][bi] is not None]
        max_vals = [c["max"][bi] for c in coins.values() if c["max"][bi] is not None]
        if med_vals:
            median_summary[bi] = round(float(np.median(med_vals)), 1)
        if max_vals:
            max_summary[bi] = round(float(np.median(max_vals)), 1)

    overall = round(float(np.median([c["median_bps"] for c in coins.values()])), 1)
    exch_summary[exch] = {
        "n_coins": n_coins,
        "median_bps": overall,
        "median": median_summary,
        "max": max_summary,
    }

# ── Write output ─────────────────────────────────────────────────────────────
output = {
    "exchanges": exchanges,
    "data_start": data_start.isoformat() + "Z",
    "data_end": data_end.isoformat() + "Z",
    "n_buckets": n_buckets,
    "bucket_ms": bucket_timestamps,
    "exchange_summary": exch_summary,
    "spreads": spreads_data,
}

with open(OUT, "w") as f:
    json.dump(output, f, separators=(",", ":"))

fsize = os.path.getsize(OUT)
print(f"\nWrote {OUT} ({fsize / 1024 / 1024:.1f} MB)")
for exch in exchanges:
    s = exch_summary[exch]
    print(f"  {exch}: {s['n_coins']} coins, median spread {s['median_bps']} bps")
print("Done.")
