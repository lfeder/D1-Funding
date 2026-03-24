"""Build gaps.json from L2 orderbook data.

Reads raw/l2_orderbook_1min.parquet and computes per-exchange, per-coin
coverage stats and gap records. Only includes coins present on 2+ exchanges.
"""
import json
import os

import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "raw", "l2_orderbook_1min_0323.parquet")
OUT = os.path.join(BASE, "gaps", "gaps.json")

EXCH_MAP = {
    "ASTERFINANCE": "Aster",
    "BINANCEFTS": "Binance",
    "BYBIT": "Bybit",
    "HYPERLIQUID": "Hyperliquid",
    "LIGHTER": "Lighter",
    "OKEX": "OKX",
}

os.makedirs(os.path.join(BASE, "gaps"), exist_ok=True)

# ── Load & normalize ────────────────────────────────────────────────────────
print("Reading L2 parquet...")
df = pd.read_parquet(PARQUET, columns=["exchange", "symbol", "minute_utc"])
print(f"  {len(df):,} rows")

df["exchange"] = df["exchange"].map(EXCH_MAP)
df = df.dropna(subset=["exchange"])
df["coin"] = df["symbol"]  # already normalized in L2 data
df["minute_utc"] = pd.to_datetime(df["minute_utc"])

# ── Filter to coins on 2+ exchanges ─────────────────────────────────────────
coin_exch_count = df.groupby("coin")["exchange"].nunique()
multi_exch_coins = set(coin_exch_count[coin_exch_count >= 2].index)
df = df[df["coin"].isin(multi_exch_coins)]
print(f"  {len(multi_exch_coins)} coins on 2+ exchanges, {len(df):,} rows after filter")

# Global data range
data_start = df["minute_utc"].min()
data_end = df["minute_utc"].max()
all_minutes = pd.date_range(data_start, data_end, freq="min")
total_minutes = len(all_minutes)
print(f"  Data range: {data_start} to {data_end} ({total_minutes:,} minutes)")

# ── Compute gaps per exchange+coin ───────────────────────────────────────────
exchanges = sorted(df["exchange"].unique())
groups = df.groupby(["exchange", "coin"])
total_groups = len(groups)
print(f"  Processing {total_groups:,} exchange+coin groups...")

gaps_data = {exch: {} for exch in exchanges}

for i, ((exch, coin), grp) in enumerate(groups):
    if (i + 1) % 500 == 0 or (i + 1) == total_groups:
        print(f"    {i+1}/{total_groups}...")

    minutes = grp["minute_utc"].sort_values().drop_duplicates()
    first_tick = minutes.iloc[0]
    last_tick = minutes.iloc[-1]

    # Coverage % uses global data range as denominator
    actual_minutes = len(minutes)
    missing_minutes = total_minutes - actual_minutes
    coverage_pct = round(100 * actual_minutes / total_minutes, 2)

    # Find gaps: consecutive minute differences > 1 min
    minute_vals = minutes.values
    diffs_ns = pd.Series(minute_vals[1:]).values.astype("int64") - pd.Series(minute_vals[:-1]).values.astype("int64")
    one_min_ns = 60_000_000_000  # 1 minute in nanoseconds

    gap_list = []
    for j in range(len(diffs_ns)):
        if diffs_ns[j] > one_min_ns:
            gap_start = pd.Timestamp(minute_vals[j])
            gap_end = pd.Timestamp(minute_vals[j + 1])
            gap_missing = int(diffs_ns[j] / one_min_ns) - 1
            gap_list.append([
                int(gap_start.timestamp() * 1000),
                int(gap_end.timestamp() * 1000),
                gap_missing,
            ])

    gaps_data[exch][coin] = {
        "coverage_pct": coverage_pct,
        "first_tick": first_tick.isoformat() + "Z",
        "last_tick": last_tick.isoformat() + "Z",
        "total_minutes": total_minutes,
        "missing_minutes": missing_minutes,
        "gaps": gap_list,
    }

# ── Write output ─────────────────────────────────────────────────────────────
output = {
    "exchanges": exchanges,
    "data_start": data_start.isoformat() + "Z",
    "data_end": data_end.isoformat() + "Z",
    "total_minutes": total_minutes,
    "gaps": gaps_data,
}

with open(OUT, "w") as f:
    json.dump(output, f, separators=(",", ":"))

# Summary
for exch in exchanges:
    coins = gaps_data[exch]
    n_coins = len(coins)
    n_with_gaps = sum(1 for c in coins.values() if c["missing_minutes"] > 0)
    avg_cov = sum(c["coverage_pct"] for c in coins.values()) / n_coins if n_coins else 0
    print(f"  {exch}: {n_coins} coins, {n_with_gaps} with gaps, avg coverage {avg_cov:.1f}%")

print(f"\nWrote {OUT}")
print("Done.")
