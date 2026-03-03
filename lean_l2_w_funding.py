"""Build a joined parquet: L2 impact data + settlement funding rate from v3.

For each L2 row, joins the most recent settlement rate (funding_event=True)
at or before that minute, sourced from funding_rates_3m_v3.parquet.
Also computes the funding interval per exchange+coin from settlement timestamps.

Output: raw/l2_impact_funding.parquet
Columns: exchange, coin, minute_utc, impact_10k_buy_quote, impact_10k_sell_quote,
         funding_rate_bps, funding_interval
"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)
L2_PATH = os.path.join(BASE, "raw", "l2_orderbook_1min.parquet")
V3_PATH = os.path.join(BASE, "raw", "funding_rates_3m_v3.parquet")
OUT_PATH = os.path.join(BASE, "raw", "l2_impact_funding.parquet")

# Exchange name mapping: L2 -> v3
EXCH_MAP = {
    "ASTERFINANCE": "Aster",
    "BINANCEFTS": "Binance",
    "BYBIT": "Bybit",
    "HYPERLIQUID": "Hyperliquid",
    "LIGHTER": "Lighter",
    "OKEX": "OKX",
}


def normalize_coin(sym):
    for suffix in ["-USDT-SWAP", "-USDC", "-USD", "USDT"]:
        if sym.endswith(suffix):
            return sym[: -len(suffix)].upper()
    return sym.upper()


# ── Load L2 ─────────────────────────────────────────────────────────────────
print("Loading L2 orderbook...")
l2 = pd.read_parquet(
    L2_PATH,
    columns=["exchange", "symbol", "minute_utc", "impact_10k_buy_quote", "impact_10k_sell_quote"],
)
print(f"  {len(l2):,} rows")

l2["exchange"] = l2["exchange"].map(EXCH_MAP)
l2["coin"] = l2["symbol"].str.upper()
l2["minute_utc"] = pd.to_datetime(l2["minute_utc"], utc=True)

# ── Load settlement rates from v3 ──────────────────────────────────────────
print("Loading v3 settlement rates...")
v3 = pd.read_parquet(V3_PATH)
sr = v3[v3["funding_event"] == True].copy()
del v3
sr["funding_rate_bps"] = sr["funding_rate_bps"] / 100  # raw -> true bps
sr["coin"] = sr["symbol"].apply(normalize_coin)
sr = sr[["timestamp", "exchange", "coin", "funding_rate_bps"]].copy()
sr = sr.sort_values(["exchange", "coin", "timestamp"])
print(f"  {len(sr):,} settlement events")

# ── Compute funding intervals from settlement timestamps ────────────────────
print("\nComputing funding intervals...")
interval_map = {}  # (exchange, coin) -> interval string
interval_changes = []

for (exch, coin), grp in sr.groupby(["exchange", "coin"]):
    ts = grp["timestamp"].sort_values()
    if len(ts) < 2:
        continue
    diffs = ts.diff().dropna()
    hours = diffs.dt.total_seconds() / 3600

    # Check for interval changes
    unique_hours = sorted(hours.unique())
    if len(unique_hours) > 1:
        # Use mode as the primary interval
        mode_h = hours.mode().iloc[0]
        # Flag non-mode intervals as changes
        non_mode = hours[hours != mode_h]
        for idx in non_mode.index:
            pos = ts.index.get_loc(idx)
            interval_changes.append({
                "exchange": exch,
                "coin": coin,
                "at": ts.iloc[pos],
                "prev_interval_h": hours.iloc[pos - 1] if pos > 1 else mode_h,
                "new_interval_h": hours.iloc[pos],
                "mode_h": mode_h,
            })
        interval_map[(exch, coin)] = f"{int(mode_h)}h"
    else:
        interval_map[(exch, coin)] = f"{int(unique_hours[0])}h"

# Report interval changes
if interval_changes:
    cdf = pd.DataFrame(interval_changes)
    # Summarize: which coins have real interval changes (not just jitter)
    real_changes = cdf[cdf["new_interval_h"] != cdf["mode_h"]]
    by_pair = real_changes.groupby(["exchange", "coin", "mode_h"]).agg(
        occurrences=("new_interval_h", "count"),
        intervals_seen=("new_interval_h", lambda x: sorted(x.unique())),
    ).reset_index()
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\n  Coins with non-standard settlement gaps ({len(by_pair)} pairs):")
    print(by_pair.to_string(index=False))
else:
    print("  No interval changes detected -- all coins have consistent intervals.")

# Report interval distribution
print("\nInterval distribution:")
from collections import Counter
interval_counts = Counter(interval_map.values())
for iv, cnt in sorted(interval_counts.items()):
    print(f"  {iv}: {cnt} exchange+coin pairs")

# Add interval to settlement data
sr["funding_interval"] = sr.apply(
    lambda r: interval_map.get((r["exchange"], r["coin"]), None), axis=1
)

# ── Join: asof merge per exchange+coin ──────────────────────────────────────
print("\nJoining (asof merge per exchange+coin)...")

l2 = l2.sort_values(["exchange", "coin", "minute_utc"])

results = []
groups = l2.groupby(["exchange", "coin"])
total = len(groups)
for i, ((exch, coin), l2_grp) in enumerate(groups):
    if (i + 1) % 200 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total}...")

    sr_grp = sr[(sr["exchange"] == exch) & (sr["coin"] == coin)]
    if sr_grp.empty:
        l2_grp = l2_grp.copy()
        l2_grp["funding_rate_bps"] = np.nan
        l2_grp["funding_interval"] = None
        results.append(l2_grp)
        continue

    merged = pd.merge_asof(
        l2_grp,
        sr_grp[["timestamp", "funding_rate_bps", "funding_interval"]],
        left_on="minute_utc",
        right_on="timestamp",
        direction="backward",
    )
    merged.drop(columns=["timestamp"], inplace=True)
    results.append(merged)

print("Concatenating...")
out = pd.concat(results, ignore_index=True)

out = out.drop(columns=["coin"])
out = out.rename(columns={"symbol": "coin"})
out["coin"] = out["coin"].str.upper()

out = out[["exchange", "coin", "minute_utc", "impact_10k_buy_quote", "impact_10k_sell_quote",
           "funding_rate_bps", "funding_interval"]]

print(f"\nOutput: {len(out):,} rows")
print(f"  With funding data: {out['funding_rate_bps'].notna().sum():,}")
print(f"  Without (no settlement match): {out['funding_rate_bps'].isna().sum():,}")

matched_coins = out[out["funding_rate_bps"].notna()].groupby("exchange")["coin"].nunique()
total_coins = out.groupby("exchange")["coin"].nunique()
print("\nCoverage per exchange:")
for exch in sorted(out["exchange"].unique()):
    m = matched_coins.get(exch, 0)
    t = total_coins.get(exch, 0)
    print(f"  {exch}: {m}/{t} coins with funding data")

print(f"\nWriting {OUT_PATH}...")
out.to_parquet(OUT_PATH, index=False)
print("Done.")
