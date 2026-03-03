"""Split parquet into per-coin JSON files for the web app."""
import json
import os
import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "raw", "funding_rates_3m_v3.parquet")
OUT_DIR = os.path.join(BASE, "funding", "data")

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET)
df = df.sort_values(["exchange", "symbol", "timestamp"])


def normalize_coin(sym):
    for suffix in ["-USDT-SWAP", "-USDC", "-USD", "USDT"]:
        if sym.endswith(suffix):
            return sym[: -len(suffix)].upper()
    return sym.upper()


# Build coin -> { exchange: [ [ts, rate, event], ... ] }
coin_data = {}

last_exch = None
for (exch, sym), grp in df.groupby(["exchange", "symbol"]):
    if exch != last_exch:
        print(f"  Processing {exch}...")
        last_exch = exch
    ts = (grp["timestamp"].astype("int64") // 10**6).tolist()
    rates = (grp["funding_rate_bps"] / 100).round(4).tolist()
    events = grp["funding_event"].astype(int).tolist()
    rows = [list(row) for row in zip(ts, rates, events)]

    coin = normalize_coin(sym)
    coin_data.setdefault(coin, {})[exch] = rows

# Write one JSON per coin (skip coins with fewer than 2 exchanges)
coins_to_write = {c: v for c, v in coin_data.items() if len(v) >= 2}
skipped = len(coin_data) - len(coins_to_write)
if skipped:
    print(f"  Skipping {skipped} coins with fewer than 2 exchanges")
total_coins = len(coins_to_write)
for i, coin in enumerate(sorted(coins_to_write), 1):
    if i % 50 == 0 or i == total_coins:
        print(f"  Writing coin {i}/{total_coins}...")
    fname = f"{coin}.json"
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        json.dump(coins_to_write[coin], f, separators=(",", ":"))
    total_rows = sum(len(v) for v in coins_to_write[coin].values())
    exchs = sorted(coins_to_write[coin].keys())
    print(f"  {fname} ({len(exchs)} exchanges, {total_rows} rows)")

# Manifest: coin -> [exchanges]
manifest = {}
for coin in sorted(coins_to_write):
    manifest[coin] = sorted(coins_to_write[coin].keys())

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

print(f"\nDone: {len(coins_to_write)} coin files + manifest.json ({skipped} skipped)")
