"""Split parquet into per-coin JSON files for the web app."""
import json
import os
import pandas as pd

PARQUET = os.path.join(os.path.dirname(__file__), "data", "funding_rates_3m_v2.parquet")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

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

for (exch, sym), grp in df.groupby(["exchange", "symbol"]):
    ts = (grp["timestamp"].astype("int64") // 10**6).tolist()
    rates = (grp["funding_rate_bps"] / 100).round(4).tolist()
    events = grp["funding_event"].astype(int).tolist()
    rows = [list(row) for row in zip(ts, rates, events)]

    coin = normalize_coin(sym)
    coin_data.setdefault(coin, {})[exch] = rows

# Write one JSON per coin
for coin in sorted(coin_data):
    fname = f"{coin}.json"
    with open(os.path.join(OUT_DIR, fname), "w") as f:
        json.dump(coin_data[coin], f, separators=(",", ":"))
    total_rows = sum(len(v) for v in coin_data[coin].values())
    exchs = sorted(coin_data[coin].keys())
    print(f"  {fname} ({len(exchs)} exchanges, {total_rows} rows)")

# Manifest: coin -> [exchanges]
manifest = {}
for coin in sorted(coin_data):
    manifest[coin] = sorted(coin_data[coin].keys())

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

print(f"\nDone: {len(coin_data)} coin files + manifest.json")
