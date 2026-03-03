"""Build grid.json from the v3 parquet (settlement rates at 00/08/16 UTC)."""
import json
import os
import pandas as pd

BASE = os.path.dirname(__file__)
PARQUET = os.path.join(BASE, "raw", "funding_rates_3m_v3.parquet")
OUT_DIR = os.path.join(BASE, "funding", "data")
EXCHANGES = ["Binance", "OKX", "Bybit", "Aster", "Hyperliquid", "Lighter"]

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_coin(sym):
    for suffix in ["-USDT-SWAP", "-USDC", "-USD", "USDT"]:
        if sym.endswith(suffix):
            return sym[: -len(suffix)].upper()
    return sym.upper()


print("Reading parquet...")
df = pd.read_parquet(PARQUET)
df = df[df["funding_event"] == True].copy()
df["funding_rate_bps"] = df["funding_rate_bps"] / 100
df["coin"] = df["symbol"].apply(normalize_coin)
df["ts8"] = df["timestamp"].dt.floor("8h")
df = df[df["timestamp"].dt.hour.isin([0, 8, 16])]

coins = sorted(df["coin"].unique())
timestamps = sorted(df["ts8"].unique())
ts_labels = [t.strftime("%b %d %H:%M") for t in pd.to_datetime(timestamps)]

ts_idx = {t: i for i, t in enumerate(timestamps)}
coin_idx = {c: i for i, c in enumerate(coins)}

rates_by_exch = {}
for exch in EXCHANGES:
    print(f"  {exch}...")
    grid = [[None] * len(coins) for _ in range(len(timestamps))]
    sub = df[df["exchange"] == exch]
    for (ts8, coin), g in sub.groupby(["ts8", "coin"]):
        if ts8 in ts_idx and coin in coin_idx:
            grid[ts_idx[ts8]][coin_idx[coin]] = round(float(g["funding_rate_bps"].iloc[0]), 2)
    rates_by_exch[exch] = grid

grid_obj = {
    "coins": coins,
    "timestamps": ts_labels,
    "exchanges": EXCHANGES,
    "rates": rates_by_exch,
}

with open(os.path.join(OUT_DIR, "grid.json"), "w") as f:
    json.dump(grid_obj, f, separators=(",", ":"))

print(f"Grid: {len(coins)} coins x {len(timestamps)} timestamps x {len(EXCHANGES)} exchanges")

# ── Rebuild manifest from existing per-coin JSON files ──────────────────────
manifest = {}
for fname in sorted(os.listdir(OUT_DIR)):
    if fname.endswith(".json") and fname not in ("manifest.json", "grid.json"):
        coin = fname[:-5]
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        manifest[coin] = sorted(data.keys())

with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

print(f"Manifest: {len(manifest)} coins")
print("Done")
