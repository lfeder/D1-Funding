"""Extract daily_ohlcv_all.parquet from zip → volume.json for the viewer.

Uses date-index arrays instead of {date: vol} maps to cut JSON size ~60%.
Format: volumes[i] corresponds to dates[i], 0 for missing days.
"""
import json, zipfile, io
from pathlib import Path

ZIP = Path(__file__).resolve().parent.parent / "raw" / "daily_ohlcv_download.zip"
OUT = Path(__file__).resolve().parent / "volume.json"

with zipfile.ZipFile(ZIP) as z:
    with z.open("daily_ohlcv_all.parquet") as f:
        import pandas as pd
        df = pd.read_parquet(io.BytesIO(f.read()))

df = df[["exchange", "base_symbol", "date", "quote_volume_usd"]].copy()
df.rename(columns={"quote_volume_usd": "vol"}, inplace=True)
df = df.groupby(["exchange", "base_symbol", "date"], as_index=False)["vol"].sum()

dates = sorted(df["date"].unique().tolist())
date_idx = {d: i for i, d in enumerate(dates)}
n = len(dates)
exchanges = sorted(df["exchange"].unique().tolist())

def to_array(vol_dict):
    """Convert {date: vol} to positional array aligned with dates[]."""
    arr = [0] * n
    for d, v in vol_dict.items():
        arr[date_idx[d]] = int(v)
    return arr

# Exchange-level daily totals
exch_daily = {}
for ex in exchanges:
    sub = df[df["exchange"] == ex].groupby("date")["vol"].sum()
    exch_daily[ex] = to_array(dict(sub))

# Per-coin, per-exchange daily volumes
coin_exch = {}
for (sym, ex), grp in df.groupby(["base_symbol", "exchange"]):
    if sym not in coin_exch:
        coin_exch[sym] = {}
    coin_exch[sym][ex] = to_array({r["date"]: r["vol"] for _, r in grp.iterrows()})

# Build coins list sorted by total volume desc
coins = []
for sym, exmap in coin_exch.items():
    total = sum(sum(a) for a in exmap.values())
    coins.append({"s": sym, "t": int(total), "e": exmap})
coins.sort(key=lambda c: c["t"], reverse=True)

out = {
    "dates": dates,
    "exchanges": exchanges,
    "exchDaily": exch_daily,
    "coins": coins,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))

import os
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT} — {len(dates)} dates, {len(exchanges)} exchanges, {len(coins)} coins, {sz:.1f} MB")
