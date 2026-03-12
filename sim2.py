"""Narrow-to-wide basis scalper with spread normalization exit.

Entry: basis transitions from narrow (<10bp) to wide (>threshold) for 2 bars.
Exit:  basis converges (<10bp) AND both leg spreads below recent median.
No stop-loss, no timeout. Includes funding P&L.

Usage:
    python sim2.py
    python sim2.py --wide-bp 50
    python sim2.py --wide-bp 25
"""

import argparse
import numpy as np
import pandas as pd
import time
from itertools import combinations
from datetime import datetime, timezone

PARQUET_PATH = r"raw\impact_and_funding_0309.parquet"
L2_PATH = r"raw\l2_orderbook_1min_0309.parquet"
FUNDING_PATH = r"raw\funding_rates_3m_60coins_0311.parquet"
NARROW_THRESHOLD = 10  # bp for both entry reset and exit convergence

FEES_BP = {
    "Aster": 4.0, "Binance": 2.0, "Bybit": 2.0,
    "Hyperliquid": 3.5, "Lighter": 0.0, "OKX": 2.0,
}
ABBREV = {
    "Aster": "AS", "Binance": "BN", "Bybit": "BY",
    "Hyperliquid": "HL", "Lighter": "LT", "OKX": "OK",
}


DEX_EXCHANGES = {"Aster", "Hyperliquid", "Lighter"}
CEX_EXCHANGES = {"Binance", "Bybit", "OKX"}

def load_data(size_k=10, exchange_filter=None):
    """Load impact data. size_k=10 uses buy_10k/sell_10k, size_k=100 computes from L2.
    exchange_filter: None=all, 'dex'=DEX only, 'cex'=CEX only."""
    paper_size = size_k * 1000
    t0 = time.perf_counter()

    if size_k == 10:
        print(f"Loading {PARQUET_PATH} (10k impacts) ...")
        df = pd.read_parquet(PARQUET_PATH, columns=["exchange", "coin", "minute_utc", "buy_10k", "sell_10k"])
        df["minute_utc"] = pd.to_datetime(df["minute_utc"], utc=True)
        df.rename(columns={"buy_10k": "buy", "sell_10k": "sell"}, inplace=True)
    else:
        print(f"Loading {L2_PATH} (100k impacts) ...")
        l2 = pd.read_parquet(L2_PATH, columns=[
            "exchange", "symbol", "minute_utc", "mid",
            "slippage_100k_buy_bps", "slippage_100k_sell_bps",
        ])
        EXCH_MAP = {
            "aster": "Aster", "binance": "Binance", "bybit": "Bybit",
            "hyperliquid": "Hyperliquid", "lighter": "Lighter", "okx": "OKX",
        }
        l2["exchange"] = l2["exchange"].map(EXCH_MAP)
        l2["coin"] = l2["symbol"]
        l2["minute_utc"] = pd.to_datetime(l2["minute_utc"], utc=True)
        l2["buy"] = l2["mid"] * (1 + l2["slippage_100k_buy_bps"] / 10000)
        l2["sell"] = l2["mid"] * (1 - l2["slippage_100k_sell_bps"] / 10000)
        df = l2[["exchange", "coin", "minute_utc", "buy", "sell"]].copy()

    if exchange_filter == "dex":
        df = df[df["exchange"].isin(DEX_EXCHANGES)]
        print(f"  Filtered to DEX: {sorted(DEX_EXCHANGES)}")
    elif exchange_filter == "cex":
        df = df[df["exchange"].isin(CEX_EXCHANGES)]
        print(f"  Filtered to CEX: {sorted(CEX_EXCHANGES)}")

    df["epoch_min"] = (df["minute_utc"].astype(np.int64) // 60_000_000_000).astype(np.int64)

    exchanges = sorted(df["exchange"].unique())
    coins = sorted(df["coin"].unique())
    n_ex = len(exchanges)
    ex_to_idx = {e: i for i, e in enumerate(exchanges)}
    coin_to_idx = {c: i for i, c in enumerate(coins)}
    unique_mins = np.sort(df["epoch_min"].unique())
    n_ticks = len(unique_mins)
    min_to_tick = {m: i for i, m in enumerate(unique_mins)}

    tick_idx = np.array([min_to_tick[m] for m in df["epoch_min"].values], dtype=np.int32)
    coin_idx = np.array([coin_to_idx[c] for c in df["coin"].values], dtype=np.int32)
    ex_idx = np.array([ex_to_idx[e] for e in df["exchange"].values], dtype=np.int32)

    buy_t = np.full((n_ticks, len(coins), n_ex), np.nan)
    sell_t = np.full((n_ticks, len(coins), n_ex), np.nan)
    buy_t[tick_idx, coin_idx, ex_idx] = df["buy"].values
    sell_t[tick_idx, coin_idx, ex_idx] = df["sell"].values

    # Pairs (both directions)
    pair_indices = []
    pair_names = []
    for a, b in combinations(range(n_ex), 2):
        pair_indices.append((a, b))
        pair_names.append((exchanges[a], exchanges[b]))
        pair_indices.append((b, a))
        pair_names.append((exchanges[b], exchanges[a]))

    fees_arr = np.array([FEES_BP.get(e, 0) for e in exchanges])
    pair_fee_ow = np.array([fees_arr[l] + fees_arr[s] for l, s in pair_indices])

    # Pre-compute entry_basis and per-leg spreads for all ticks
    print("  Pre-computing basis and spreads ...")
    n_pairs = len(pair_indices)

    entry_basis = np.full((n_ticks, len(coins), n_pairs), np.nan)
    exit_basis = np.full((n_ticks, len(coins), n_pairs), np.nan)
    leg_spread = buy_t - sell_t  # ask - bid

    # Median spread per (coin, exchange) in bp — used for entry score adjustment
    print("  Computing median spreads per coin/exchange ...")
    with np.errstate(invalid="ignore", divide="ignore"):
        mid_all = (buy_t + sell_t) * 0.5
        spread_bp_all = (buy_t - sell_t) / mid_all * 10000  # (n_ticks, n_coins, n_ex)
    median_spread = np.nanmedian(spread_bp_all, axis=0)  # (n_coins, n_ex)
    median_spread = np.where(np.isnan(median_spread), 0.0, median_spread)

    entry_score = np.full((n_ticks, len(coins), n_pairs), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        for p, (l_idx, s_idx) in enumerate(pair_indices):
            buy_l = buy_t[:, :, l_idx]
            sell_l = sell_t[:, :, l_idx]
            buy_s = buy_t[:, :, s_idx]
            sell_s = sell_t[:, :, s_idx]
            valid = (buy_l > 0) & (sell_s > 0) & (sell_l > 0) & (buy_s > 0)
            fee = pair_fee_ow[p]

            mid_e = (buy_l + sell_s) * 0.5
            eb = (sell_s - buy_l) / mid_e * 10000 - fee
            entry_basis[:, :, p] = np.where(valid & (mid_e > 0), eb, np.nan)

            # entry_score = entry_basis - 0.5 * median_spread_long - 0.5 * median_spread_short
            penalty = 1.0 * median_spread[:, l_idx] + 1.0 * median_spread[:, s_idx]  # (n_coins,)
            es = eb - penalty[np.newaxis, :]  # broadcast (n_ticks, n_coins)
            entry_score[:, :, p] = np.where(valid & (mid_e > 0), es, np.nan)

            mid_x = (buy_s + sell_l) * 0.5
            xb = (buy_s - sell_l) / mid_x * 10000 + fee
            exit_basis[:, :, p] = np.where(valid & (mid_x > 0), xb, np.nan)

    # Load funding events
    print(f"  Loading funding rates ...")
    fdf = pd.read_parquet(FUNDING_PATH)
    fdf = fdf[fdf["funding_event"] == 1].copy()
    fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
    # Map symbol to coin (strip USDT suffix)
    fdf["coin"] = fdf["symbol"].str.replace("USDT$", "", regex=True)
    # Keep only coins/exchanges we care about
    fdf = fdf[fdf["coin"].isin(coins) & fdf["exchange"].isin(exchanges)]
    fdf["epoch_sec"] = (fdf["timestamp"].astype(np.int64) // 1_000_000_000).astype(np.int64)

    # Build lookup: (exchange, coin) -> sorted arrays of (epoch_sec, rate_bps)
    funding_events = {}
    for (ex, coin), grp in fdf.groupby(["exchange", "coin"]):
        grp = grp.sort_values("epoch_sec")
        funding_events[(ex, coin)] = (
            grp["epoch_sec"].values.astype(np.int64),
            grp["funding_rate_bps"].values.astype(np.float64),
        )

    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    print(f"  {n_ticks:,} ticks, {len(coins)} coins, {len(exchanges)} exchanges, {n_pairs} pairs")
    print(f"  {len(fdf):,} funding events loaded")

    return {
        "buy_t": buy_t, "sell_t": sell_t,
        "entry_basis": entry_basis, "entry_score": entry_score,
        "exit_basis": exit_basis, "leg_spread": leg_spread,
        "unique_mins": unique_mins, "n_ticks": n_ticks,
        "coins": coins, "exchanges": exchanges,
        "pair_indices": pair_indices, "pair_names": pair_names,
        "pair_fee_ow": pair_fee_ow,
        "funding_events": funding_events,
        "paper_size": paper_size,
    }


def calc_funding_pl(funding_events, exchange, coin, entry_epoch, exit_epoch, is_long):
    """Sum funding payments between entry and exit.
    Long pays funding (negative P&L when rate > 0).
    Short receives funding (positive P&L when rate > 0).
    Returns P&L in bps of notional.
    """
    key = (exchange, coin)
    if key not in funding_events:
        return 0.0
    times, rates = funding_events[key]
    mask = (times > entry_epoch) & (times <= exit_epoch)
    if not mask.any():
        return 0.0
    total_bps = rates[mask].sum()
    # Long pays, short receives
    return -total_bps if is_long else total_bps


def run_sim(data, wide_threshold, wait_entry=True, wait_exit=False, max_positions=10, exit_threshold=None, quiet=False):
    buy_t = data["buy_t"]
    sell_t = data["sell_t"]
    entry_basis = data["entry_basis"]
    entry_score = data["entry_score"]
    exit_basis = data["exit_basis"]
    leg_spread = data["leg_spread"]
    unique_mins = data["unique_mins"]
    n_ticks = data["n_ticks"]
    coins = data["coins"]
    exchanges = data["exchanges"]
    pair_indices = data["pair_indices"]
    pair_names = data["pair_names"]
    pair_fee_ow = data["pair_fee_ow"]
    funding_events = data["funding_events"]
    paper_size = data["paper_size"]
    n_coins = len(coins)
    n_pairs = len(pair_indices)
    exit_thresh = exit_threshold if exit_threshold is not None else NARROW_THRESHOLD

    # State: -1=unknown, 0=wide_1tick, 1=narrow, 2=wide_confirmed
    state = np.full((n_coins, n_pairs), -1, dtype=np.int8)
    # Track exit confirmation: how many consecutive bars exit_basis < threshold
    exit_confirm = {}  # (coin_idx, pair_idx) -> count
    max_concurrent = 0
    capacity_blocked = 0
    coin_blocked = 0

    active = []
    trade_log = []

    if not quiet:
        print(f"\nSimulating {n_ticks:,} ticks (wide={wide_threshold}bp) ...")
    t0 = time.perf_counter()
    last_report = t0

    for t in range(n_ticks):
        tick_min = int(unique_mins[t])

        # ── 1. Check exits ───────────────────────────────────────────
        still_active = []
        for pos in active:
            c_idx = pos["coin_idx"]
            p_idx = pos["pair_idx"]
            l_idx = pos["l_idx"]
            s_idx = pos["s_idx"]

            xb = exit_basis[t, c_idx, p_idx]
            if np.isnan(xb):
                still_active.append(pos)
                continue

            key = (c_idx, p_idx)
            if xb < exit_thresh:
                exit_confirm[key] = exit_confirm.get(key, 0) + 1
            else:
                exit_confirm[key] = 0

            exit_bars_needed = 2 if wait_exit else 1
            if exit_confirm.get(key, 0) >= exit_bars_needed:
                sell_l = sell_t[t, c_idx, l_idx]
                buy_s = buy_t[t, c_idx, s_idx]

                if np.isnan(sell_l) or np.isnan(buy_s) or sell_l <= 0 or buy_s <= 0:
                    still_active.append(pos)
                    continue

                mid = (pos["entry_buy_l"] + pos["entry_sell_s"]) * 0.5
                qty = paper_size / mid

                pl_long = (sell_l - pos["entry_buy_l"]) * qty
                pl_short = (pos["entry_sell_s"] - buy_s) * qty
                fee_entry = pos["entry_fee"] / 10000 * paper_size
                fee_exit = pair_fee_ow[p_idx] / 10000 * paper_size
                pl_spread = pl_long + pl_short - fee_entry - fee_exit

                # Funding P&L
                entry_epoch = pos["entry_tick"] * 60
                exit_epoch = tick_min * 60
                l_ex = pair_names[p_idx][0]
                s_ex = pair_names[p_idx][1]
                coin = coins[c_idx]

                fund_l_bp = calc_funding_pl(funding_events, l_ex, coin, entry_epoch, exit_epoch, is_long=True)
                fund_s_bp = calc_funding_pl(funding_events, s_ex, coin, entry_epoch, exit_epoch, is_long=False)
                funding_bp = fund_l_bp + fund_s_bp
                funding_usd = funding_bp / 10000 * paper_size

                pl_total = pl_spread + funding_usd

                l_name = ABBREV[l_ex]
                s_name = ABBREV[s_ex]
                exit_dt = datetime.fromtimestamp(exit_epoch, tz=timezone.utc)

                trade_log.append({
                    "coin": coin,
                    "pair": f"{l_name}/{s_name}",
                    "entry_ts": datetime.fromtimestamp(entry_epoch, tz=timezone.utc),
                    "exit_ts": exit_dt,
                    "exit_date": exit_dt.strftime("%m-%d"),
                    "duration_min": tick_min - pos["entry_tick"],
                    "entry_basis_bp": round(pos["entry_basis_bp"], 1),
                    "pl_spread_usd": round(float(pl_spread), 2),
                    "funding_bp": round(float(funding_bp), 2),
                    "funding_usd": round(float(funding_usd), 2),
                    "pl_usd": round(float(pl_total), 2),
                    "pl_bp": round(float(pl_total) / paper_size * 10000, 1),
                })
                exit_confirm.pop(key, None)
            else:
                still_active.append(pos)

        active = still_active

        # ── 2. Update state and enter (vectorized) ────────────────────
        held_coins = set(p["coin_idx"] for p in active)

        eb_slice = entry_basis[t]  # raw basis — used for narrow check
        es_slice = entry_score[t]  # spread-adjusted — used for wide check & entry
        valid_eb = ~np.isnan(eb_slice)
        valid_es = ~np.isnan(es_slice)
        wide = valid_es & (es_slice > wide_threshold)
        narrow = valid_eb & (eb_slice < NARROW_THRESHOLD)

        # Compute new state in a copy to avoid order-of-operations issues
        new_state = state.copy()

        # -1 (unknown) -> 1 (narrow) or stay unknown
        valid = valid_eb | valid_es
        init = valid & (state == -1)
        new_state[init & narrow] = 1
        new_state[init & ~narrow] = 0

        if wait_entry:
            # 1 (narrow) -> 0 (wide_1tick) if wide
            new_state[(state == 1) & wide] = 0

            # 0 (wide_1tick) -> 2 (confirmed) if still wide, or 1 (narrow) if narrow
            was_wide1 = (state == 0)
            confirm = was_wide1 & wide
            new_state[confirm] = 2
            new_state[was_wide1 & narrow] = 1
        else:
            # 1 (narrow) -> 2 (enter immediately) if wide
            confirm = (state == 1) & wide
            new_state[confirm] = 2

        # 2 (wide_confirmed) -> 1 (narrow) if narrow
        new_state[(state == 2) & narrow] = 1

        state[:] = new_state

        # Find new entries from confirmed transitions
        entry_candidates = np.argwhere(confirm)  # (N, 2) array of [coin_idx, pair_idx]
        for row in entry_candidates:
            c, p = int(row[0]), int(row[1])
            if len(active) >= max_positions:
                capacity_blocked += 1
                continue
            if c in held_coins:
                coin_blocked += 1
                continue
            # Skip if spread-adjusted score doesn't clear threshold
            if np.isnan(es_slice[c, p]) or es_slice[c, p] <= wide_threshold:
                continue
            l_idx, s_idx = pair_indices[p]
            buy_l = buy_t[t, c, l_idx]
            sell_s = sell_t[t, c, s_idx]
            if np.isnan(buy_l) or np.isnan(sell_s) or buy_l <= 0 or sell_s <= 0:
                continue
            mid = (buy_l + sell_s) * 0.5
            fee = pair_fee_ow[p]
            active.append({
                "coin_idx": c,
                "pair_idx": p,
                "l_idx": l_idx,
                "s_idx": s_idx,
                "entry_tick": tick_min,
                "entry_tick_idx": t,
                "entry_buy_l": float(buy_l),
                "entry_sell_s": float(sell_s),
                "entry_basis_bp": float(eb_slice[c, p]),
                "entry_fee": float(fee),
            })
            held_coins.add(c)

        if len(active) > max_concurrent:
            max_concurrent = len(active)

        now = time.perf_counter()
        if not quiet and now - last_report > 5.0:
            pct = (t + 1) / n_ticks * 100
            elapsed = now - t0
            rate = (t + 1) / elapsed
            eta = (n_ticks - t - 1) / rate if rate > 0 else 0
            print(f"  {pct:5.1f}% | tick {t+1:,}/{n_ticks:,} | {len(trade_log)} closed | "
                  f"active={len(active)} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
            last_report = now

    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"\nDone: {elapsed:.1f}s, {len(trade_log)} closed trades, {len(active)} still open")
        print(f"  Max concurrent positions: {max_concurrent}")
        print(f"  Capacity-blocked entries: {capacity_blocked}")
        print(f"  Coin-blocked entries: {coin_blocked}")

    # Add zombies (still-open positions) with mark-to-market P&L
    last_tick_min = int(unique_mins[-1])
    last_t = n_ticks - 1
    for pos in active:
        dur = last_tick_min - pos["entry_tick"]
        entry_epoch = pos["entry_tick"] * 60
        exit_epoch = last_tick_min * 60
        c_idx = pos["coin_idx"]
        p_idx = pos["pair_idx"]
        l_ex = pair_names[p_idx][0]
        s_ex = pair_names[p_idx][1]
        coin = coins[c_idx]
        l_name = ABBREV[l_ex]
        s_name = ABBREV[s_ex]

        # Mark-to-market spread P&L using last tick prices
        sell_l = sell_t[last_t, c_idx, pos["l_idx"]]
        buy_s_val = buy_t[last_t, c_idx, pos["s_idx"]]
        pl_spread = None
        if not (np.isnan(sell_l) or np.isnan(buy_s_val) or sell_l <= 0 or buy_s_val <= 0):
            mid = (pos["entry_buy_l"] + pos["entry_sell_s"]) * 0.5
            qty = paper_size / mid
            pl_long = (sell_l - pos["entry_buy_l"]) * qty
            pl_short = (pos["entry_sell_s"] - buy_s_val) * qty
            fee_entry = pos["entry_fee"] / 10000 * paper_size
            fee_exit = pair_fee_ow[p_idx] / 10000 * paper_size
            pl_spread = round(float(pl_long + pl_short - fee_entry - fee_exit), 2)

        fund_l_bp = calc_funding_pl(funding_events, l_ex, coin, entry_epoch, exit_epoch, is_long=True)
        fund_s_bp = calc_funding_pl(funding_events, s_ex, coin, entry_epoch, exit_epoch, is_long=False)
        funding_bp = fund_l_bp + fund_s_bp
        funding_usd = funding_bp / 10000 * paper_size

        pl_total = None
        if pl_spread is not None:
            pl_total = round(pl_spread + float(funding_usd), 2)

        if not quiet:
            mtm = f"  mtm=${pl_total:+.0f}" if pl_total is not None else ""
            print(f"    {coin} {l_name}/{s_name}  entry_basis={pos['entry_basis_bp']:.1f}bp  open {dur}min ({dur//60}h){mtm}")

        trade_log.append({
            "coin": coin,
            "pair": f"{l_name}/{s_name}",
            "entry_ts": datetime.fromtimestamp(entry_epoch, tz=timezone.utc),
            "exit_ts": None,
            "exit_date": None,
            "duration_min": dur,
            "entry_basis_bp": round(pos["entry_basis_bp"], 1),
            "pl_spread_usd": pl_spread,
            "funding_bp": round(float(funding_bp), 2),
            "funding_usd": round(float(funding_usd), 2),
            "pl_usd": pl_total,
            "pl_bp": round(pl_total / paper_size * 10000, 1) if pl_total is not None else None,
            "zombie": True,
        })

    # Mark closed trades
    for t in trade_log:
        if "zombie" not in t:
            t["zombie"] = False

    return trade_log


def print_results(trade_log, wide_threshold):
    if not trade_log:
        print("No trades.")
        return

    df = pd.DataFrame(trade_log)

    n = len(df)
    wins = (df["pl_usd"] > 0).sum()
    total_pl = df["pl_usd"].sum()
    total_spread = df["pl_spread_usd"].sum()
    total_funding = df["funding_usd"].sum()
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({wide_threshold})")
    print(f"  {n} trades, {wins} wins ({wins/n*100:.1f}%)")
    print(f"  Spread P&L:  ${total_spread:+,.2f}")
    print(f"  Funding P&L: ${total_funding:+,.2f}")
    print(f"  Total P&L:   ${total_pl:+,.2f}")
    print(f"  Avg P&L: ${total_pl/n:+,.2f} ({df['pl_bp'].mean():+.1f} bp)")
    print(f"  Avg hold: {df['duration_min'].mean():.0f}min ({df['duration_min'].mean()/60:.1f}h)")
    print(f"{'='*70}")

    suffix = f"_{wide_threshold}"
    df.to_csv(f"sim2_trades{suffix}.csv", index=False)
    print(f"  Trade log written to sim2_trades{suffix}.csv")

    # Pivot table
    pivot = df.pivot_table(index="coin", columns="exit_date", values="pl_usd", aggfunc="mean")
    pivot["Total"] = df.groupby("coin")["pl_usd"].sum()
    pivot["N"] = df.groupby("coin")["pl_usd"].count().astype(int)

    totals = df.groupby("exit_date")["pl_usd"].sum()
    totals_row = pd.DataFrame(totals).T
    totals_row.index = ["TOTAL"]
    totals_row["Total"] = total_pl
    totals_row["N"] = n
    pivot = pd.concat([pivot, totals_row])

    pivot.to_csv(f"sim2_pivot{suffix}.csv", float_format="%.1f")
    print(f"  Pivot table written to sim2_pivot{suffix}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-bp", type=float, default=100, help="Wide threshold in bp")
    parser.add_argument("--no-wait", action="store_true", help="No 2-bar confirmation on entry")
    parser.add_argument("--wait-exit", action="store_true", help="2-bar confirmation on exit")
    parser.add_argument("--max-pos", type=int, default=10, help="Max concurrent positions")
    parser.add_argument("--size", type=int, default=10, choices=[10, 100], help="Position size in $k (10 or 100)")
    parser.add_argument("--exchanges", type=str, default=None, choices=["dex", "cex"], help="Filter exchanges")
    args = parser.parse_args()

    data = load_data(size_k=args.size, exchange_filter=args.exchanges)
    wait_entry = not args.no_wait
    trade_log = run_sim(data, args.wide_bp, wait_entry=wait_entry, wait_exit=args.wait_exit, max_positions=args.max_pos)
    parts = [f"{args.wide_bp}bp"]
    if args.no_wait:
        parts.append("no-wait")
    else:
        parts.append("2bar-entry")
    if args.wait_exit:
        parts.append("2bar-exit")
    parts.append(f"max={args.max_pos}")
    print_results(trade_log, " ".join(parts))
