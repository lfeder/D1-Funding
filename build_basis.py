"""Build basis.json — precompute trades for all parameter combos.

Runs sim2 for a grid of (entry_thresh, exit_thresh, size, exchanges).
Output: basis/basis.json consumed by the viewer.
"""
import json
import os
import time
from datetime import timezone

from sim2 import load_data, run_sim

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "basis", "basis.json")
os.makedirs(os.path.join(BASE, "basis"), exist_ok=True)

ENTRY_THRESHOLDS = [25, 50, 75, 100, 125, 150]
EXIT_THRESHOLDS = [0, 10, 25]
SIZES = [10, 100]
EXCHANGE_FILTERS = [None, "dex"]  # None = all exchanges
MAX_POSITIONS = [10, 25, 9999]  # 9999 = uncapped


def trade_to_dict(t):
    """Convert a trade log entry to a JSON-friendly dict."""
    return {
        "coin": t["coin"],
        "pair": t["pair"],
        "entry_ts": int(t["entry_ts"].timestamp()) if t["entry_ts"] else None,
        "exit_ts": int(t["exit_ts"].timestamp()) if t["exit_ts"] else None,
        "duration_min": t["duration_min"],
        "entry_basis_bp": t["entry_basis_bp"],
        "spread_pl": t["pl_spread_usd"],
        "funding_pl": t["funding_usd"],
        "total_pl": t["pl_usd"],
        "zombie": t["zombie"],
    }


def main():
    t_start = time.perf_counter()
    all_results = {}

    for size_k in SIZES:
        for exch_filter in EXCHANGE_FILTERS:
            exch_label = "dex" if exch_filter == "dex" else "all"
            data_key = f"{size_k}k_{exch_label}"
            print(f"\n{'='*60}")
            print(f"Loading data: size={size_k}k, exchanges={exch_label}")
            print(f"{'='*60}")
            data = load_data(size_k=size_k, exchange_filter=exch_filter)

            for entry_bp in ENTRY_THRESHOLDS:
                for exit_bp in EXIT_THRESHOLDS:
                    for max_pos in MAX_POSITIONS:
                        max_label = "all" if max_pos == 9999 else str(max_pos)
                        key = f"{size_k}k_{exch_label}_e{entry_bp}_x{exit_bp}_m{max_label}"
                        print(f"  Running {key} ...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        trade_log = run_sim(
                            data, entry_bp,
                            wait_entry=False,
                            wait_exit=False,
                            max_positions=max_pos,
                            exit_threshold=exit_bp,
                            quiet=True,
                        )
                        elapsed = time.perf_counter() - t0
                        trades = [trade_to_dict(t) for t in trade_log]
                        closed = [t for t in trades if not t["zombie"]]
                        zombies = [t for t in trades if t["zombie"]]
                        total_pl = sum(t["total_pl"] for t in closed if t["total_pl"] is not None)
                        print(f"{len(closed)} trades, {len(zombies)} zombies, ${total_pl:+,.0f} in {elapsed:.1f}s")
                        all_results[key] = trades

    # Build time range from any dataset
    all_entry_ts = []
    all_exit_ts = []
    for trades in all_results.values():
        for t in trades:
            if t["entry_ts"]:
                all_entry_ts.append(t["entry_ts"])
            if t["exit_ts"]:
                all_exit_ts.append(t["exit_ts"])

    output = {
        "entry_thresholds": ENTRY_THRESHOLDS,
        "exit_thresholds": EXIT_THRESHOLDS,
        "sizes": SIZES,
        "exchange_filters": ["all", "dex"],
        "max_positions": [10, 25, "all"],
        "time_range": {
            "start": min(all_entry_ts) if all_entry_ts else 0,
            "end": max(all_exit_ts) if all_exit_ts else 0,
        },
        "results": all_results,
    }

    with open(OUT, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    fsize = os.path.getsize(OUT)
    total_trades = sum(len(v) for v in all_results.values())
    elapsed = time.perf_counter() - t_start
    print(f"\nWrote {OUT} ({fsize / 1024:.0f} KB)")
    print(f"  {len(all_results)} combos, {total_trades} total trade records")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
