#!/usr/bin/env python3
"""FAA outage attribution forecaster — make.fun polymarket-faa-attribution-02.

Reads one case per run: every market of the family with its minute-by-minute
Yes prices for the 72 hours up to a checkpoint, and writes, for each market
and each of the 240 minutes after the checkpoint, five quantiles
(5/25/50/75/95%) of that minute's price.

Method (fresh implementation of the approach that scored 0.2607 on the same
practice data in the NVIDIA bounty):
- Per-market volatility from the 90th percentile of |1-minute moves| over
  the last 24h / 6h / 2h, averaged (floored and capped).
- Median path: the price is pulled toward its share of the family's total
  (the attributions are mutually exclusive, so the family sums near 1),
  with an asymmetric pull — stronger when the family total is above 1
  (prices collectively too high) than below it — plus a quarter of the
  last 30-minute move continued as momentum.
- Spread grows like h**0.3 around the median, with Student-t(5) quantile
  offsets; the inner 25%/75% quantiles are tightened by 0.85 because most
  1-minute moves are exactly zero (discrete prices). A quarter of the
  last 30-minute move is continued as momentum, growing linearly across
  the horizon.

Stdlib only, deterministic, no network, no wall clock.
"""

import json
import math
import os
import sys

HORIZON = 240
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Student-t with 5 degrees of freedom, quantiles at the five levels.
T5 = (-2.015048, -0.726687, 0.0, 0.726687, 2.015048)
INNER_TIGHTEN = 0.85
SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.03
SPREAD_POWER = 0.3
# Asymmetric family-sum pull: stronger downward correction (total > 1)
# than upward (total < 1).
PULL_DOWN = 1.5
PULL_UP = 0.5
# Horizon (minutes) over which the family-sum gap is closed linearly.
PULL_HORIZON = 240.0
MOMENTUM_SHARE = 0.25
MOMENTUM_LOOKBACK = 30
# Horizon (minutes) over which the continued 30-minute move ramps in.
# 240.0 means it grows linearly across the whole horizon.
MOMENTUM_HORIZON = 240.0


def _q90(values):
    """90th percentile of a non-empty list of numbers (nearest-rank)."""
    ordered = sorted(values)
    idx = int(math.ceil(0.9 * len(ordered))) - 1
    if idx < 0:
        idx = 0
    return ordered[idx]


def volatility(prices):
    """Per-market sigma from recent 1-minute moves.

    Mean of q90(|1-min move|) over the last 24h, 6h and 2h windows,
    floored and capped. Windows shorter than the history are skipped;
    a market with no moves at all gets the floor.
    """
    moves = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    if not moves:
        return SIGMA_FLOOR
    estimates = []
    for window in (1440, 360, 120):
        recent = moves[-window:]
        if len(recent) >= 10:
            estimates.append(_q90([abs(m) for m in recent]))
    if not estimates:
        estimates.append(_q90([abs(m) for m in moves]))
    sigma = sum(estimates) / len(estimates)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def median_curve(prices, share_target, sig):
    """Median price for each of the 240 horizon minutes.

    share_target is the price this market would have if the family total
    were exactly 1 (its share of 1). The gap to that target is closed
    over ~2 hours, asymmetrically; a quarter of the last 30-minute move
    is continued as momentum, ramping over the first hour.
    """
    x = prices[-1]
    gap = share_target - x
    pull = PULL_DOWN if gap < 0 else PULL_UP
    if len(prices) > MOMENTUM_LOOKBACK:
        drift30 = prices[-1] - prices[-1 - MOMENTUM_LOOKBACK]
    else:
        drift30 = 0.0
    curve = []
    for h in range(1, HORIZON + 1):
        frac = min(1.0, h / PULL_HORIZON)
        m = x + gap * frac * pull
        mfrac = min(1.0, h / MOMENTUM_HORIZON)
        m += MOMENTUM_SHARE * drift30 * mfrac
        curve.append(min(1.0, max(0.0, m)))
    return curve


def forecast_market(prices, share_target):
    """240 rows x 5 quantiles for one market."""
    sig = volatility(prices)
    medians = median_curve(prices, share_target, sig)
    rows = []
    for h in range(1, HORIZON + 1):
        spread = sig * (h ** SPREAD_POWER)
        med = medians[h - 1]
        row = []
        for k, z in enumerate(T5):
            width = spread * (INNER_TIGHTEN if k in (1, 3) else 1.0)
            row.append(min(1.0, max(0.0, med + z * width)))
        # Enforce non-decreasing quantiles after clamping/rounding.
        row = [round(v, 6) for v in row]
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def forecast_case(case):
    markets = case["markets"]
    total = sum(m["prices"][-1] for m in markets if m["prices"])
    out = []
    for m in markets:
        prices = m["prices"]
        if not prices:
            # No history at all: flat 0.5 fan (should not happen).
            rows = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(HORIZON)]
            out.append({"id": m["id"], "q": rows})
            continue
        x = prices[-1]
        share = x / total if total > 0 else x
        out.append({"id": m["id"], "q": forecast_market(prices, share)})
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
