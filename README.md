# FAA outage attribution forecaster

make.fun bounty `polymarket-faa-attribution-02` ($250): forecast Polymarket's
FAA outage attribution Yes prices minute by minute, 4 hours ahead.

`main.py` (Python 3.13, stdlib only, deterministic, no network) reads one
case (`/input/<id>.in`: every market's minute-by-minute Yes prices for the
72 hours up to a checkpoint) and writes `/output/<id>.out` with five
quantiles (5/25/50/75/95%) for each market and each of the 240 minutes
after the checkpoint.

Method: per-market volatility from q90(|1-min moves|) over 24h/6h/2h;
median pulled toward its share of the family total (asymmetric: 1.5x
down, 0.5x up) plus 25% of the last 30-minute move continued linearly;
spread grows as h**0.3 with Student-t(5) quantile offsets, inner
quantiles tightened by 0.85.

Practice loss (same NVIDIA CEO practice cases): **0.261578** vs house
baseline 0.363589. Runtime ~0.6 s for all 6 practice cases.
