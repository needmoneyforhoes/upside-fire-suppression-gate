# upside-fire-suppression-gate

Offline analysis suite that finds a pre-fire gate to block wrong UP fires in Polymarket 5-minute crypto up/down markets.

The live engine sometimes fires UP in markets that settle DN. These scripts mine recorded tick data for a gate condition that blocks the wrong UP fires in eventual-DN markets without blocking correct UP fires in eventual-UP markets. Features use only ticks recorded before each fire (no look-ahead). Evaluation is per-fire, so reversal markets do not produce sticky false locks. Output is a gate spec to wire back into the engine.

All four scripts are read-only: standard-library Python, no network, no order placement, no funds.

## Scripts

- `up_gate_backtest.py`: primary backtest. Per-fire tick analysis with Bonferroni correction, walk-forward 70/30 IS/OOS split, 10k-permutation market-level test, lag-1 autocorrelation check, net PnL across all markets.
- `up_gate_maximizer.py`: ranks every gate by actual PnL instead of block ratio. Sweeps DN-price, persistence, and fire-checkpoint gates; reports saves vs missed profit and entry-price-weighted breakdowns.
- `up_gate_scanner.py`: indicator-combo discovery. Adapter of the engine's `strategy_scanner_quant_v2.py` with the win definition inverted (`winner=='DN' and bet=='UP'` is a wrong fire). BH-FDR, walk-forward OOS, MC permutation, bootstrap CI, Kelly.
- `up_gate_tick_quant.py`: validates 2-indicator momentum gates (crowd, d3s, d10s, BN-delta) over full ticks. Structural-break split, Bonferroni, 5k permutations, 3k bootstrap.

`up_gate_scanner.py` is an adapter; the upstream `strategy_scanner_quant_v2.py` is not vendored here. The adapter runs standalone.

## Requirements

Python 3, standard library only (`json`, `math`, `random`, `itertools`, `collections`). No pip install.

## Usage

```bash
# Primary backtest: auto-discovers data, or pass a data dir or explicit files
python3 up_gate_backtest.py
python3 up_gate_backtest.py $DATA_DIR
python3 up_gate_backtest.py --mh market_history.jsonl --recap market_recap_history.jsonl

# Rank every gate by total PnL
python3 up_gate_maximizer.py $DATA_DIR/market_history.jsonl

# Indicator-combo gate discovery (auto-discovers or takes an explicit .jsonl)
python3 up_gate_scanner.py $DATA_DIR/market_history.jsonl

# Momentum 2-indicator gate validation
python3 up_gate_tick_quant.py $DATA_DIR/market_history.jsonl
```

## Data

Reads recorded tick data from the private `polymarket-data` repo:

- `market_history.jsonl`: per-market `ticks` and `tick_columns` (cd, up/dn bid/ask, depth, EMAs, deltas, crowd, BN-delta) plus `winner`.
- `market_recap_history.jsonl`: per-fire recap records, used by `up_gate_backtest.py`.

Pass the data dir or file path as an argument, or run from a directory where the files are discoverable. `backtest` and `scanner` also search the working dir and `$DATA_DIR`. Data files are gitignored and never committed.
