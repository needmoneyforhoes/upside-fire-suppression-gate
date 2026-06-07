# upside-fire-suppression-gate

UP-fire suppression gate suite for Polymarket: tick-based backtest, PnL maximizer,
scanner adapter, and momentum quant test for **5-minute crypto up/down markets**.

## Why it exists

The live engine sometimes fires the **UP** side in markets that ultimately settle **DN** —
small wrong bets that bleed PnL. This suite mines historical tick data to discover a
pre-fire **gate condition** that blocks the wrong UP fires in eventual-DN markets *without*
blocking correct UP fires in eventual-UP markets. Every feature uses only ticks recorded
**before** each fire (no look-ahead), and evaluation is per-fire so reversal markets don't
trigger sticky false locks. Output is a deployable gate spec to wire back into the engine.

## How it works

| Script | Role | Method |
|---|---|---|
| `up_gate_backtest.py` | **Primary backtest** — finds the optimal blocking gate | Per-fire, tick-based; Bonferroni correction, walk-forward 70/30 IS/OOS split, 10k-permutation market-level test, lag-1 autocorrelation check, net PnL across all markets |
| `up_gate_maximizer.py` | **PnL maximizer** — ranks every gate by *actual* PnL, not block ratio | Sweeps DN-price / persistence / fire-checkpoint gates; reports saves vs. missed profit and entry-price-weighted breakdowns |
| `up_gate_scanner.py` | **Indicator-combo discovery** — adapter of the engine's `strategy_scanner_quant_v2.py` | Inverts the win definition (`winner=='DN' and bet=='UP'` ⇒ wrong fire) to surface indicator combinations with positive edge; BH-FDR, walk-forward OOS, MC permutation, bootstrap CI, Kelly |
| `up_gate_tick_quant.py` | **Momentum quant test** — validates 2-indicator momentum gates | Full-tick momentum gates (crowd / Δ3s / Δ10s / BN-delta) with structural-break split, Bonferroni, 5k permutations, 3k bootstrap |

All four are **offline, read-only analysis tools**: pure Python standard library, no network,
no order placement, no funds.

## Requirements

- Python 3 (standard library only — `json`, `math`, `random`, `itertools`, `collections`; no pip install needed)
- Read access to the private **`polymarket-data`** repo for the input tick files (see Data)

> `up_gate_scanner.py` is an *adapter of* the engine's `strategy_scanner_quant_v2.py`; that
> upstream scanner is **not vendored here**. The adapter is self-contained and runs standalone.

## Usage

```bash
# Primary backtest — auto-discovers data, or point at a data dir / explicit files
python3 up_gate_backtest.py
python3 up_gate_backtest.py /path/to/polymarket-data
python3 up_gate_backtest.py --mh market_history.jsonl --recap market_recap_history.jsonl

# Rank every gate by total PnL
python3 up_gate_maximizer.py /path/to/polymarket-data/market_history.jsonl

# Indicator-combination gate discovery (auto-discovers or takes an explicit .jsonl)
python3 up_gate_scanner.py /path/to/polymarket-data/market_history.jsonl

# Momentum 2-indicator gate validation
python3 up_gate_tick_quant.py /path/to/polymarket-data/market_history.jsonl
```

## Data

These scripts read recorded market tick data and live in the private **`polymarket-data`** repo:

- `market_history.jsonl` — per-market `ticks` + `tick_columns` (cd, up/dn bid/ask, depth, EMAs, deltas, crowd, BN-delta, …) and `winner`
- `market_recap_history.jsonl` — per-fire recap records (used by `up_gate_backtest.py`)

Pass the data directory or file path as an argument, or run from a directory where the files
are discoverable (`backtest`/`scanner` also search the working dir and `/home/polybot/polymarket-bot`).
Data files are `.gitignore`d and never committed.

> Private research software. No warranty; trades/handles real funds at your own risk.
