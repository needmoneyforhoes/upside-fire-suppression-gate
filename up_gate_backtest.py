#!/usr/bin/env python3
"""
UP Gate Backtest — Per-fire, tick-based analysis
Finds the optimal gate condition to block wrong UP fires in DN winner markets
without blocking correct UP fires in UP winner markets.

No future bias: all features use only tick data from BEFORE each fire.
Per-fire evaluation: handles reversal markets (no market-sticky false locks).

Usage:
    python3 up_gate_backtest.py
    python3 up_gate_backtest.py /path/to/data/dir
    python3 up_gate_backtest.py --mh market_history.jsonl --recap market_recap_history.jsonl

Statistical rigor:
    - Bonferroni correction across all tested configs
    - Walk-forward 70/30 chronological split (IS/OOS)
    - Market-level permutation test (10,000 permutations)
    - Lag-1 autocorrelation check on per-market outcomes
    - Net PnL across ALL markets (not just recall on wrong fires)
"""

import json, os, sys, math, random, itertools
from collections import defaultdict

random.seed(42)

# ── File discovery ────────────────────────────────────────────────────
SEARCH = [os.getcwd(), '.',
          os.path.dirname(os.path.abspath(__file__))]

def find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    return None

MH_PATH    = None
RECAP_PATH = None
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--mh'    and i+1<len(args): MH_PATH    = args[i+1]; i+=2
    elif args[i] == '--recap' and i+1<len(args): RECAP_PATH = args[i+1]; i+=2
    elif os.path.isdir(args[i]): SEARCH.insert(0, args[i]); i+=1
    else: i+=1

MH_PATH    = MH_PATH    or find('market_history.jsonl')
RECAP_PATH = RECAP_PATH or find('market_recap_history.jsonl')

if not MH_PATH or not RECAP_PATH:
    print("ERROR: Cannot find data files.")
    print(f"  market_history.jsonl     : {MH_PATH or 'NOT FOUND'}")
    print(f"  market_recap_history.jsonl: {RECAP_PATH or 'NOT FOUND'}")
    sys.exit(1)

print(f"market_history : {MH_PATH}")
print(f"market_recap   : {RECAP_PATH}")

# ── Config ────────────────────────────────────────────────────────────
N_PERM       = 10000   # permutations for significance test
IS_FRAC      = 0.70    # 70% train, 30% OOS
ENTRY_FLOOR  = 0.15    # never block fires below this (DDS data collection)

# ── Load market history with tick data ───────────────────────────────
print("\nLoading market_history.jsonl (streaming)...", flush=True)
mh_data = {}   # slug -> {'ticks': [...], 'cols': {...col:idx}, 'ts': str, 'winner': str}

n_rows = 0
with open(MH_PATH, buffering=8*1024*1024) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
        except: continue
        slug   = r.get('slug', '')
        winner = r.get('winner', '')
        ticks  = r.get('ticks', [])
        cols   = r.get('tick_columns', [])
        ts     = r.get('ts', '')
        if not slug or not ticks or not cols: continue
        col_idx = {c: i for i, c in enumerate(cols)}
        mh_data[slug] = {'ticks': ticks, 'cols': col_idx, 'ts': ts, 'winner': winner}
        n_rows += 1
        if n_rows % 500 == 0:
            print(f"  {n_rows:,} markets loaded...", flush=True)

print(f"  Total markets with tick data: {n_rows:,}")

# ── Load recap fires ──────────────────────────────────────────────────
print("Loading market_recap_history.jsonl...", flush=True)
recap_data = []   # list of dicts per market with fires

with open(RECAP_PATH, buffering=4*1024*1024) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
        except: continue
        slug   = r.get('slug', '')
        winner = r.get('winner', '')
        fires  = r.get('fires', [])
        ts     = r.get('ts', '')
        if winner not in ('UP', 'DN') or not fires: continue
        recap_data.append({'slug': slug, 'winner': winner, 'fires': fires, 'ts': ts})

recap_data.sort(key=lambda r: r['ts'])
print(f"  Total recap markets: {len(recap_data):,}")

# ── Per-fire feature extraction ───────────────────────────────────────
def extract_features(slug, fire_cd):
    """
    Extract per-fire features from tick data strictly before the fire.
    At fire cd=X, ticks with tick_cd > X happened earlier (cd counts down).
    Lookback windows: [X, X+20] for sustained conditions, [X, X+60] for crowd.
    No future bias: only uses data from before the fire.
    """
    md = mh_data.get(slug)
    if not md: return None

    ticks = md['ticks']
    ci    = md['cols']

    cd_i  = ci.get('cd',  -1)
    dn_i  = ci.get('dn_ask', -1)
    up_i  = ci.get('up_ask', -1)
    cs_i  = ci.get('crowd_side', -1)
    cc_i  = ci.get('crowd_conviction', -1)
    ud_i  = ci.get('up_delta_10s', -1)
    dd_i  = ci.get('dn_delta_10s', -1)
    dv_i  = ci.get('depth_velocity_dn', -1)

    if cd_i < 0 or dn_i < 0: return None

    # Get ticks in lookback window [fire_cd, fire_cd+60]
    # These are ticks that happened BEFORE the fire (higher cd = earlier)
    lb60 = [t for t in ticks if fire_cd <= t[cd_i] <= fire_cd + 60]
    lb20 = [t for t in lb60 if t[cd_i] <= fire_cd + 20]

    if not lb20: return None

    # ── Sustained dn_ask — 2 consecutive 10s samples ─────────────────
    # Sample 1: tick closest to fire_cd (most recent)
    # Sample 2: tick closest to fire_cd+10 (10s before that)
    def nearest(tlist, target_cd):
        """Tick nearest to target_cd."""
        if not tlist: return None
        return min(tlist, key=lambda t: abs(t[cd_i] - target_cd))

    s1 = nearest(lb20, fire_cd)       # most recent (closest to fire)
    s2 = nearest(lb20, fire_cd + 10)  # 10s earlier

    dn_s1 = s1[dn_i] if s1 else None
    dn_s2 = s2[dn_i] if s2 else None

    # ── Crowd fraction DN over last 60s ───────────────────────────────
    crowd_dn_60 = None
    if lb60 and cs_i >= 0:
        crowd_dn_60 = sum(1 for t in lb60 if t[cs_i] == -1) / len(lb60)

    # ── up_delta_10s at fire time (slope of UP price) ─────────────────
    ud_val = s1[ud_i] if (s1 and ud_i >= 0 and len(s1) > ud_i) else None

    # ── dn_delta_10s at fire time ─────────────────────────────────────
    dd_val = s1[dd_i] if (s1 and dd_i >= 0 and len(s1) > dd_i) else None

    # ── depth_velocity_dn 30s avg ─────────────────────────────────────
    lb30 = [t for t in lb60 if t[cd_i] <= fire_cd + 30]
    dv_avg = None
    if lb30 and dv_i >= 0:
        vals = [t[dv_i] for t in lb30 if len(t) > dv_i]
        dv_avg = sum(vals)/len(vals) if vals else None

    return {
        'dn_s1':       dn_s1,       # dn_ask at fire moment
        'dn_s2':       dn_s2,       # dn_ask 10s before fire
        'crowd_dn_60': crowd_dn_60, # fraction of last 60s crowd=DN
        'up_delta':    ud_val,      # up_ask slope last 10s
        'dn_delta':    dd_val,      # dn_ask slope last 10s
        'dv_dn_30':    dv_avg,      # depth velocity DN 30s avg
        'n_lb20':      len(lb20),
        'n_lb60':      len(lb60),
    }

# ── Build per-fire records ────────────────────────────────────────────
print("\nExtracting per-fire features from tick data...", flush=True)
fire_records = []   # one row per UP fire
matched_mkts = 0
skipped_no_ticks = 0

for mkt in recap_data:
    slug   = mkt['slug']
    winner = mkt['winner']
    ts     = mkt['ts']

    if slug not in mh_data:
        skipped_no_ticks += 1
        continue

    matched_mkts += 1
    for f in mkt['fires']:
        side  = f.get('side', '')
        entry = f.get('entry_price') or 0
        cd    = f.get('cd') or 0
        pnl   = f.get('hypo_pnl') or 0
        if side != 'UP' or entry <= 0: continue

        feats = extract_features(slug, cd)
        fire_records.append({
            'slug':    slug,
            'ts':      ts,
            'winner':  winner,
            'entry':   entry,
            'cd':      cd,
            'pnl':     pnl,
            'feats':   feats,
        })

fire_records.sort(key=lambda r: r['ts'])
print(f"  Matched markets: {matched_mkts}  (skipped no-tick: {skipped_no_ticks})")
print(f"  Total UP fires: {len(fire_records):,}")

up_dn = [r for r in fire_records if r['winner']=='DN']
up_up = [r for r in fire_records if r['winner']=='UP']
has_feats_dn = sum(1 for r in up_dn if r['feats'])
has_feats_up = sum(1 for r in up_up if r['feats'])
print(f"  Wrong UP fires (DN markets): {len(up_dn):,}  with features: {has_feats_dn:,}")
print(f"  Correct UP fires (UP markets): {len(up_up):,}  with features: {has_feats_up:,}")
total_loss = sum(r['pnl'] for r in up_dn)
total_gain = sum(r['pnl'] for r in up_up)
print(f"  Total loss from wrong UP fires:   ${total_loss:+.2f}")
print(f"  Total gain from correct UP fires: ${total_gain:+.2f}")

# ── Gate evaluation function ──────────────────────────────────────────
def gate_fires(fn, records, floor=ENTRY_FLOOR):
    """
    fn(feats) -> True = block this fire.
    Returns: (saved, missed, n_blocked)
    saved  = losses avoided (wrong UP fires blocked)
    missed = profits missed (correct UP fires blocked = false positives)
    """
    saved = missed = n = 0.0
    for r in records:
        if r['entry'] < floor: continue
        f = r['feats']
        if f is None: continue
        if fn(f):
            n += 1
            if r['winner'] == 'DN':  # wrong fire, correctly blocked
                saved += -r['pnl']
            else:                     # correct fire, wrongly blocked
                missed += r['pnl']
    return saved, missed, int(n)

def net_pnl(fn, records, floor=ENTRY_FLOOR):
    s, m, _ = gate_fires(fn, records, floor)
    return s - m

# ── Walk-forward split ────────────────────────────────────────────────
# Split by market (unique slugs in chronological order)
all_slugs = list(dict.fromkeys(r['slug'] for r in fire_records))
n_is      = int(len(all_slugs) * IS_FRAC)
is_slugs  = set(all_slugs[:n_is])
oos_slugs = set(all_slugs[n_is:])

IS_fires  = [r for r in fire_records if r['slug'] in is_slugs]
OOS_fires = [r for r in fire_records if r['slug'] in oos_slugs]

print(f"\nWalk-forward split: IS={len(is_slugs)} markets | OOS={len(oos_slugs)} markets")
print(f"  IS fires: {len(IS_fires):,}  OOS fires: {len(OOS_fires):,}")

# ── Autocorrelation helper ────────────────────────────────────────────
def autocorr_lag1(fn, records, floor=ENTRY_FLOOR):
    """Lag-1 autocorrelation of per-market net PnL outcomes."""
    per_mkt = defaultdict(lambda: [0.0, 0.0])
    for r in records:
        if r['entry'] < floor or not r['feats']: continue
        blocked = fn(r['feats'])
        if r['winner']=='DN' and blocked:
            per_mkt[r['slug']][0] += -r['pnl']
        elif r['winner']=='UP' and blocked:
            per_mkt[r['slug']][1] -= r['pnl']
    seq = [v[0]+v[1] for v in per_mkt.values()]
    if len(seq) < 5: return None
    n = len(seq); m = sum(seq)/n; v = sum((x-m)**2 for x in seq)/n
    if v == 0: return 0.0
    c = sum((seq[i]-m)*(seq[i+1]-m) for i in range(n-1))/(n-1)
    return c/v

# ── Permutation test ─────────────────────────────────────────────────
def perm_test(fn, records, n_perm=N_PERM, floor=ENTRY_FLOOR):
    """
    Market-level permutation: shuffle winner labels across markets.
    Tests: is gate better than random market selection?
    Returns p-value (lower = more significant).
    """
    # Group fires by market
    by_mkt = defaultdict(list)
    for r in records:
        if r['entry'] >= floor and r['feats']:
            by_mkt[r['slug']].append(r)

    slugs   = list(by_mkt.keys())
    winners = {slug: by_mkt[slug][0]['winner'] for slug in slugs}

    # Observed net PnL
    obs = net_pnl(fn, records, floor)

    # Permuted: shuffle winner labels
    perm_nets = []
    winner_vals = list(winners.values())
    for _ in range(n_perm):
        random.shuffle(winner_vals)
        perm_winners = dict(zip(slugs, winner_vals))
        # Rebuild records with shuffled winners (only affects pnl accounting)
        pnet = 0.0
        for slug, fires in by_mkt.items():
            pw = perm_winners[slug]
            for r in fires:
                if fn(r['feats']):
                    if pw == 'DN': pnet += -r['pnl']
                    else:          pnet -=  r['pnl']
        perm_nets.append(pnet)

    return sum(1 for x in perm_nets if x >= obs) / n_perm

# ── Recall / FPR helpers ─────────────────────────────────────────────
def recall_fpr(fn, records, floor=ENTRY_FLOOR):
    tp = fp = fn_c = tn = 0
    for r in records:
        if r['entry'] < floor or not r['feats']: continue
        blocked = fn(r['feats'])
        if r['winner']=='DN':
            if blocked: tp += 1
            else:       fn_c += 1
        else:
            if blocked: fp += 1
            else:       tn += 1
    rec = tp/(tp+fn_c) if (tp+fn_c) else 0
    fpr = fp/(fp+tn)   if (fp+tn)   else 0
    return rec, fpr

# ── Gate configurations to test ──────────────────────────────────────
# Core features:
#   dn_s1: dn_ask at fire moment
#   dn_s2: dn_ask 10s before fire
#   crowd_dn_60: fraction of last 60s crowd=DN
#   up_delta: up_ask slope last 10s

configs = []

# 1. Pure sustained dn_ask (2 consecutive 10s samples)
for t in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
    label = f"dn_sustained≥{t:.2f}"
    fn = (lambda f, t=t:
          f.get('dn_s1') is not None and f['dn_s1'] >= t and
          f.get('dn_s2') is not None and f['dn_s2'] >= t)
    configs.append((label, fn))

# 2. Sustained dn_ask + crowd confirmation
for t in [0.55, 0.60, 0.65]:
    for c in [0.70, 0.75, 0.80]:
        label = f"dn≥{t:.2f}+crowd_DN≥{c:.0%}"
        fn = (lambda f, t=t, c=c:
              f.get('dn_s1') is not None and f['dn_s1'] >= t and
              f.get('dn_s2') is not None and f['dn_s2'] >= t and
              f.get('crowd_dn_60') is not None and f['crowd_dn_60'] >= c)
        configs.append((label, fn))

# 3. Sustained dn_ask + up momentum (UP price declining)
for t in [0.55, 0.60, 0.65]:
    for ud in [-0.01, -0.02, -0.03]:
        label = f"dn≥{t:.2f}+up_delta≤{ud:.2f}"
        fn = (lambda f, t=t, ud=ud:
              f.get('dn_s1') is not None and f['dn_s1'] >= t and
              f.get('dn_s2') is not None and f['dn_s2'] >= t and
              f.get('up_delta') is not None and f['up_delta'] <= ud)
        configs.append((label, fn))

# 4. Triple: dn_ask + crowd + up_delta
for t in [0.55, 0.60]:
    for c in [0.70, 0.75]:
        for ud in [-0.01, -0.02]:
            label = f"dn≥{t:.2f}+crowd≥{c:.0%}+ud≤{ud:.2f}"
            fn = (lambda f, t=t, c=c, ud=ud:
                  f.get('dn_s1') is not None and f['dn_s1'] >= t and
                  f.get('dn_s2') is not None and f['dn_s2'] >= t and
                  f.get('crowd_dn_60') is not None and f['crowd_dn_60'] >= c and
                  f.get('up_delta') is not None and f['up_delta'] <= ud)
            configs.append((label, fn))

# 5. Crowd-only (no dn_ask threshold)
for c in [0.75, 0.80, 0.85, 0.90]:
    label = f"crowd_DN≥{c:.0%}"
    fn = (lambda f, c=c:
          f.get('crowd_dn_60') is not None and f['crowd_dn_60'] >= c)
    configs.append((label, fn))

# 6. dn_ask single-sample (not sustained — weaker baseline)
for t in [0.55, 0.60, 0.65, 0.70]:
    label = f"dn_s1_only≥{t:.2f}"
    fn = (lambda f, t=t:
          f.get('dn_s1') is not None and f['dn_s1'] >= t)
    configs.append((label, fn))

N_TESTS    = len(configs)
ALPHA_BONF = 0.05 / N_TESTS

print(f"\nTesting {N_TESTS} gate configurations")
print(f"Bonferroni α = 0.05/{N_TESTS} = {ALPHA_BONF:.5f}")
print(f"Permutation test: {N_PERM:,} permutations per config (top candidates only)")

# ── Run sweep ─────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print("GATE SWEEP RESULTS")
print(f"{'='*110}")
print(f"  {'Config':<40} {'IS_net':>9} {'IS_rec':>8} {'IS_fpr':>7}"
      f" {'OOS_net':>9} {'OOS_rec':>8} {'OOS_fpr':>7} {'Verdict'}")
print(f"  {'-'*40} {'-'*9} {'-'*8} {'-'*7}"
      f" {'-'*9} {'-'*8} {'-'*7} {'-------'}")

sweep_results = []

for label, fn in configs:
    is_net  = net_pnl(fn, IS_fires)
    oos_net = net_pnl(fn, OOS_fires)
    is_rec, is_fpr   = recall_fpr(fn, IS_fires)
    oos_rec, oos_fpr = recall_fpr(fn, OOS_fires)

    # Quick verdict without permutation (save time for top candidates)
    if is_net > 0 and oos_net > 0:
        v = "✅ both+"
    elif is_net > 0:
        v = "⚠️  IS only"
    else:
        v = "❌"

    sweep_results.append({
        'label': label, 'fn': fn,
        'is_net': is_net, 'oos_net': oos_net,
        'is_rec': is_rec, 'is_fpr': is_fpr,
        'oos_rec': oos_rec, 'oos_fpr': oos_fpr,
        'both_pos': is_net > 0 and oos_net > 0,
        'verdict': v,
    })

    print(f"  {label:<40} {is_net:>+9.2f} {is_rec*100:>7.1f}% {is_fpr*100:>6.1f}%"
          f" {oos_net:>+9.2f} {oos_rec*100:>7.1f}% {oos_fpr*100:>6.1f}%  {v}")

# ── Top candidates: full statistical validation ───────────────────────
candidates = [r for r in sweep_results if r['both_pos']]
candidates.sort(key=lambda r: -(r['is_net'] + r['oos_net']))

print(f"\n{'='*110}")
print(f"FULL STATISTICAL VALIDATION (top candidates, both IS+OOS positive)")
print(f"Bonferroni α = {ALPHA_BONF:.5f}  |  Permutation test: {N_PERM:,} reps")
print(f"{'='*110}")

if not candidates:
    print("  No configs positive on both IS and OOS.")
    print("  See individual IS results above — check if VPS has more markets.")
else:
    print(f"  {'Config':<40} {'Total_net':>10} {'p_perm':>9} {'AC_lag1':>9}"
          f" {'OOS_rec':>8} {'OOS_fpr':>7} {'Verdict'}")
    print(f"  {'-'*40} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")

    final_results = []
    for r in candidates[:10]:   # full stats on top 10
        fn = r['fn']
        pp = perm_test(fn, fire_records)
        ac = autocorr_lag1(fn, fire_records)

        bonf_pass = pp < ALPHA_BONF
        oos_target = r['oos_rec'] >= 0.70 and r['oos_fpr'] <= 0.25

        if bonf_pass and oos_target:
            v = "✅ PROMOTE (Bonf+OOS+target)"
        elif bonf_pass:
            v = "✅ Bonf, OOS borderline"
        elif pp < 0.05 and r['both_pos']:
            v = f"⚠️  p<0.05 but not Bonf ({N_TESTS} tests)"
        else:
            v = "— accumulate more data"

        ac_str = f"{ac:+.3f}" if ac is not None else "N/A"
        iid_flag = " ⚠️ AC" if (ac is not None and abs(ac) > 0.20) else ""

        print(f"  {r['label']:<40} {r['is_net']+r['oos_net']:>+10.2f} {pp:>9.5f}"
              f" {ac_str:>9} {r['oos_rec']*100:>7.1f}% {r['oos_fpr']*100:>6.1f}%  {v}{iid_flag}")

        final_results.append({**r, 'pp': pp, 'ac': ac, 'verdict_full': v})

    # ── Best config detail ────────────────────────────────────────────
    promote = [r for r in final_results if 'PROMOTE' in r.get('verdict_full','')]
    if promote:
        best = promote[0]
        print(f"\n{'='*65}")
        print(f"BEST GATE: {best['label']}")
        print(f"{'='*65}")
        s_is, m_is, n_is_b   = gate_fires(best['fn'], IS_fires)
        s_oos, m_oos, n_oos_b = gate_fires(best['fn'], OOS_fires)
        print(f"  IS:  blocked={n_is_b}  saved=${s_is:+.2f}  missed=${m_is:+.2f}  "
              f"net=${s_is-m_is:+.2f}")
        print(f"  OOS: blocked={n_oos_b}  saved=${s_oos:+.2f}  missed=${m_oos:+.2f}  "
              f"net=${s_oos-m_oos:+.2f}")
        print(f"  recall={best['oos_rec']*100:.1f}%  FPR={best['oos_fpr']*100:.1f}%")
        print(f"  p_perm={best['pp']:.5f}  Bonf α={ALPHA_BONF:.5f}  "
              f"{'PASS ✅' if best['pp']<ALPHA_BONF else 'FAIL ❌'}")
        print(f"  autocorr lag-1 = {best['ac']:+.3f}  "
              f"{'IID OK ✅' if best['ac'] is None or abs(best['ac'])<0.20 else 'correlated ⚠️'}")
        print(f"\n  Implementation (mirrors dn_live):")
        print(f"  if dn_ask_s1 >= T and dn_ask_s2 >= T  (2 consecutive 10s samples before fire)")
        print(f"  [+ crowd_DN_frac_60s if applicable]")
        print(f"  → block this UP fire")

print(f"\n{'='*110}")
print(f"  Total configs tested: {N_TESTS}")
print(f"  Both IS+OOS positive: {len(candidates)}")
print(f"  Pass Bonferroni ({ALPHA_BONF:.5f}): "
      f"{sum(1 for r in final_results if r.get('pp',1)<ALPHA_BONF) if candidates else 0}")
print(f"  Entry floor: ${ENTRY_FLOOR}  IS/OOS split: {IS_FRAC*100:.0f}/{(1-IS_FRAC)*100:.0f}%")
print(f"{'='*110}\n")
