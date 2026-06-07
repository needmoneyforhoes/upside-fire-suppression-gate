#!/usr/bin/env python3
"""
UP Gate Scanner — Adapter of strategy_scanner_quant_v2.py for gate discovery.

PROBLEM: Find indicator combinations that identify UP fires in DN winner markets
         (wrong fires to block) while not misidentifying UP fires in UP winner
         markets (correct fires to allow).

KEY MATHEMATICAL INVERSION:
  Original scanner: is_w = (bs == winner) → finds conditions where fires WIN
  This scanner:     is_w = (winner == 'DN' and bs == 'UP')
                         → finds conditions where UP fires are WRONG (in DN market)
  pnl_gate = +(ask * shares) if wrong UP fire (save the loss)
           = -(1-ask) * shares if correct UP fire blocked (miss the profit)

  The scanner then finds indicator combinations with POSITIVE edge under this
  inverted definition = combinations that correctly identify bad UP fires more
  often than they misidentify good ones = our gate conditions.

Full statistical rigor:
  - BH-FDR correction across all tested combinations
  - Walk-forward OOS (68/32 train/test split by market)
  - Monte Carlo permutation test (500 permutations)
  - Bootstrap 95% CI on edge
  - Kelly criterion

Usage:
  python3 up_gate_scanner.py /path/to/market_history.jsonl
  python3 up_gate_scanner.py  (auto-discovers files)
"""

import json, math, sys, random, os
from collections import defaultdict

random.seed(42)

# ── Column indices (match market_history.jsonl tick_columns) ─────────
CD=0; UP_BID=1; UP_ASK=2; DN_BID=3; DN_ASK=4
UP_DEPTH=5; DN_DEPTH=6; UP_ZC=7; DN_ZC=8; UP_SR=9; DN_SR=10
BN_PRICE=11; BN_DELTA=12; UP_EMA=13; DN_EMA=14
UP_D3=15; DN_D3=16; UP_D10=17; DN_D10=18
CROWD_SIDE=19; CROWD_CV=20; ASK_SUM=21
DV_UP=22; DV_DN=23; CL_DELTA=24; CL_AGE=25; BN_SPREAD=26

SHARES = 5

# Price zones to scan — UP fires in $0.10-$0.50 are the problem range
ZONES = [
    (0.10, 0.20, "$0.10-0.20 very cheap"),
    (0.20, 0.35, "$0.20-0.35 mid-cheap"),
    (0.35, 0.50, "$0.35-0.50 mid"),
    (0.10, 0.35, "$0.10-0.35 combined"),
    (0.10, 0.50, "$0.10-0.50 all bad zone"),
]

# ── File discovery ────────────────────────────────────────────────────
SEARCH = [os.getcwd(), '/home/polybot/polymarket-bot',
          os.path.dirname(os.path.abspath(__file__))]

def find(name):
    for d in SEARCH:
        p = os.path.join(d, name)
        if os.path.isfile(p): return p
    return None

MH_PATH = None
args = sys.argv[1:]
for a in args:
    if os.path.isfile(a) and a.endswith('.jsonl'): MH_PATH = a; break
    elif os.path.isdir(a): SEARCH.insert(0, a)
MH_PATH = MH_PATH or find('market_history.jsonl')

if not MH_PATH:
    print("ERROR: Cannot find market_history.jsonl"); sys.exit(1)

print(f"market_history: {MH_PATH}")

# ── Load markets ──────────────────────────────────────────────────────
print("Loading markets with tick data...", flush=True)
markets = []
with open(MH_PATH, buffering=8*1024*1024) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except: continue
        if r.get('ticks') and r.get('winner') in ('UP', 'DN'):
            markets.append(r)
        if len(markets) % 200 == 0 and markets:
            print(f"  {len(markets)} markets...", flush=True)

print(f"  Total: {len(markets)} markets")

# ── Lookback helper ───────────────────────────────────────────────────
def lb(ticks, ti, col, cd, secs):
    for j in range(ti-1, max(ti-800, -1), -1):
        if j < 0: return None
        if abs(ticks[j][CD] - (cd + secs)) < 3: return ticks[j][col] or 0
        if ticks[j][CD] > cd + secs + 5: return None
    return None

# ── Compute UP-specific indicators at fire tick ───────────────────────
def compute_up_indicators(t, ti, ticks):
    """
    Compute all indicators oriented for the UP side.
    These are the features available at the exact moment of a UP fire.
    No future bias: only uses ticks[0..ti] (past ticks).
    """
    cd  = t[CD]
    ba  = t[UP_ASK]  or 0    # UP ask (our entry price)
    opp = t[DN_ASK]  or 0    # DN ask (opponent price)

    # Oriented depth/EMA/delta columns
    od  = t[UP_DEPTH] or 0;  xd  = t[DN_DEPTH]  or 0
    odv = t[DV_UP]    or 0;  xdv = t[DV_DN]     or 0
    oe  = t[UP_EMA]   or 0
    od3 = t[UP_D3]    or 0;  xd3 = t[DN_D3]     or 0
    od10= t[UP_D10]   or 0;  xd10= t[DN_D10]    or 0
    os_ = t[UP_SR]    or 0;  xs  = t[DN_SR]      or 0
    oz  = t[UP_ZC]    or 0;  xz  = t[DN_ZC]      or 0
    ob  = t[UP_BID]   or 0

    bn  = t[BN_DELTA] or 0
    cv  = t[CROWD_CV] or 0
    cs  = t[CROWD_SIDE] or 0

    # Opponent ask gap (opp_ask - our_ask → large = opponent dominant)
    ask_gap = opp - ba

    # SR divergence
    sr_diff = xs - os_
    sr_max  = max(t[UP_SR] or 0, t[DN_SR] or 0)
    depth_asym = (od - xd) / (od + xd) if (od + xd) > 0 else 0
    dr = od / max(xd, 1)

    # BN aligned with UP: positive = BTC rising = good for UP
    bn_wb = bn

    # Crowd aligned with UP: +1 if crowd=UP, -1 if crowd=DN
    crowd_wb = 1 if cs == 1 else (-1 if cs == -1 else 0)
    bn_confirms_crowd = 1 if (bn > 0 and cs == 1) or (bn < 0 and cs == -1) else (-1 if cs != 0 else 0)

    # ── BN lookbacks ──────────────────────────────────────────────────
    bn_10 = bn_30 = bn_60 = None
    for s, attr in [(10, 'bn_10'), (30, 'bn_30'), (60, 'bn_60')]:
        v = lb(ticks, ti, BN_DELTA, cd, s)
        if v is not None:
            val = (bn - v)   # positive = BTC rising (favorable for UP)
            if s == 10:   bn_10 = val
            elif s == 30: bn_30 = val
            else:          bn_60 = val

    bn_vals = []
    for j in range(ti, max(ti - 800, -1), -1):
        if ticks[j][CD] - cd > 60: break
        bn_vals.append(ticks[j][BN_DELTA] or 0)
    bn_range_60 = (max(bn_vals) - min(bn_vals)) if len(bn_vals) > 2 else 0
    bn_std_60   = 0
    if len(bn_vals) > 5:
        m_ = sum(bn_vals) / len(bn_vals)
        bn_std_60 = math.sqrt(sum((x - m_)**2 for x in bn_vals) / len(bn_vals))

    bn_slope_30_wb = bn_slope_10_wb = 0
    bv30 = []; bv10 = []
    for j in range(ti, max(ti - 500, -1), -1):
        dt = ticks[j][CD] - cd
        if dt > 30: break
        bv = ticks[j][BN_DELTA] or 0
        bv30.append(bv)
        if dt <= 10: bv10.append(bv)
    if len(bv30) >= 2: bn_slope_30_wb = bv30[0] - bv30[-1]  # positive = BN rising
    if len(bv10) >= 2: bn_slope_10_wb = bv10[0] - bv10[-1]

    bn_sign_flips = 0
    for k in range(1, len(bv30)):
        if bv30[k] * bv30[k-1] < 0: bn_sign_flips += 1

    # ── Depth trajectories ────────────────────────────────────────────
    depth_chg_10 = depth_chg_30 = depth_chg_60 = None
    for s in [10, 30, 60]:
        v = lb(ticks, ti, UP_DEPTH, cd, s)
        if v is not None:
            if s == 10:   depth_chg_10 = od - v
            elif s == 30: depth_chg_30 = od - v
            else:          depth_chg_60 = od - v

    opp_depth_chg_10 = opp_depth_chg_30 = None
    for s in [10, 30]:
        v = lb(ticks, ti, DN_DEPTH, cd, s)
        if v is not None:
            if s == 10:   opp_depth_chg_10 = xd - v
            else:          opp_depth_chg_30 = xd - v

    ema_chg_10 = ema_chg_30 = None
    for s in [10, 30]:
        v = lb(ticks, ti, UP_EMA, cd, s)
        if v is not None:
            if s == 10:   ema_chg_10 = oe - v
            else:          ema_chg_30 = oe - v

    ask_chg_10 = ask_chg_30 = None
    for s in [10, 30]:
        v = lb(ticks, ti, UP_ASK, cd, s)
        if v is not None:
            if s == 10:   ask_chg_10 = ba - v   # negative = price falling (bad for UP)
            else:          ask_chg_30 = ba - v

    opp_ask_chg_10 = opp_ask_chg_30 = None
    for s in [10, 30]:
        v = lb(ticks, ti, DN_ASK, cd, s)
        if v is not None:
            if s == 10:   opp_ask_chg_10 = opp - v   # positive = DN rising
            else:          opp_ask_chg_30 = opp - v

    dv_chg_10 = lb(ticks, ti, DV_UP, cd, 10)
    if dv_chg_10 is not None: dv_chg_10 = odv - dv_chg_10

    zc_chg_10 = lb(ticks, ti, UP_ZC, cd, 10)
    if zc_chg_10 is not None: zc_chg_10 = oz - zc_chg_10

    # ── Conviction velocity ───────────────────────────────────────────
    cv_5s  = lb(ticks, ti, CROWD_CV, cd, 5)
    cv_10s = lb(ticks, ti, CROWD_CV, cd, 10)
    conv_vel   = (cv - (cv_5s or cv)) / 5 if cv_5s else 0
    conv_accel = 0
    if cv_5s is not None and cv_10s is not None:
        conv_accel = ((cv - cv_5s) - (cv_5s - cv_10s)) / 5

    # ── Lead changes (market leadership history) ──────────────────────
    lead_changes = 0; prev_lead = None
    for j in range(0, min(ti, len(ticks))):
        ub = ticks[j][UP_BID] or 0; db = ticks[j][DN_BID] or 0
        lead = 'UP' if ub > db else ('DN' if db > ub else prev_lead)
        if lead and prev_lead and lead != prev_lead: lead_changes += 1
        prev_lead = lead

    # ── Crowd fraction DN over last 60s ───────────────────────────────
    crowd_dn_60 = 0; n_60 = 0
    for j in range(ti, max(ti - 800, -1), -1):
        if ticks[j][CD] - cd > 60: break
        n_60 += 1
        if (ticks[j][CROWD_SIDE] or 0) == -1: crowd_dn_60 += 1
    crowd_dn_frac = crowd_dn_60 / max(n_60, 1)

    # Sustained opp_ask above threshold (number of last-20s ticks with DN≥0.50)
    dn_above_50 = dn_above_55 = dn_above_60 = 0; n_20 = 0
    for j in range(ti, max(ti - 300, -1), -1):
        if ticks[j][CD] - cd > 20: break
        da = ticks[j][DN_ASK] or 0
        n_20 += 1
        if da >= 0.50: dn_above_50 += 1
        if da >= 0.55: dn_above_55 += 1
        if da >= 0.60: dn_above_60 += 1

    return {
        'cd': cd, 'ask': ba, 'opp_ask': opp, 'ask_gap': ask_gap,
        # Depth
        'own_depth': od, 'opp_depth': xd, 'depth_ratio': dr,
        'depth_asym': depth_asym, 'own_dv': odv, 'opp_dv': xdv,
        # SR/ZC
        'own_sr': os_, 'opp_sr': xs, 'sr_diff': sr_diff, 'sr_max': sr_max,
        'own_zc': oz, 'opp_zc': xz,
        # EMA
        'own_ema': oe, 'ema_va': oe - ba if oe else 0,
        # BN
        'bn_wb': bn_wb, 'bn_abs': abs(bn),
        'bn_10': bn_10, 'bn_30': bn_30, 'bn_60': bn_60,
        'bn_range_60': bn_range_60, 'bn_std_60': bn_std_60,
        'bn_slope_30': bn_slope_30_wb, 'bn_slope_10': bn_slope_10_wb,
        'bn_sign_flips': bn_sign_flips,
        # Crowd
        'cv': cv, 'crowd_wb': crowd_wb, 'crowd_dn_frac': crowd_dn_frac,
        'bn_confirms_crowd': bn_confirms_crowd,
        # Deltas (oriented: positive = own side rising)
        'd3_wb': od3 - xd3, 'd10_wb': od10 - xd10,
        # Trajectories (negative = own ask falling = bad for UP)
        'ask_chg_10': ask_chg_10, 'ask_chg_30': ask_chg_30,
        'opp_ask_chg_10': opp_ask_chg_10, 'opp_ask_chg_30': opp_ask_chg_30,
        'depth_chg_10': depth_chg_10, 'depth_chg_30': depth_chg_30,
        'depth_chg_60': depth_chg_60,
        'opp_depth_chg_10': opp_depth_chg_10, 'opp_depth_chg_30': opp_depth_chg_30,
        'ema_chg_10': ema_chg_10, 'ema_chg_30': ema_chg_30,
        'dv_chg_10': dv_chg_10, 'zc_chg_10': zc_chg_10,
        # Conviction
        'conv_vel': conv_vel, 'conv_accel': conv_accel,
        # Sustained opponent dominance
        'dn_above_50_20s': dn_above_50 / max(n_20, 1),
        'dn_above_55_20s': dn_above_55 / max(n_20, 1),
        'dn_above_60_20s': dn_above_60 / max(n_20, 1),
        # Lead history
        'lead_changes': lead_changes,
        # Market-level
        'ask_sum': t[ASK_SUM] or 0,
    }

# ── Build inverted gate fires ─────────────────────────────────────────
def compute_gate_fires(markets, zone_lo, zone_hi):
    """
    For each market, find the first tick where UP ask enters the zone.
    Compute all indicators at that exact tick.

    INVERTED DEFINITION:
      is_w = (winner == 'DN')   → correct to block (bad UP fire)
      pnl  = +ask*SHARES if DN winner (saved loss)
           = -(1-ask)*SHARES if UP winner (missed profit)

    A gate condition with positive edge under this definition correctly
    identifies bad UP fires more profitably than it misidentifies good ones.
    """
    fires = []
    for mi, r in enumerate(markets):
        winner  = r['winner']
        ticks   = r.get('ticks', [])
        slug    = r.get('slug', '')
        if not ticks: continue

        # Find first tick where UP ask enters zone (first fire opportunity)
        for ti, t in enumerate(ticks):
            ba = t[UP_ASK] or 0
            if not ba or not (zone_lo <= ba < zone_hi): continue

            # Inverted: is_w = market is DN winner = bad UP fire
            is_w  = (winner == 'DN')
            # pnl: what blocking this fire is worth
            if is_w:
                pnl = SHARES * ba              # saved the loss (entry price × shares)
            else:
                pnl = -SHARES * (1.0 - ba)    # missed the profit

            ind = compute_up_indicators(t, ti, ticks)
            ind['is_w']   = is_w
            ind['pnl']    = pnl
            ind['slug']   = slug
            ind['mi']     = mi
            ind['winner'] = winner
            fires.append(ind)
            break   # one fire per market per zone

    return fires

# ── Scan helpers (adapted from scanner) ──────────────────────────────
def _eval(fires, fn, min_n=6):
    p = [f for f in fires if fn(f)]
    if len(p) < min_n: return None
    w = sum(1 for f in p if f['is_w'])
    pnl = sum(f['pnl'] for f in p)
    if pnl <= 0: return None
    wp  = [f['pnl'] for f in p if f['is_w']]
    lp_ = [f['pnl'] for f in p if not f['is_w']]
    aw  = sum(wp) / len(wp)   if wp  else 0
    al  = sum(lp_) / len(lp_) if lp_ else 0
    wr  = w / len(p) * 100
    if aw > 0 and al < 0 and (aw - al) != 0: be = -al / (aw - al) * 100
    elif not lp_: be = 0
    else: be = 100
    edge = wr - be
    if edge <= 0: return None
    wlr = aw / abs(al) if al != 0 else 999
    return (edge, be, pnl, len(p), w, wr, aw, al, wlr)

def scan(fires, min_n=8):
    skip = {'is_w', 'pnl', 'slug', 'mi', 'ask', 'winner'}
    keys = [k for k in fires[0].keys() if k not in skip and fires[0][k] is not None]
    results = []
    for key in keys:
        vals = [f[key] for f in fires if f[key] is not None and isinstance(f[key], (int, float))]
        if len(vals) < min_n: continue
        sv = sorted(set(vals))
        for pct in range(5, 96, 5):
            idx   = min(int(len(sv) * pct / 100), len(sv) - 1)
            thresh = sv[idx]
            for d in ['>', '<']:
                if d == '>':
                    fn  = lambda f, th=thresh, k=key: f.get(k) is not None and f[k] > th
                    lbl = f"{key}>{thresh:.4f}"
                else:
                    fn  = lambda f, th=thresh, k=key: f.get(k) is not None and f[k] < th
                    lbl = f"{key}<{thresh:.4f}"
                r = _eval(fires, fn, min_n)
                if r:
                    results.append((r[0], r[1], r[2], lbl, r[3], r[4], r[5], r[6], r[7], r[8], key))
    results.sort(reverse=True)
    return results

def _build_candidates(fires, top_n=20):
    skip = {'is_w', 'pnl', 'slug', 'mi', 'ask', 'winner'}
    keys = [k for k in fires[0].keys() if k not in skip]
    scored = []
    for key in keys:
        vals = [f[key] for f in fires if f.get(key) is not None and isinstance(f.get(key), (int, float))]
        if len(vals) < 6: continue
        sv = sorted(set(vals)); best = None
        for pct in range(5, 96, 5):
            idx    = min(int(len(sv) * pct / 100), len(sv) - 1)
            thresh = sv[idx]
            for d in ['>', '<']:
                if d == '>': fn = lambda f, th=thresh, k=key: f.get(k) is not None and f[k] > th; lbl = f"{key}>{thresh:.4f}"
                else:        fn = lambda f, th=thresh, k=key: f.get(k) is not None and f[k] < th; lbl = f"{key}<{thresh:.4f}"
                r = _eval(fires, fn, min_n=5)
                if r and (best is None or r[0] > best[0]):
                    best = (r[0], lbl, fn, key)
        if best: scored.append(best)
    scored.sort(reverse=True)
    seen = set(); out = []
    for edge, lbl, fn, key in scored:
        if key in seen: continue
        seen.add(key); out.append((lbl, fn, key))
        if len(out) >= top_n: break
    return out

def forward_stepwise(fires, max_depth=7, min_n=6, top_candidates=20):
    candidates = _build_candidates(fires, top_n=top_candidates)
    if not candidates: return []
    results = []; best_single = None
    for lbl, fn, key in candidates:
        r = _eval(fires, fn, min_n)
        if r and (best_single is None or r[0] > best_single[0]):
            best_single = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], lbl, fn, key, [key])
    if not best_single: return []
    edge, be, pnl, n, w, wr, aw, al, wlr, lbl, fn, key, used = best_single
    results.append((1, 'BASE', lbl, edge, be, pnl, n, w, wr, aw, al, wlr))
    cur_fn = fn; cur_lbl = lbl; cur_edge = edge; cur_used = [key]
    for depth in range(2, max_depth + 1):
        best_and = best_or = None
        for c_lbl, c_fn, c_key in candidates:
            if c_key in cur_used: continue
            and_fn = lambda f, cf=cur_fn, nf=c_fn: cf(f) and nf(f)
            r = _eval(fires, and_fn, min_n)
            if r and (best_and is None or r[0] > best_and[0]):
                best_and = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], c_lbl, and_fn, c_key)
            or_fn = lambda f, cf=cur_fn, nf=c_fn: cf(f) or nf(f)
            r = _eval(fires, or_fn, min_n)
            if r and (best_or is None or r[0] > best_or[0]):
                best_or = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], c_lbl, or_fn, c_key)
        picked = None; mode = None
        if best_and and best_or:
            picked, mode = (best_and, 'AND') if best_and[0] >= best_or[0] else (best_or, 'OR')
        elif best_and: picked, mode = best_and, 'AND'
        elif best_or:  picked, mode = best_or,  'OR'
        if not picked: break
        edge, be, pnl, n, w, wr, aw, al, wlr, add_lbl, new_fn, add_key = picked
        if edge <= cur_edge and depth > 2: break
        cur_lbl = f"{cur_lbl} {mode} {add_lbl}"
        cur_fn = new_fn; cur_edge = edge; cur_used.append(add_key)
        results.append((depth, mode, cur_lbl, edge, be, pnl, n, w, wr, aw, al, wlr))
        if n < min_n * 2: break
    return results

# ── OOS validation ────────────────────────────────────────────────────
def scan_with_oos(fires, min_n=6, train_ratio=0.70):
    slugs      = list(dict.fromkeys(f['slug'] for f in fires))
    split      = int(len(slugs) * train_ratio)
    train_s    = set(slugs[:split]); test_s = set(slugs[split:])
    train      = [f for f in fires if f['slug'] in train_s]
    test       = [f for f in fires if f['slug'] in test_s]
    if len(train) < min_n or len(test) < 3: return []
    train_res  = scan(train, min_n=max(4, min_n - 2))
    oos_res    = []
    seen       = set()
    for edge_tr, be_tr, pnl_tr, lbl, n_tr, w_tr, wr_tr, aw_tr, al_tr, wlr_tr, key in train_res:
        if key in seen: continue; seen.add(key)
        parts = lbl.split('>')
        if len(parts) == 2:
            k, th = parts[0], float(parts[1])
            fn = lambda f, k=k, th=th: f.get(k) is not None and f[k] > th
        else:
            parts = lbl.split('<')
            if len(parts) != 2: continue
            k, th = parts[0], float(parts[1])
            fn = lambda f, k=k, th=th: f.get(k) is not None and f[k] < th
        r = _eval(test, fn, min_n=3)
        if r:
            oos_res.append((edge_tr, be_tr, pnl_tr, lbl, n_tr, w_tr, wr_tr,
                            aw_tr, al_tr, wlr_tr, key,
                            r[0], r[3], r[4], r[5]))
    oos_res.sort(key=lambda x: -(x[0] + x[11]) / 2)
    return oos_res

# ── Statistical validation ────────────────────────────────────────────
def _binom_p(wins, n, be_wr):
    if n == 0 or be_wr <= 0 or be_wr >= 100: return 1.0
    p_null = be_wr / 100.0; p_val = 0.0
    for k in range(wins, n + 1):
        lc = 0.0
        for i in range(k): lc += math.log(n - i) - math.log(i + 1)
        lp = lc + k * math.log(max(p_null, 1e-15)) + (n - k) * math.log(max(1 - p_null, 1e-15))
        p_val += math.exp(lp)
    return min(p_val, 1.0)

def _bh_correction(p_values):
    n = len(p_values)
    if n == 0: return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n; prev = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = n - rank_from_end
        adj = min(p * n / rank, prev); adjusted[orig_idx] = min(adj, 1.0); prev = adj
    return adjusted

def _bootstrap_ci(wins, n, avg_w, avg_l, n_boot=1000):
    if n < 4: return (0, 0)
    pnls = [avg_w] * wins + [avg_l] * (n - wins); edges = []
    for _ in range(n_boot):
        s = random.choices(pnls, k=n); sw = sum(1 for x in s if x > 0); sl = n - sw
        saw = sum(x for x in s if x > 0) / max(sw, 1)
        sal = sum(x for x in s if x <= 0) / max(sl, 1)
        swr = sw / n * 100
        if saw > 0 and sal < 0 and (saw - sal) != 0: sbe = -sal / (saw - sal) * 100
        elif sl == 0: sbe = 0
        else: sbe = 100
        edges.append(swr - sbe)
    edges.sort()
    return (edges[int(n_boot * 0.025)], edges[min(int(n_boot * 0.975), n_boot - 1)])

def validate_results(results, n_total_tests=3600):
    if not results: return []
    raw_p   = [_binom_p(w, n, be) for edge, be, pnl, lbl, n, w, wr, aw, al, wlr, key in results]
    all_p   = raw_p + [1.0] * max(0, n_total_tests - len(raw_p))
    adj_p   = _bh_correction(all_p)[:len(raw_p)]
    enriched = []
    for i, (edge, be, pnl, lbl, n, w, wr, aw, al, wlr, key) in enumerate(results):
        ci_lo, ci_hi = _bootstrap_ci(w, n, aw, al)
        enriched.append((edge, be, pnl, lbl, n, w, wr, aw, al, wlr, key, adj_p[i], ci_lo, ci_hi))
    return enriched

def monte_carlo(fires, n_perms=500, min_n=6):
    real = scan(fires, min_n=min_n)
    if not real: return None
    real_best = real[0][0]; perm_bests = []; exceed = 0
    for pi in range(n_perms):
        if pi % 50 == 0 and pi > 0:
            sys.stderr.write(f"\r  MC {pi}/{n_perms}  p_so_far={exceed/pi:.3f}"); sys.stderr.flush()
            if pi >= 100:
                if exceed / pi < 0.005: break   # clearly significant
                if exceed / pi > 0.35:  break   # clearly not significant
        labels = [f['is_w'] for f in fires]; random.shuffle(labels)
        sf = []
        for j, f in enumerate(fires):
            d = dict(f); d['is_w'] = labels[j]
            d['pnl'] = (SHARES * f['ask'] if labels[j] else -SHARES * (1.0 - f['ask']))
            sf.append(d)
        pr_ = scan(sf, min_n=min_n); pb = pr_[0][0] if pr_ else 0
        perm_bests.append(pb)
        if pb >= real_best: exceed += 1
    n_act = len(perm_bests)
    sys.stderr.write(f"\r  MC done ({n_act} perms)        \n"); sys.stderr.flush()
    perm_bests.sort()
    return (real_best, perm_bests[int(n_act * 0.95)], exceed / n_act)

# ── Print helpers ─────────────────────────────────────────────────────
def pr_validated(title, validated, n=20):
    print(f"\n{'='*130}")
    print(f"  {title}")
    print(f"  Interpretation: positive edge = gate correctly identifies bad UP fires")
    print(f"  gate condition = block UP fires when this indicator combination is true")
    print(f"{'='*130}")
    print(f"\n{'#':>2} {'indicator combo':<32} {'F':>4} {'W':>3} {'WR':>5} {'BE':>5}"
          f" {'edge':>7} {'p_BH':>7} {'95%CI':>12} {'PnL':>9} {'verdict'}")
    print("-" * 130)
    seen = set(); c = 0
    for edge, be, pnl, lbl, nn, w, wr, aw, al, wlr, key, p_adj, ci_lo, ci_hi in validated:
        if key in seen: continue; seen.add(key); c += 1
        ls = lbl[:32] if len(lbl) <= 32 else lbl[:29] + "..."
        ci = f"({ci_lo:+.0f},{ci_hi:+.0f})"
        if p_adj < 0.05 and ci_lo > 0:   v = "✅ GATE PROVEN"
        elif p_adj < 0.10 and ci_lo > -3: v = "🟡 GATE LIKELY"
        elif p_adj < 0.20:                v = "⚪ GATE WEAK"
        else:                              v = "❌ NOISE"
        print(f"{c:>2} {ls:<32} {nn:>4} {w:>3} {wr:>4.0f}% {be:>4.0f}% "
              f"{edge:>+6.1f}pp {p_adj:>7.4f} {ci:>12} ${pnl:>+8.2f} {v}")
        if c >= n: break

def pr_oos(title, oos_res, n=15):
    print(f"\n{'='*130}")
    print(f"  {title}  (IS edge | OOS validation)")
    print(f"{'='*130}")
    print(f"\n{'#':>2} {'indicator':<30} {'IS_edge':>8} {'IS_n':>5} "
          f"{'OOS_edge':>9} {'OOS_n':>6} {'OOS_wr':>8} {'verdict'}")
    print("-" * 95)
    c = 0
    for row in oos_res[:n]:
        edge_tr, be_tr, pnl_tr, lbl, n_tr, w_tr, wr_tr, aw_tr, al_tr, wlr_tr, key, \
            oos_edge, oos_n, oos_w, oos_wr = row
        c += 1; ls = lbl[:30] if len(lbl) <= 30 else lbl[:27] + "..."
        v = "✅ IS+OOS" if oos_edge > 0 else "❌ OOS fail"
        print(f"{c:>2} {ls:<30} {edge_tr:>+7.1f}pp {n_tr:>5} "
              f"{oos_edge:>+8.1f}pp {oos_n:>6} {oos_wr:>7.1f}%  {v}")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*80}")
    print("UP GATE SCANNER — Multi-indicator combination discovery")
    print(f"Markets: {len(markets)}")
    print("Inverted target: is_w = (winner == 'DN') = bad UP fire to block")
    print(f"{'='*80}\n")

    for zone_lo, zone_hi, zone_name in ZONES:
        print(f"\n{'#'*80}")
        print(f"  ZONE: {zone_name}  |  UP fires entering this price range")
        print(f"{'#'*80}")

        fires = compute_gate_fires(markets, zone_lo, zone_hi)
        if len(fires) < 20:
            print(f"  Too few fires: {len(fires)} — skipping"); continue

        n_bad    = sum(1 for f in fires if f['is_w'])     # UP fires in DN markets
        n_good   = sum(1 for f in fires if not f['is_w']) # UP fires in UP markets
        pnl_base = sum(f['pnl'] for f in fires)
        print(f"  Total fires: {len(fires)}  (bad={n_bad} DN-mkt, good={n_good} UP-mkt)")
        print(f"  Baseline pnl if gate always blocks: ${pnl_base:+.2f}")
        if pnl_base <= 0:
            print(f"  Baseline negative = blocking ALL fires in this zone loses money")
            print(f"  Need to find specific INDICATOR COMBOS that are net positive")

        # ── Phase 1: Single indicator scan ────────────────────────────
        print(f"\n  Phase 1: Single indicator scan...", flush=True)
        singles = scan(fires, min_n=8)
        validated = validate_results(singles[:100])
        pr_validated(f"Zone {zone_name} — top single indicators (BH-FDR corrected)",
                     validated, n=15)

        if not validated or validated[0][12] < 0:
            print(f"  No indicators with CI > 0 in single scan")

        # ── Phase 2A: OOS validation ──────────────────────────────────
        print(f"\n  Phase 2A: Out-of-sample validation (70/30 walk-forward)...", flush=True)
        oos = scan_with_oos(fires, train_ratio=0.70)
        pr_oos(f"Zone {zone_name} — OOS validated indicators", oos, n=10)

        # ── Phase 2B: Forward stepwise combination ────────────────────
        print(f"\n  Phase 2B: Forward stepwise combination (depth up to 7)...", flush=True)
        fw = forward_stepwise(fires, max_depth=7, min_n=6, top_candidates=20)
        if fw:
            print(f"\n  FORWARD STEPWISE — best combination chain:")
            print(f"  {'depth':>5} {'mode':>5} {'edge':>7} {'n':>5} {'wr':>6} {'be':>6} {'PnL':>9}  {'indicators'}")
            print("  " + "-" * 100)
            for depth, mode, lbl, edge, be, pnl, n, w, wr, aw, al, wlr in fw:
                ls = lbl[:50] if len(lbl) <= 50 else lbl[:47] + "..."
                print(f"  {depth:>5} {mode:>5} {edge:>+6.1f}pp {n:>5} {wr:>5.1f}% {be:>5.1f}% "
                      f"${pnl:>+8.2f}  {ls}")
            best = fw[-1]
            depth, mode, lbl, edge, be, pnl_fw, n, w, wr, aw, al, wlr = best
            print(f"\n  BEST COMBO: {lbl}")
            print(f"  Edge={edge:+.1f}pp  WR={wr:.1f}%  BE={be:.1f}%  n={n}  PnL=${pnl_fw:+.2f}")
        else:
            print(f"  No valid combination found")

        # ── Phase 3: Monte Carlo on best zone fires ────────────────────
        print(f"\n  Phase 3: Monte Carlo permutation test...", flush=True)
        mc = monte_carlo(fires, n_perms=500, min_n=6)
        if mc:
            real_best, p95, p_val = mc
            print(f"  MC: real_best_edge={real_best:+.1f}pp  p95_perm={p95:+.1f}pp  "
                  f"p_value={p_val:.4f}  "
                  f"{'✅ SIGNIFICANT' if p_val < 0.05 else '⚠️ p=' + str(round(p_val,3))}")

        print(f"\n  [Done zone: {zone_name}]")

    print(f"\n{'='*80}")
    print("GATE SCANNER COMPLETE")
    print("Top-ranked indicator combinations with ✅ GATE PROVEN:")
    print("  → implement as: if condition → block this UP fire")
    print("  → condition = the indicator combo listed above")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
