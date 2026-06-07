#!/usr/bin/env python3
"""
UP Gate Quant Test — Full Tick Data
Tests momentum-based 2-indicator gates using all ticks from market_history.jsonl.
Runs on VPS where full tick data is available.

Usage: python3 up_gate_tick_quant.py [path_to_market_history.jsonl]
"""
import json, sys, random, math
from collections import defaultdict
random.seed(42)

MH_PATH = sys.argv[1] if len(sys.argv)>1 else 'market_history.jsonl'
FLOOR   = 0.10   # min entry price
BREAK   = 121    # structural break market index (chronological)
N_BONF  = 4      # number of gates tested (Bonferroni)
ALPHA   = 0.05 / N_BONF
N_PERM  = 5000
N_BOOT  = 3000

print(f"Loading {MH_PATH} ...")
markets = []
with open(MH_PATH) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: markets.append(json.loads(line))
        except: pass

print(f"Loaded {len(markets)} markets")

# ── Build fire records with tick-level features ──────────────────────
def build_fires(markets):
    all_fires = []
    for r in markets:
        winner = r.get('winner','')
        slug   = r.get('slug','')
        ticks  = r.get('ticks',[])
        cols   = r.get('tick_columns',[])
        if not ticks or not cols or winner not in ('UP','DN'):
            continue

        ci = {c:i for i,c in enumerate(cols)}
        CD   = ci.get('cd',-1)
        UA   = ci.get('up_ask',-1)
        DA   = ci.get('dn_ask',-1)
        CS   = ci.get('crowd_side',-1)
        CC   = ci.get('crowd_conviction',-1)
        D10U = ci.get('up_delta_10s',-1)
        D10D = ci.get('dn_delta_10s',-1)
        D3U  = ci.get('up_delta_3s',-1)
        D3D  = ci.get('dn_delta_3s',-1)
        BD   = ci.get('bn_delta_pct', ci.get('bn_delta',-1))

        if UA<0 or CD<0: continue

        fired = set()
        for t in ticks:
            ua = t[UA] if UA>=0 and len(t)>UA else None
            if ua is None or ua<=FLOOR or ua>=0.55: continue
            band = round(ua*20)/20
            if band in fired: continue
            fired.add(band)

            cd   = t[CD]  if CD>=0  and len(t)>CD  else None
            da   = t[DA]  if DA>=0  and len(t)>DA  else 0.0
            cs   = t[CS]  if CS>=0  and len(t)>CS  else 0
            cc   = t[CC]  if CC>=0  and len(t)>CC  else 0.0
            d10u = t[D10U]if D10U>=0 and len(t)>D10U else None
            d10d = t[D10D]if D10D>=0 and len(t)>D10D else None
            d3u  = t[D3U] if D3U>=0  and len(t)>D3U  else None
            d3d  = t[D3D] if D3D>=0  and len(t)>D3D  else None
            bn   = t[BD]  if BD>=0  and len(t)>BD  else 0.0

            pnl = 5*((1-ua) if winner=='UP' else -ua)

            all_fires.append({
                'slug':slug, 'winner':winner,
                'cd':cd, 'ua':ua, 'da':da or 0,
                'cs':cs, 'cc':cc or 0,
                'd10u':d10u, 'd10d':d10d,
                'd3u':d3u,   'd3d':d3d,
                'bn':bn or 0, 'pnl':pnl,
            })
    return all_fires

print("Extracting per-fire tick features ...")
all_fires = build_fires(markets)
print(f"Total UP fires: {len(all_fires)}")

# ── Chronological market ordering & regime split ─────────────────────
# Use slug timestamp for ordering
def slug_ts(slug):
    try: return int(slug.split('-')[-1])
    except: return 0

mkt_slugs_ordered = []
seen = set()
# order by slug timestamp from the markets list
for r in sorted(markets, key=lambda m: slug_ts(m.get('slug',''))):
    sl = r.get('slug','')
    if sl and sl not in seen and r.get('winner') in ('UP','DN'):
        seen.add(sl); mkt_slugs_ordered.append(sl)

print(f"Ordered markets: {len(mkt_slugs_ordered)}")
print(f"Structural break at market #{BREAK}")

regime_slugs = mkt_slugs_ordered[BREAK:]
n_is  = int(len(regime_slugs)*0.70)
IS_sl = set(regime_slugs[:n_is])
OOS_sl= set(regime_slugs[n_is:])

fires_all = [f for f in all_fires if f['slug'] in set(regime_slugs)]
fires_IS  = [f for f in all_fires if f['slug'] in IS_sl]
fires_OOS = [f for f in all_fires if f['slug'] in OOS_sl]

avg_rate = len(fires_all)/max(len(regime_slugs),1)
print(f"Current regime: {len(regime_slugs)} mkts  "
      f"IS={len(IS_sl)} OOS={len(OOS_sl)}")
print(f"Regime fires: {len(fires_all)} ({avg_rate:.1f}/mkt)")
print()

# Count how many fires have d10u/d3u fields available
has_d10 = sum(1 for f in fires_all if f['d10u'] is not None)
has_d3  = sum(1 for f in fires_all if f['d3u']  is not None)
print(f"Fires with d10_up: {has_d10}/{len(fires_all)} "
      f"({has_d10/max(len(fires_all),1)*100:.0f}%)")
print(f"Fires with d3_up:  {has_d3}/{len(fires_all)} "
      f"({has_d3/max(len(fires_all),1)*100:.0f}%)")
print()

# ── Gate evaluation ──────────────────────────────────────────────────
def eval_gate(fires, fn):
    saved=missed=0.0; n=0
    for f in fires:
        if fn(f):
            n+=1
            if f['winner']=='DN': saved+=-f['pnl']
            else:                 missed+=f['pnl']
    return saved-missed, n, saved, missed

def per_type(fires, fn):
    dn=up=0.0
    for f in fires:
        if not fn(f):
            if f['winner']=='DN': dn+=f['pnl']
            else:                 up+=f['pnl']
    return dn, up

def perm_test(fires, fn, n_perm=N_PERM):
    """Market-level permutation with recomputed pnl under shuffled winners."""
    by_mkt = defaultdict(list)
    for f in fires: by_mkt[f['slug']].append(f)
    slugs_l = list(by_mkt.keys())
    winners  = {sl:by_mkt[sl][0]['winner'] for sl in slugs_l}
    obs,_,_,_ = eval_gate(fires, fn)
    exceed=0; wv=list(winners.values())
    for _ in range(n_perm):
        random.shuffle(wv); pw=dict(zip(slugs_l,wv)); pnet=0.0
        for sl,fs in by_mkt.items():
            for f in fs:
                if fn(f):
                    e=f['ua']
                    if pw[sl]=='DN': pnet+=e*5
                    else:            pnet-=(1-e)*5
        if pnet>=obs: exceed+=1
    return exceed/n_perm

def boot_ci(fires, fn, n_boot=N_BOOT):
    slugs_l = list({f['slug'] for f in fires})
    by_mkt  = defaultdict(list)
    for f in fires: by_mkt[f['slug']].append(f)
    nets=[]
    for _ in range(n_boot):
        sample_sl = random.choices(slugs_l, k=len(slugs_l))
        sample_f  = [f for sl in sample_sl for f in by_mkt[sl]]
        nets.append(eval_gate(sample_f, fn)[0])
    nets.sort()
    return nets[int(0.025*n_boot)], nets[int(0.975*n_boot)], sum(nets)/n_boot

def autocorr(fires, fn):
    by_mkt=defaultdict(list)
    for f in fires: by_mkt[f['slug']].append(f)
    seq=[eval_gate(fs,fn)[0] for fs in by_mkt.values()]
    n=len(seq); m=sum(seq)/n
    v=sum((x-m)**2 for x in seq)/n
    if v==0: return 0.0
    c=sum((seq[i]-m)*(seq[i+1]-m) for i in range(n-1))/(n-1)
    return c/v

# ── 4 gate definitions ───────────────────────────────────────────────
gates = [
    # 1. Momentum: d10_up falling AND d3_up falling (primary from logs)
    ("d10u≤-0.04 AND d3u≤-0.05",
     lambda f: f['d10u'] is not None and f['d3u'] is not None
               and f['d10u']<=-0.04 and f['d3u']<=-0.05),

    # 2. Bidirectional momentum: d10_up falling AND d10_dn rising
    ("d10u≤-0.04 AND d10d≥0.04",
     lambda f: f['d10u'] is not None and f['d10d'] is not None
               and f['d10u']<=-0.04 and f['d10d']>=0.04),

    # 3. Price level + crowd conviction (snapshot-available proxy)
    ("dn≥0.65 AND crowd_DN≥0.70",
     lambda f: f['da']>=0.65 and f['cs']==-1 and f['cc']>=0.70),

    # 4. Price level + time window (our previously validated gate for reference)
    ("dn≥0.65 AND cd=150-300",
     lambda f: f['da']>=0.65 and f['cd'] is not None and 150<=f['cd']<=300),
]

print("="*90)
print(f"QUANT VALIDATION — {len(gates)} GATES ON FULL TICK DATA")
print(f"Current regime {len(regime_slugs)} mkts | IS={len(IS_sl)} | OOS={len(OOS_sl)}")
print(f"Bonferroni α=0.05/{N_BONF}={ALPHA:.4f} | {N_PERM} perms | {N_BOOT} bootstrap")
print("="*90)

results=[]
for i,(label,fn) in enumerate(gates):
    is_net,is_n,is_s,is_m=eval_gate(fires_IS, fn)
    oos_net,oos_n,_,_   =eval_gate(fires_OOS,fn)
    is_dn, is_up  =per_type(fires_IS, fn)
    oos_dn,oos_up =per_type(fires_OOS,fn)

    print(f"\n{'─'*90}")
    print(f"Gate {i+1}: {label}")
    print(f"  IS  [{len(IS_sl):>3}mkts {is_n:>5}blk]: "
          f"net=${is_net:>+9.2f}  DN=${is_dn:>+9.2f}  UP=${is_up:>+9.2f}")
    print(f"  OOS [{len(OOS_sl):>3}mkts {oos_n:>5}blk]: "
          f"net=${oos_net:>+9.2f}  DN=${oos_dn:>+9.2f}  UP=${oos_up:>+9.2f}")

    print(f"  Running perm ({N_PERM})...", end='', flush=True)
    pp      = perm_test(fires_all, fn)
    pp_oos  = perm_test(fires_OOS, fn)
    print(f" done")

    ci_lo,ci_hi,ci_m = boot_ci(fires_all, fn)
    ac = autocorr(fires_all, fn)

    bonf = pp < ALPHA; both = is_net>0 and oos_net>0
    if bonf and both:       v="✅✅ PROMOTE — Bonf+IS+OOS"
    elif pp<0.05 and both:  v="🟡 SHADOW+ — sig+IS+OOS"
    elif both:              v="⚠️  both+ not sig"
    elif is_net>0:          v="❌  IS only"
    else:                   v="❌  no edge"

    print(f"  p_perm={pp:.4f} {'✅Bonf' if bonf else ('✅sig' if pp<0.05 else '❌')}  "
          f"p_OOS={pp_oos:.4f} {'✅' if pp_oos<0.05 else '❌'}")
    print(f"  Bootstrap CI: mean=${ci_m:>+.2f}  95%=[${ci_lo:>+.2f}, ${ci_hi:>+.2f}]  "
          f"{'✅CI+' if ci_lo>0 else '⚠️ incl 0'}")
    print(f"  Autocorr lag-1: {ac:>+.3f}  {'✅IID' if abs(ac)<0.20 else '⚠️ correlated'}")
    print(f"  VERDICT: {v}")
    results.append((label,is_net,oos_net,pp,pp_oos,ci_lo,ci_hi,ci_m,v,is_n,oos_n))

print(f"\n{'='*90}")
print("FINAL RANKING")
print(f"{'#':<3} {'Gate':<38} {'IS':>9} {'OOS':>9} {'p_full':>8} {'p_OOS':>8} "
      f"{'CI_lo':>8} {'blk_IS':>7} {'blk_OOS':>8}  Verdict")
print("─"*110)
for i,(lbl,isn,oon,pf,po,clo,chi,cm,v,bn_is,bn_oos) in enumerate(results):
    print(f" {i+1}. {lbl:<36} {isn:>+9.2f} {oon:>+9.2f} {pf:>8.4f} {po:>8.4f} "
          f"{clo:>+8.2f} {bn_is:>7} {bn_oos:>8}  {v}")

print()
print("Done.")
