#!/usr/bin/env python3
"""
UP Gate PnL Maximizer
=======================
Instead of minimizing ratio, MAXIMIZE total PnL.
A gate with 100:40 ratio (2.5:1) blocking 140 fires might beat
a gate with 21:0 ratio blocking only 21 fires.

Tests EVERY possible gate and ranks by ACTUAL PnL, not ratio.
Also shows the entry-price-weighted analysis for each gate.

Run: python3 up_gate_maximizer.py market_history.jsonl
"""
import json, sys

CD,UP_BID,UP_ASK,DN_BID,DN_ASK = 0,1,2,3,4
BN_DELTA = 12

def load(fp):
    mkts = []
    with open(fp) as f:
        for line in f:
            if not line.strip(): continue
            m = json.loads(line)
            if m.get('ticks') and m.get('winner') in ('UP','DN'):
                mkts.append(m)
    mkts.sort(key=lambda m: m.get('ts',''))
    return mkts

def sample_Ns(ticks, idx, interval):
    s = []; prev = 999
    for t in ticks:
        if t[CD] < prev - (interval - 0.5):
            s.append((t[CD], t[idx] or 0))
            prev = t[CD]
    return s

def simulate_up_gate(markets, interval, dn_thresh, dn_pct, fire_cd):
    """Returns detailed per-fire PnL breakdown."""
    total = 0; blocked_saves = 0; blocked_misses = 0
    blocked = 0; blocked_won = 0; fires = 0
    blocked_loss_entries = []; blocked_win_entries = []
    allowed_win_entries = []; allowed_loss_entries = []

    for m in markets:
        ticks = m['ticks']
        winner = m['winner']
        up_s = sample_Ns(ticks, UP_ASK, interval)
        dn_s = sample_Ns(ticks, DN_ASK, interval)

        # Gate check: DN≥thresh for pct% before checkpoint
        gate = False
        before = [p for cd, p in dn_s if cd >= fire_cd]
        if len(before) >= 2:
            above = sum(1 for p in before if p >= dn_thresh)
            if (above / len(before) * 100) >= dn_pct:
                gate = True

        # Simulate UP fires after checkpoint
        fired = set()
        for cd, price in up_s:
            if cd < 30 or price <= 0: continue
            if cd >= fire_cd: continue
            zone = round(price * 20) / 20
            if zone in fired: continue
            fired.add(zone)
            won = (winner == 'UP')
            pnl = ((1.0 - price) * 5) if won else (-price * 5)
            fires += 1

            if gate:
                blocked += 1
                if won:
                    blocked_won += 1
                    blocked_misses += abs(pnl)  # profit we missed
                    blocked_win_entries.append(price)
                else:
                    blocked_saves += abs(pnl)  # loss we avoided
                    blocked_loss_entries.append(price)
            else:
                total += pnl
                if won:
                    allowed_win_entries.append(price)
                else:
                    allowed_loss_entries.append(price)

    return {
        'pnl': total, 'fires': fires,
        'blocked': blocked, 'blocked_won': blocked_won,
        'blocked_lost': blocked - blocked_won,
        'blocked_saves': blocked_saves,    # $ saved from prevented losses
        'blocked_misses': blocked_misses,  # $ missed from blocked wins
        'net_block_impact': blocked_saves - blocked_misses,
        'avg_blocked_loss_entry': sum(blocked_loss_entries)/len(blocked_loss_entries) if blocked_loss_entries else 0,
        'avg_blocked_win_entry': sum(blocked_win_entries)/len(blocked_win_entries) if blocked_win_entries else 0,
        'avg_allowed_win_entry': sum(allowed_win_entries)/len(allowed_win_entries) if allowed_win_entries else 0,
        'avg_allowed_loss_entry': sum(allowed_loss_entries)/len(allowed_loss_entries) if allowed_loss_entries else 0,
    }

def simulate_up_gate_bn(markets, interval, bn_thresh, bn_pct, fire_cd):
    total = 0; blocked = 0; blocked_won = 0
    blocked_saves = 0; blocked_misses = 0; fires = 0

    for m in markets:
        ticks = m['ticks']
        winner = m['winner']
        up_s = sample_Ns(ticks, UP_ASK, interval)
        bn_s = sample_Ns(ticks, BN_DELTA, interval)

        gate = False
        before = [p for cd, p in bn_s if cd >= fire_cd]
        if len(before) >= 2:
            neg = sum(1 for b in before if b < bn_thresh)
            if (neg / len(before) * 100) >= bn_pct:
                gate = True

        fired = set()
        for cd, price in up_s:
            if cd < 30 or price <= 0: continue
            if cd >= fire_cd: continue
            zone = round(price * 20) / 20
            if zone in fired: continue
            fired.add(zone)
            won = (winner == 'UP')
            pnl = ((1.0 - price) * 5) if won else (-price * 5)
            fires += 1

            if gate:
                blocked += 1
                if won:
                    blocked_won += 1
                    blocked_misses += abs(pnl)
                else:
                    blocked_saves += abs(pnl)
            else:
                total += pnl

    return {
        'pnl': total, 'fires': fires,
        'blocked': blocked, 'blocked_won': blocked_won,
        'blocked_lost': blocked - blocked_won,
        'blocked_saves': blocked_saves,
        'blocked_misses': blocked_misses,
        'net_block_impact': blocked_saves - blocked_misses,
    }

def main():
    fp = sys.argv[1] if len(sys.argv) > 1 else "market_history.jsonl"
    print(f"Loading {fp}...")
    raw = load(fp)
    n = len(raw)
    split = int(n * 0.68)
    train = raw[:split]
    test = raw[split:]
    print(f"Loaded {n} markets (Train={len(train)} Test={len(test)})")

    intervals = [3, 5, 7, 10, 15, 20, 30]

    for ds_name, dataset in [("OOS", test), ("TRAIN", train)]:
        # Baseline
        base_pnl = 0
        for m in dataset:
            up_s = sample_Ns(m['ticks'], UP_ASK, 10)
            fired = set()
            for cd, price in up_s:
                if cd < 30 or price <= 0: continue
                zone = round(price * 20) / 20
                if zone in fired: continue
                fired.add(zone)
                won = (m['winner'] == 'UP')
                base_pnl += ((1.0 - price) * 5) if won else (-price * 5)

        print(f"\n{'='*140}")
        print(f"UP GATE MAXIMIZER — {ds_name} — Baseline: ${base_pnl:+.0f}")
        print(f"{'='*140}")

        # DN price gates
        results = []
        for iv in intervals:
            for thresh_int in range(50, 81, 5):
                for pct in [30, 40, 50, 60, 70]:
                    for fire_cd in [250, 230, 210]:
                        r = simulate_up_gate(dataset, iv, thresh_int/100, pct, fire_cd)
                        if r['blocked'] < 3: continue
                        r['vs'] = r['pnl'] - base_pnl
                        r['name'] = f"int={iv}s DN≥${thresh_int/100:.2f} p{pct}% cd={fire_cd}"
                        r['interval'] = iv
                        r['ratio'] = f"{r['blocked_lost']}:{r['blocked_won']}"
                        results.append(r)

        # BN gates
        for iv in intervals:
            for bn_t in [-0.01, -0.02, -0.03, -0.05]:
                for bn_p in [30, 40, 50, 60, 70, 80]:
                    for fire_cd in [250, 230, 210]:
                        r = simulate_up_gate_bn(dataset, iv, bn_t, bn_p, fire_cd)
                        if r['blocked'] < 3: continue
                        r['vs'] = r['pnl'] - base_pnl
                        r['name'] = f"int={iv}s BN<{bn_t} p{bn_p}% cd={fire_cd}"
                        r['interval'] = iv
                        r['ratio'] = f"{r['blocked_lost']}:{r['blocked_won']}"
                        results.append(r)

        # Sort by ACTUAL PnL improvement (not ratio)
        results.sort(key=lambda r: -r['vs'])

        print(f"  Tested {len(results)} combos")
        print(f"\n  TOP 40 BY PnL (not ratio):")
        print(f"  {'#':<4}{'Config':<45}{'vs':>7}{'Blk':>6}{'L:W':>8}"
              f"{'$saved':>8}{'$missed':>9}{'net$':>8}{'Fires':>7}")
        print(f"  {'─'*100}")

        shown = set()
        count = 0
        for r in results:
            if count >= 40: break
            if r['name'] in shown: continue
            shown.add(r['name'])
            count += 1
            print(f"  {count:<4}{r['name']:<45}${r['vs']:>+5.0f}  {r['blocked']:>4}  {r['ratio']:>6}"
                  f"  ${r['blocked_saves']:>+6.0f}  ${r['blocked_misses']:>+7.0f}"
                  f"  ${r['net_block_impact']:>+6.0f}  {r['fires']:>5}")

        # Show entry price analysis for top 5
        top5 = [r for r in results if 'avg_blocked_loss_entry' in r][:5]
        if top5:
            print(f"\n  ENTRY PRICE ANALYSIS (top 5):")
            print(f"  {'Config':<45}{'BlkLoss$':>9}{'BlkWin$':>9}{'AllowW$':>9}{'AllowL$':>9}")
            print(f"  {'─'*85}")
            for r in top5:
                print(f"  {r['name']:<45}"
                      f"${r.get('avg_blocked_loss_entry',0):>7.3f}"
                      f"  ${r.get('avg_blocked_win_entry',0):>7.3f}"
                      f"  ${r.get('avg_allowed_win_entry',0):>7.3f}"
                      f"  ${r.get('avg_allowed_loss_entry',0):>7.3f}")

        # Compare: tight (21:0) vs best wide vs best BN
        print(f"\n  HEAD TO HEAD COMPARISON:")
        tight = [r for r in results if r['blocked_won'] == 0 and r['blocked'] >= 10]
        wide = [r for r in results if r['blocked'] >= 50 and r['blocked_won'] > 0]
        bn_gates = [r for r in results if 'BN<' in r['name']]

        for cat, clist in [("Best tight (0 wins missed)", tight),
                           ("Best wide (50+ blocks)", wide),
                           ("Best BN-based", bn_gates)]:
            if clist:
                best = max(clist, key=lambda r: r['vs'])
                print(f"    {cat}:")
                print(f"      {best['name']}")
                print(f"      PnL: ${best['vs']:+.0f} | Blocks: {best['blocked']} ({best['ratio']}) "
                      f"| $saved: ${best['blocked_saves']:+.0f} | $missed: ${best['blocked_misses']:+.0f}")

if __name__ == "__main__":
    main()
