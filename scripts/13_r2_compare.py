#!/usr/bin/env python3
"""
Round-1 vs Round-2 comparison.

Aggregates AF2/spec results across all R2 seeds and compares with R1 qualifiers.

Usage:
  python3 13_r2_compare.py [--seeds b7_t03_s7 b7_t03_s2 b17_s1 b93_s6]
"""

import os, sys, json, glob, argparse
import numpy as np

DESIGN_DIR = "/workspace/pmhc_design/designs"

R1_RESULTS = f"{DESIGN_DIR}/r1_analysis_results.json"

R1_LABELS = {
    "binder_7_T03_s7":  {"priority": 1, "mechanism": "Asp@51→P5",       "seed": "binder_7"},
    "binder_7_T03_s2":  {"priority": 2, "mechanism": "Asp@51→P5",       "seed": "binder_7"},
    "binder_17_s1":     {"priority": 3, "mechanism": "Arg@44 salt-bridge","seed": "binder_17"},
    "binder_7_s6":      {"priority": 4, "mechanism": "Gln@51→P5",        "seed": "binder_7"},
    "binder_93_s6":     {"priority": 5, "mechanism": "Arg@55 salt-bridge","seed": "binder_93"},
    "binder_7_s14":     {"priority": 6, "mechanism": "Gln@51→P5",        "seed": "binder_7"},
    "binder_7_s4":      {"priority": 7, "mechanism": "Gln@51→P5",        "seed": "binder_7"},
    "binder_7_s3":      {"priority": 8, "mechanism": "Gln@51→P5",        "seed": "binder_7"},
    "binder_86_s7":     {"priority": 9, "mechanism": "Gln@59→P5",        "seed": "binder_86"},
}

CHARGE_MAP = {"K":+1,"R":+1,"H":+0.5,"D":-1,"E":-1}
HYDROPHOBIC = set("VILMFYWAC")

def net_charge(seq): return sum(CHARGE_MAP.get(aa,0) for aa in seq)
def poly_ala(seq):
    mx, cur = 0, 0
    for aa in seq:
        cur = cur+1 if aa=="A" else 0
        mx = max(mx,cur)
    return mx
def frac_hydrophobic(seq): return sum(1 for aa in seq if aa in HYDROPHOBIC)/len(seq) if seq else 0


def load_r1():
    if not os.path.exists(R1_RESULTS):
        print(f"WARNING: R1 results not found at {R1_RESULTS}")
        return []
    with open(R1_RESULTS) as f:
        return json.load(f)


def load_r2(seeds):
    all_results = []
    for seed in seeds:
        af2_file = os.path.join(DESIGN_DIR, f"r2_{seed}", "af2_results.json")
        scored_file = os.path.join(DESIGN_DIR, f"r2_{seed}", "scored_designs.json")

        if os.path.exists(af2_file):
            with open(af2_file) as f:
                data = json.load(f)
            for d in data:
                d['_r2_seed'] = seed
                d['_has_af2'] = True
            all_results.extend(data)
        elif os.path.exists(scored_file):
            with open(scored_file) as f:
                data = json.load(f)
            for d in data:
                d['_r2_seed'] = seed
                d['_has_af2'] = False
            all_results.extend(data)

    return all_results


def sequence_identity(seq1, seq2):
    if not seq1 or not seq2:
        return 0.0
    l = min(len(seq1), len(seq2))
    return sum(a==b for a,b in zip(seq1[:l],seq2[:l])) / l


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs='+',
                   default=["b7_t03_s7","b7_t03_s2","b17_s1","b93_s6"],
                   help="R2 seeds to include")
    p.add_argument("--plddt_thresh", type=float, default=67.0)
    p.add_argument("--spec_thresh",  type=float, default=0.025)
    args = p.parse_args()

    r1 = load_r1()
    r2 = load_r2(args.seeds)

    print("=" * 100)
    print("ROUND-1 vs ROUND-2 COMPARISON")
    print("=" * 100)

    # ── R1 summary ─────────────────────────────────────────────────────────────
    print(f"\n{'━'*100}")
    print(f"ROUND 1  ({len(r1)} qualifiers)")
    print(f"{'━'*100}")
    hdr = (f"{'#':>2}  {'Label':24s}  {'pLDDT':>6}  {'pAE':>6}  {'Spec':>7}  "
           f"{'RMSD':>5}  {'Chg':>4}  {'Helix':>6}  {'polyA':>5}  {'Mechanism'}")
    print(hdr)
    print("-"*100)
    for r in sorted(r1, key=lambda x: x.get('priority') or 99):
        def f(v, fmt=".2f"): return format(v, fmt) if v is not None else "—"
        pr = r.get('priority') or 0
        print(
            f"{pr:>2}  {r.get('label', r.get('seq','?')[:20]):24s}"
            f"  {f(r.get('pLDDT')):>6}"
            f"  {f(r.get('pae_interaction')):>6}"
            f"  {r.get('spec',0):>+7.3f}"
            f"  {f(r.get('rmsd')):>5}"
            f"  {r.get('net_charge',0):>+4.0f}"
            f"  {f(r.get('frac_helix'),'.0%') if r.get('frac_helix') is not None else '—':>6}"
            f"  {r.get('poly_ala',0):>5}"
            f"  {r.get('mechanism','—')}"
        )

    # ── R2 summary ─────────────────────────────────────────────────────────────
    print(f"\n{'━'*100}")
    print(f"ROUND 2  ({len(r2)} total designs across seeds: {args.seeds})")
    print(f"{'━'*100}")

    if not r2:
        print("  No R2 results available yet.")
        return

    passing_r2 = [d for d in r2
                  if d.get('binder_plddt', 0) >= args.plddt_thresh
                  and d.get('specificity', -999) >= args.spec_thresh
                  and d.get('_has_af2', False)]

    print(f"  AF2-evaluated  : {sum(1 for d in r2 if d.get('_has_af2'))}")
    print(f"  Passing R2     : {len(passing_r2)} "
          f"(pLDDT ≥ {args.plddt_thresh}, spec ≥ {args.spec_thresh})")

    by_seed = {}
    for d in r2:
        s = d.get('_r2_seed','?')
        by_seed.setdefault(s,[]).append(d)

    for seed in args.seeds:
        designs = by_seed.get(seed,[])
        n_af2   = sum(1 for d in designs if d.get('_has_af2'))
        n_pass  = sum(1 for d in designs
                      if d.get('binder_plddt',0) >= args.plddt_thresh
                      and d.get('specificity',-999) >= args.spec_thresh
                      and d.get('_has_af2'))
        specs   = [d['specificity'] for d in designs if 'specificity' in d]
        spec_str = f"spec {min(specs):.3f}–{max(specs):.3f}" if specs else "—"
        print(f"  {seed:20s}: {len(designs):4d} designs, {n_af2:3d} AF2-run, "
              f"{n_pass:3d} passing  ({spec_str})")

    if passing_r2:
        print(f"\n  Top R2 passing designs:")
        hdr2 = (f"  {'Design':32s}  {'Seed':20s}  {'pLDDT':>6}  {'pAE':>6}  "
                f"{'Spec':>7}  {'RMSD':>5}  {'Chg':>4}  {'polyA':>5}")
        print(hdr2)
        print("  " + "-"*95)
        passing_r2.sort(key=lambda x: -x.get('binder_plddt',0))
        for d in passing_r2[:20]:
            seq = d.get('binder_sequence','')
            print(
                f"  {d.get('design_name',d.get('backbone','?')):32s}"
                f"  {d.get('_r2_seed','?'):20s}"
                f"  {d.get('binder_plddt',0):>6.2f}"
                f"  {d.get('pae_interaction',999):>6.2f}"
                f"  {d.get('specificity',0):>+7.3f}"
                f"  {d.get('binder_rmsd',999):>5.1f}"
                f"  {net_charge(seq):>+4.0f}"
                f"  {poly_ala(seq):>5}"
            )

    # ── Sequence comparison ────────────────────────────────────────────────────
    r1_seqs = {r.get('label','?'): r.get('seq','') for r in r1 if r.get('seq')}
    r2_seqs = {d.get('design_name','?'): d.get('binder_sequence','')
               for d in r2 if d.get('binder_sequence')}

    if r1_seqs and r2_seqs:
        print(f"\n{'━'*100}")
        print("SEQUENCE NOVELTY (R2 vs R1 — max identity to any R1 sequence)")
        print(f"{'━'*100}")

        r2_top = [d for d in r2 if d.get('_has_af2') and d.get('binder_plddt',0) >= args.plddt_thresh]
        r2_top.sort(key=lambda x: -x.get('specificity',-999))

        if r2_top:
            print(f"  {'R2 Design':34s} {'pLDDT':>6} {'Spec':>7} {'MaxID_R1':>9} {'Closest_R1'}")
            print(f"  {'-'*80}")
            for d in r2_top[:20]:
                seq2 = d.get('binder_sequence','')
                max_id = 0
                closest = "—"
                for lab, seq1 in r1_seqs.items():
                    sid = sequence_identity(seq1, seq2)
                    if sid > max_id:
                        max_id = sid; closest = lab
                print(f"  {d.get('design_name','?'):34s}"
                      f" {d.get('binder_plddt',0):>6.2f}"
                      f" {d.get('specificity',0):>+7.3f}"
                      f" {max_id:>9.1%}"
                      f" {closest}")
        else:
            print("  No R2 AF2-evaluated designs with pLDDT above threshold yet.")

    # ── Save comparison JSON ───────────────────────────────────────────────────
    out_file = os.path.join(DESIGN_DIR, "r2_comparison.json")
    with open(out_file, 'w') as f:
        json.dump({
            "r1": r1,
            "r2": r2,
            "r2_passing": passing_r2,
        }, f, indent=2, default=float)
    print(f"\nComparison saved to {out_file}")


if __name__ == '__main__':
    main()
