#!/usr/bin/env python3
"""
Score and rank RFdiffusion backbones by interface geometry quality.

Metrics:
1. n_pep_contacts: Number of binder residues within 8Å of peptide (heavy atom-like via Cβ)
2. p5_distance: Closest binder Cβ distance to P5/G12D residue (lower = better specificity potential)
3. p4_distance: Closest binder Cβ distance to P4/ALA
4. p7_distance: Closest binder Cβ distance to P7/VAL
5. hotspot_coverage: Fraction of hotspot residues (P1,P4,P5,P7) contacted within 8Å
6. pep_frac: Fraction of binder residues in peptide interface
7. mhc_frac: Fraction of binder residues in MHC interface (we want LOW mhc, HIGH pep)
8. polar_needed_at_P5: Number of binder residues near P5 that could be polar

Usage:
  python3 07_score_backbones.py \\
    --binder_dir .../rfdiffusion \\
    --target_pdb .../target_9UV8_clean.pdb \\
    --output_dir .../backbone_scores
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_TARGET_PDB = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/backbone_scores"

HOTSPOT_PEP_RESNUMS = {1, 4, 5, 7}  # C1, C4, C5(G12D), C7

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binder_dir", default=DEFAULT_BINDER_DIR)
    parser.add_argument("--target_pdb", default=DEFAULT_TARGET_PDB)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="binder_*.pdb")
    parser.add_argument("--contact_cutoff", type=float, default=8.0,
                        help="Cβ-Cβ distance cutoff for contacts (Å)")
    return parser.parse_args()


def get_cb_positions(pdb_file, chain):
    """
    Get Cβ positions for each residue in a chain.
    For GLY (no Cβ), use Cα instead.
    Returns dict: resnum -> {'cb': np.array, 'ca': np.array, 'resname': str}
    """
    atoms = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain:
                continue
            rnum = int(line[22:26])
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

            if rnum not in atoms:
                atoms[rnum] = {'ca': None, 'cb': None, 'resname': resname}
            if atom == 'CA':
                atoms[rnum]['ca'] = coord
            elif atom == 'CB':
                atoms[rnum]['cb'] = coord

    # For GLY or missing Cβ, use Cα
    for rnum in atoms:
        if atoms[rnum]['cb'] is None:
            atoms[rnum]['cb'] = atoms[rnum]['ca']

    return atoms


def score_backbone(binder_pdb, target_pdb, cutoff=8.0):
    """Score a binder backbone by interface geometry."""
    # Get peptide (chain C) Cβ positions
    pep_atoms = get_cb_positions(target_pdb, 'C')
    if not pep_atoms:
        return None

    pep_resnums = sorted(pep_atoms.keys())
    pep_seq = ''.join(AA3TO1.get(pep_atoms[r]['resname'], 'X')
                      for r in pep_resnums)

    # Get MHC heavy chain (A) Cβ positions (just groove residues, say 1-180)
    mhc_atoms = {}
    with open(target_pdb) as f:
        for line in f:
            if not line.startswith('ATOM') or line[21] != 'A':
                continue
            rnum = int(line[22:26])
            if rnum > 180:  # Only groove region
                continue
            atom = line[12:16].strip()
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rnum not in mhc_atoms:
                mhc_atoms[rnum] = {'ca': None, 'cb': None}
            if atom == 'CA':
                mhc_atoms[rnum]['ca'] = coord
            elif atom == 'CB':
                mhc_atoms[rnum]['cb'] = coord
    for r in mhc_atoms:
        if mhc_atoms[r]['cb'] is None:
            mhc_atoms[r]['cb'] = mhc_atoms[r]['ca']

    # Get binder (chain D) Cβ positions
    binder_atoms = get_cb_positions(binder_pdb, 'D')
    if not binder_atoms:
        return None

    binder_resnums = sorted(binder_atoms.keys())
    binder_len = len(binder_resnums)

    # Build coordinate arrays
    pep_cb = np.array([pep_atoms[r]['cb'] for r in pep_resnums
                       if pep_atoms[r]['cb'] is not None])
    pep_resnum_arr = [r for r in pep_resnums if pep_atoms[r]['cb'] is not None]

    mhc_cb = np.array([mhc_atoms[r]['cb'] for r in sorted(mhc_atoms)
                       if mhc_atoms[r]['cb'] is not None])

    binder_cb = np.array([binder_atoms[r]['cb'] for r in binder_resnums
                          if binder_atoms[r]['cb'] is not None])
    binder_resnums_valid = [r for r in binder_resnums
                            if binder_atoms[r]['cb'] is not None]

    if len(pep_cb) == 0 or len(binder_cb) == 0:
        return None

    # Compute binder-peptide distances
    # Shape: (n_binder, n_pep)
    diff_pep = binder_cb[:, None, :] - pep_cb[None, :, :]
    dists_pep = np.linalg.norm(diff_pep, axis=2)

    # Contacts at cutoff
    binder_pep_contacts = (dists_pep <= cutoff)
    n_binder_contacting = binder_pep_contacts.any(axis=1).sum()
    n_pep_contacted = binder_pep_contacts.any(axis=0).sum()

    # Minimum distance from each binder position to each peptide position
    min_dist_to_pep = dists_pep.min(axis=1)
    min_pep_pos = np.array([pep_resnum_arr[np.argmin(dists_pep[i])]
                             for i in range(len(binder_resnums_valid))])

    # Hotspot coverage
    hotspot_contacted = set()
    for i, (bpos, bd) in enumerate(zip(binder_resnums_valid, binder_cb)):
        for j, ppos in enumerate(pep_resnum_arr):
            if dists_pep[i, j] <= cutoff and ppos in HOTSPOT_PEP_RESNUMS:
                hotspot_contacted.add(ppos)

    # Per-hotspot closest distances
    p5_idx = pep_resnum_arr.index(5) if 5 in pep_resnum_arr else None
    p4_idx = pep_resnum_arr.index(4) if 4 in pep_resnum_arr else None
    p7_idx = pep_resnum_arr.index(7) if 7 in pep_resnum_arr else None

    p5_min_dist = float(dists_pep[:, p5_idx].min()) if p5_idx is not None else 999
    p4_min_dist = float(dists_pep[:, p4_idx].min()) if p4_idx is not None else 999
    p7_min_dist = float(dists_pep[:, p7_idx].min()) if p7_idx is not None else 999

    # Find which binder positions are near P5 (within 12Å)
    near_p5_positions = []
    if p5_idx is not None:
        for i, bpos in enumerate(binder_resnums_valid):
            if dists_pep[i, p5_idx] <= 12.0:
                bidx = binder_resnums.index(bpos) if bpos in binder_resnums else i
                near_p5_positions.append({
                    'binder_idx': bidx,
                    'binder_rnum': bpos,
                    'dist_to_p5': float(dists_pep[i, p5_idx]),
                })
        near_p5_positions.sort(key=lambda x: x['dist_to_p5'])

    # MHC contacts (to check binder doesn't purely interact with MHC)
    n_mhc_contacting = 0
    if len(mhc_cb) > 0:
        diff_mhc = binder_cb[:, None, :] - mhc_cb[None, :, :]
        dists_mhc = np.linalg.norm(diff_mhc, axis=2)
        n_mhc_contacting = (dists_mhc.min(axis=1) <= cutoff).sum()

    # Compute the "peptide specificity ratio" (pep contacts / total contacts)
    n_total_contact = max(n_binder_contacting + n_mhc_contacting, 1)
    pep_specificity = n_binder_contacting / n_total_contact

    return {
        'binder_len': binder_len,
        'n_binder_contacting_pep': int(n_binder_contacting),
        'n_pep_residues_contacted': int(n_pep_contacted),
        'pep_frac': round(n_binder_contacting / max(binder_len, 1), 3),
        'n_mhc_contacting': int(n_mhc_contacting),
        'mhc_frac': round(n_mhc_contacting / max(binder_len, 1), 3),
        'pep_specificity_ratio': round(pep_specificity, 3),
        'hotspot_coverage': round(len(hotspot_contacted) / len(HOTSPOT_PEP_RESNUMS), 3),
        'hotspots_contacted': sorted(hotspot_contacted),
        'p4_min_dist': round(p4_min_dist, 2),
        'p5_min_dist': round(p5_min_dist, 2),
        'p7_min_dist': round(p7_min_dist, 2),
        'near_p5_positions': near_p5_positions,
        'pep_seq': pep_seq,
    }


def main():
    args = parse_args()

    print("=== RFdiffusion Backbone Quality Scoring ===")

    binder_pdbs = sorted(glob.glob(os.path.join(args.binder_dir, args.pattern)))
    print(f"Found {len(binder_pdbs)} backbone PDBs")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for pdb in binder_pdbs:
        name = os.path.splitext(os.path.basename(pdb))[0]
        score = score_backbone(pdb, args.target_pdb, cutoff=args.contact_cutoff)
        if score is None:
            print(f"  {name}: failed to score")
            continue
        score['name'] = name
        score['pdb'] = pdb
        results.append(score)

    # Sort by composite quality score:
    # Higher peptide contacts, lower P5 distance, higher hotspot coverage
    def quality_score(r):
        # Composite: maximize pep contacts, minimize P5 dist
        return -(r['n_binder_contacting_pep'] * 2 +
                 r['hotspot_coverage'] * 5 -
                 r['p5_min_dist'] * 0.5)

    results.sort(key=quality_score)

    # Print table
    print(f"\n{'Backbone':<20} {'Len':>4} {'nCont':>6} {'nPep':>5} {'PepF':>5} {'HsCov':>6} "
          f"{'P4':>5} {'P5':>5} {'P7':>5} {'PepSpec':>8}")
    print('-' * 80)
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['binder_len']:>4} "
              f"{r['n_binder_contacting_pep']:>6} "
              f"{r['n_pep_residues_contacted']:>5} "
              f"{r['pep_frac']:>5.2f} "
              f"{r['hotspot_coverage']:>6.2f} "
              f"{r['p4_min_dist']:>5.1f} "
              f"{r['p5_min_dist']:>5.1f} "
              f"{r['p7_min_dist']:>5.1f} "
              f"{r['pep_specificity_ratio']:>8.3f}")

    print(f"\nTop 5 by interface quality:")
    for r in results[:5]:
        print(f"\n  {r['name']} (len={r['binder_len']})")
        print(f"    Pep contacts: {r['n_binder_contacting_pep']} ({r['pep_frac']*100:.0f}% of binder)")
        print(f"    Hotspot coverage: {r['hotspot_coverage']*100:.0f}% ({r['hotspots_contacted']})")
        print(f"    P5(G12D) distance: {r['p5_min_dist']:.2f}Å")
        print(f"    Positions near P5 (within 12Å): {len(r['near_p5_positions'])}")
        if r['near_p5_positions']:
            print(f"      Closest: pos {r['near_p5_positions'][0]['binder_idx']+1} "
                  f"(d={r['near_p5_positions'][0]['dist_to_p5']:.2f}Å)")

    # Save results
    out_file = os.path.join(args.output_dir, "backbone_scores.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to: {out_file}")

    # Write prioritized list for MPNN
    top_backbones = [r['name'] for r in results[:20]]
    priority_file = os.path.join(args.output_dir, "priority_backbones.txt")
    with open(priority_file, 'w') as f:
        for name in top_backbones:
            f.write(name + '\n')
    print(f"Priority backbones: {priority_file}")


if __name__ == '__main__':
    main()
