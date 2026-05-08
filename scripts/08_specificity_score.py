#!/usr/bin/env python3
"""
Compute G12D-vs-WT peptide specificity scores for designed binders.

For each designed binder sequence, compute ProteinMPNN log-likelihood on:
  - G12D complex (target with VVGADGVGK peptide)  → score_g12d
  - WT complex   (target with VVGAVGVGK peptide)  → score_wt

Specificity = score_wt - score_g12d
  > 0: binder sequence is more compatible with G12D backbone → G12D-selective
  < 0: binder sequence prefers WT backbone → non-specific

This works because the P5 backbone geometry differs slightly between ASP (G12D)
and VAL (WT), changing the packing environment at the interface.

Usage:
  python3 08_specificity_score.py \\
    --designs_json .../mpnn/all_designs.json \\
    --binder_dir .../rfdiffusion \\
    --output_dir .../specificity_scores
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np
import tempfile
from pathlib import Path

MPNN_DIR = "/workspace/ProteinMPNN"
DEFAULT_DESIGNS_JSON = "/workspace/pmhc_design/designs/mpnn/all_designs.json"
DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/specificity_scores"
TARGET_G12D = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"
TARGET_WT = "/workspace/pmhc_design/designs/target_9UV8_wt.pdb"

AA1TO3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs_json", default=DEFAULT_DESIGNS_JSON)
    parser.add_argument("--binder_dir", default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_g12d", default=TARGET_G12D)
    parser.add_argument("--target_wt", default=TARGET_WT)
    parser.add_argument("--max_designs", type=int, default=None)
    return parser.parse_args()


def read_pdb_atoms(pdb_file, chain_filter=None):
    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                if chain_filter is None or chain in chain_filter:
                    lines.append(line)
    return lines


def create_scored_complex_pdb(target_pdb, backbone_pdb, binder_seq, output_pdb):
    """
    Create complex PDB with designed binder sequence for ProteinMPNN scoring.
    Target chain A,B,C from target_pdb; binder backbone from backbone_pdb with
    residue names updated to binder_seq.
    """
    target_lines = read_pdb_atoms(target_pdb, chain_filter={'A', 'B', 'C'})
    backbone_d = read_pdb_atoms(backbone_pdb, chain_filter={'D'})

    resnums = []
    seen = set()
    for line in backbone_d:
        rnum = int(line[22:26])
        if rnum not in seen:
            resnums.append(rnum)
            seen.add(rnum)

    n = min(len(resnums), len(binder_seq))
    rnum_to_resname = {rnum: AA1TO3.get(aa, 'GLY')
                       for rnum, aa in zip(resnums[:n], binder_seq[:n])}

    updated_d = []
    for line in backbone_d:
        rnum = int(line[22:26])
        if rnum not in rnum_to_resname:
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        new_line = line[:17] + f"{rnum_to_resname[rnum]:<3}" + line[20:]
        updated_d.append(new_line)

    with open(output_pdb, 'w') as f:
        f.writelines(target_lines)
        f.write('TER\n')
        f.writelines(updated_d)
        f.write('TER\nEND\n')


def run_mpnn_score(pdb_files_dir, output_dir):
    """
    Run ProteinMPNN in score-only mode on all PDBs in pdb_files_dir.
    Chain D is the 'designed' chain (we're scoring it); A, B, C are fixed context.

    Returns: dict of {pdb_stem: float score}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: parse PDBs to JSONL
    parsed_jsonl = os.path.join(output_dir, "parsed.jsonl")
    result = subprocess.run([
        "/venv/main/bin/python3",
        f"{MPNN_DIR}/helper_scripts/parse_multiple_chains.py",
        "--input_path", pdb_files_dir,
        "--output_path", parsed_jsonl,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"parse_multiple_chains failed: {result.stderr[:300]}")
        return {}

    # Step 2: assign which chain to score (D)
    assigned_jsonl = os.path.join(output_dir, "assigned.jsonl")
    result = subprocess.run([
        "/venv/main/bin/python3",
        f"{MPNN_DIR}/helper_scripts/assign_fixed_chains.py",
        "--input_path", parsed_jsonl,
        "--output_path", assigned_jsonl,
        "--chain_list", "D",  # D is the designed/scored chain
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"assign_fixed_chains failed: {result.stderr[:300]}")
        return {}

    # Step 3: run ProteinMPNN in score-only mode
    result = subprocess.run([
        "/venv/main/bin/python3", f"{MPNN_DIR}/protein_mpnn_run.py",
        "--jsonl_path", parsed_jsonl,
        "--chain_id_jsonl", assigned_jsonl,
        "--out_folder", output_dir,
        "--score_only", "1",
        "--batch_size", "1",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"MPNN score_only failed: {result.stderr[:500]}")
        return {}

    # Parse score output from score_only/ npz files
    # ProteinMPNN --score_only writes {name}_pdb.npz to score_only/
    scores = {}
    for npz_file in glob.glob(os.path.join(output_dir, "score_only/*.npz")):
        # Filename: binder_5_s5_pdb.npz → stem: binder_5_s5
        basename = os.path.basename(npz_file)
        stem = basename.replace('_pdb.npz', '')
        try:
            d = np.load(npz_file)
            # 'score' is the per-chain D score (designed chain)
            # 'global_score' is the global score over all chains
            scores[stem] = float(d['score'][0])
        except Exception as e:
            print(f"  Failed to parse {npz_file}: {e}")

    return scores


def main():
    args = parse_args()

    print("=== G12D vs WT Specificity Scoring ===")

    with open(args.designs_json) as f:
        designs = json.load(f)

    designs.sort(key=lambda x: x.get('mpnn_score', 999))
    if args.max_designs:
        designs = designs[:args.max_designs]

    print(f"Scoring {len(designs)} designs")

    os.makedirs(args.output_dir, exist_ok=True)

    g12d_pdb_dir = os.path.join(args.output_dir, "complex_pdbs_g12d")
    wt_pdb_dir   = os.path.join(args.output_dir, "complex_pdbs_wt")
    g12d_score_dir = os.path.join(args.output_dir, "mpnn_scores_g12d")
    wt_score_dir   = os.path.join(args.output_dir, "mpnn_scores_wt")

    os.makedirs(g12d_pdb_dir, exist_ok=True)
    os.makedirs(wt_pdb_dir, exist_ok=True)

    # Prepare complex PDBs for both G12D and WT
    print("\nPreparing complex PDBs (G12D and WT)...")
    valid_designs = []
    for d in designs:
        backbone_name = d['backbone']
        binder_seq = d.get('binder_sequence', '')
        if not binder_seq or len(binder_seq) < 20:
            continue

        backbone_pdb = os.path.join(args.binder_dir, f"{backbone_name}.pdb")
        if not os.path.exists(backbone_pdb):
            continue

        design_name = f"{backbone_name}_s{d['sample']}"

        # Create G12D complex
        g12d_pdb = os.path.join(g12d_pdb_dir, f"{design_name}.pdb")
        create_scored_complex_pdb(args.target_g12d, backbone_pdb, binder_seq, g12d_pdb)

        # Create WT complex
        wt_pdb = os.path.join(wt_pdb_dir, f"{design_name}.pdb")
        create_scored_complex_pdb(args.target_wt, backbone_pdb, binder_seq, wt_pdb)

        valid_designs.append({
            **d,
            'design_name': design_name,
            'backbone_pdb': backbone_pdb,
        })

    print(f"Prepared {len(valid_designs)} design pairs")

    # Run ProteinMPNN in score-only mode on both sets
    print("\nRunning MPNN score-only on G12D complexes...")
    g12d_scores = run_mpnn_score(g12d_pdb_dir, g12d_score_dir)
    print(f"  Got scores for {len(g12d_scores)} designs")

    print("Running MPNN score-only on WT complexes...")
    wt_scores = run_mpnn_score(wt_pdb_dir, wt_score_dir)
    print(f"  Got scores for {len(wt_scores)} designs")

    # Compute specificity scores
    results = []
    for d in valid_designs:
        design_name = d['design_name']
        score_g12d = g12d_scores.get(design_name, None)
        score_wt   = wt_scores.get(design_name, None)

        if score_g12d is None or score_wt is None:
            continue

        # Specificity: positive = prefers G12D, negative = prefers WT
        specificity = score_wt - score_g12d

        result = {
            'design_name': design_name,
            'backbone': d['backbone'],
            'sample': d['sample'],
            'mpnn_score': d.get('mpnn_score', 999),
            'binder_sequence': d.get('binder_sequence', ''),
            'score_g12d': score_g12d,
            'score_wt': score_wt,
            'specificity': specificity,
        }
        results.append(result)

    # Sort by specificity (most G12D-selective first)
    results.sort(key=lambda x: -x['specificity'])

    print(f"\n{len(results)} designs scored")
    print(f"\n{'Design':<25} {'MPNN':>6} {'G12D':>8} {'WT':>8} {'Spec':>8} {'G12D-sel?':>10}")
    print('-' * 70)
    for r in results[:30]:
        flag = ' YES' if r['specificity'] > 0 else ' no'
        print(f"{r['design_name']:<25} "
              f"{r['mpnn_score']:>6.3f} "
              f"{r['score_g12d']:>8.4f} "
              f"{r['score_wt']:>8.4f} "
              f"{r['specificity']:>8.4f}"
              f"{flag}")

    n_specific = sum(1 for r in results if r['specificity'] > 0)
    print(f"\n{n_specific}/{len(results)} designs are G12D-selective (specificity > 0)")

    # Save
    out_file = os.path.join(args.output_dir, "specificity_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {out_file}")

    # Save top G12D-selective sequences
    fasta_out = os.path.join(args.output_dir, "top_g12d_selective.fa")
    g12d_selective = [r for r in results if r['specificity'] > 0]
    with open(fasta_out, 'w') as f:
        for r in g12d_selective[:50]:
            f.write(f">{r['design_name']} spec={r['specificity']:.4f} "
                    f"g12d={r['score_g12d']:.4f} wt={r['score_wt']:.4f}\n"
                    f"{r['binder_sequence']}\n")
    print(f"G12D-selective sequences: {fasta_out}")


if __name__ == '__main__':
    main()
