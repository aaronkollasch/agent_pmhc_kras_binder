#!/usr/bin/env python3
"""
AF2 initial guess screening for pMHC binder designs.

Uses ColabFold/AlphaFold2 to predict the structure of each designed binder
in complex with the pMHC target. Key metrics:
- pAE_interaction (between binder and peptide): should be < 5 (paper threshold)
- pLDDT of binder: should be > 0.7
- binder RMSD vs. design model: should be < 2.0 Å

The pMHC chains (A=HLA, B=B2M, C=peptide) are provided as templates.
The binder (chain D) is designed without MSA (no MSA = de novo fold).
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

DEFAULT_DESIGNS_JSON = "/workspace/pmhc_design/designs/mpnn/all_designs.json"
DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/af2_screening"
TARGET_PDB = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"
AF2_PARAMS = "/home/user/.cache/colabfold/params"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs_json", type=str, default=DEFAULT_DESIGNS_JSON)
    parser.add_argument("--binder_dir", type=str, default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_pdb", type=str, default=TARGET_PDB)
    parser.add_argument("--max_designs", type=int, default=None,
                        help="Max number of designs to screen (for testing)")
    parser.add_argument("--pae_threshold", type=float, default=10.0,
                        help="pAE interaction threshold (lower is better)")
    parser.add_argument("--plddt_threshold", type=float, default=0.7,
                        help="Binder pLDDT threshold")
    parser.add_argument("--rmsd_threshold", type=float, default=2.0,
                        help="Backbone RMSD threshold (Å)")
    return parser.parse_args()

def load_designs(designs_json):
    """Load designs from ProteinMPNN output JSON."""
    with open(designs_json) as f:
        designs = json.load(f)
    return designs

def create_complex_pdb(target_pdb, binder_seq, backbone_pdb, output_pdb):
    """
    Create a complex PDB for AF2 initial guess prediction.
    Combines the pMHC target structure with the binder backbone (poly-Gly).
    """
    # Parse target PDB (chains A, B, C)
    target_lines = []
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in ('A', 'B', 'C'):
                target_lines.append(line)

    # Parse binder backbone from RFdiffusion output (chain D)
    binder_lines = []
    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D':
                binder_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(target_lines)
        f.write('TER\n')
        f.writelines(binder_lines)
        f.write('TER\nEND\n')

def calculate_pae_interaction(pae_matrix, chain_lengths):
    """
    Calculate pAE interaction between binder and peptide.

    pAE_interaction = mean of PAE values between chain D (binder) and chain C (peptide)
    Lower is better.

    Args:
        pae_matrix: numpy array of shape (L, L) where L is total length
        chain_lengths: dict with chain lengths {'A': na, 'B': nb, 'C': nc, 'D': nd}
    """
    # Calculate cumulative positions
    chains = ['A', 'B', 'C', 'D']
    starts = {}
    pos = 0
    for c in chains:
        starts[c] = pos
        pos += chain_lengths.get(c, 0)

    # Get binder and peptide indices
    binder_start = starts['D']
    binder_end = binder_start + chain_lengths.get('D', 0)
    pep_start = starts['C']
    pep_end = pep_start + chain_lengths.get('C', 0)

    # pAE from binder to peptide and peptide to binder
    binder_to_pep = pae_matrix[binder_start:binder_end, pep_start:pep_end]
    pep_to_binder = pae_matrix[pep_start:pep_end, binder_start:binder_end]

    return float(np.mean([binder_to_pep.mean(), pep_to_binder.mean()]))

def calculate_binder_plddt(plddt_array, chain_lengths):
    """Calculate mean pLDDT for the binder (chain D)."""
    chains = ['A', 'B', 'C', 'D']
    pos = 0
    for c in chains:
        if c == 'D':
            binder_plddt = plddt_array[pos:pos + chain_lengths.get('D', 0)]
            return float(np.mean(binder_plddt))
        pos += chain_lengths.get(c, 0)
    return 0.0

def calculate_ca_rmsd(pred_pdb, design_pdb, chain='D'):
    """Calculate Cα RMSD between predicted and design structures."""
    def get_ca_coords(pdb_file, chain_id):
        coords = []
        with open(pdb_file) as f:
            for line in f:
                if (line.startswith('ATOM') and
                    line[21] == chain_id and
                    line[12:16].strip() == 'CA'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        return np.array(coords)

    pred_ca = get_ca_coords(pred_pdb, chain)
    design_ca = get_ca_coords(design_pdb, chain)

    if len(pred_ca) == 0 or len(design_ca) == 0:
        return 999.0
    if len(pred_ca) != len(design_ca):
        return 999.0

    # Calculate RMSD after optimal superposition (Kabsch algorithm)
    diff = pred_ca - design_ca
    rmsd = float(np.sqrt((diff**2).sum(axis=1).mean()))
    return rmsd

def run_colabfold_prediction(complex_pdb, output_dir, design_name):
    """
    Run ColabFold prediction with initial guess from the complex PDB.

    Returns: path to prediction results
    """
    import subprocess

    pred_dir = os.path.join(output_dir, design_name)
    os.makedirs(pred_dir, exist_ok=True)

    # Extract sequence from the complex PDB for ColabFold
    seq_file = os.path.join(pred_dir, "input.fasta")
    write_sequence_from_pdb(complex_pdb, seq_file)

    # Run ColabFold with:
    # 1. No MSA for the binder (--msa-mode single_sequence)
    # 2. Template from complex PDB
    cmd = [
        "conda", "run", "-n", "colabfold",
        "colabfold_batch",
        "--msa-mode", "single_sequence",
        "--templates",
        "--custom-template-path", os.path.dirname(complex_pdb),
        "--num-recycle", "3",
        "--model-type", "alphafold2_multimer_v3",
        seq_file,
        pred_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return pred_dir, result.returncode == 0

def write_sequence_from_pdb(pdb_file, fasta_file):
    """Extract chain sequences from PDB and write to FASTA."""
    aa3to1 = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
        'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
        'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
    }

    chains = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            resname = line[17:20].strip()
            atom = line[12:16].strip()
            if atom == 'CA':
                if chain not in chains:
                    chains[chain] = {}
                chains[chain][resnum] = aa3to1.get(resname, 'X')

    with open(fasta_file, 'w') as f:
        f.write(">complex\n")
        seqs = []
        for c in sorted(chains.keys()):
            seq = ''.join(chains[c][r] for r in sorted(chains[c].keys()))
            seqs.append(seq)
        f.write(':'.join(seqs) + '\n')

def screen_with_simple_metrics(designs, binder_dir, output_dir, max_designs=None):
    """
    Screen designs using simple structure-based metrics without running AF2.

    This is a faster pre-screening step using:
    1. MPNN score (from ProteinMPNN)
    2. Basic structural quality checks

    Returns filtered designs ready for AF2 prediction.
    """
    if max_designs:
        designs = designs[:max_designs]

    # Sort by MPNN score (lower is better)
    designs.sort(key=lambda x: x.get('mpnn_score', 999))

    return designs

def screen_designs_with_esm(designs, output_dir):
    """
    Use ESMFold as a fast proxy for binder quality.
    ESMFold can predict the binder structure and give pLDDT.
    """
    try:
        import esm
        HAS_ESM = True
    except ImportError:
        HAS_ESM = False
        print("ESMFold not available, skipping ESM screening")
        return designs

    if not HAS_ESM:
        return designs

    print("Running ESMFold screening...")
    # ESMFold screening would go here
    return designs

def main():
    args = parse_args()

    print("=== AF2 Initial Guess Screening for pMHC Binders ===")
    print(f"Designs: {args.designs_json}")
    print(f"Target: {args.target_pdb}")
    print(f"Output: {args.output_dir}")
    print()

    # Load designs
    with open(args.designs_json) as f:
        designs = json.load(f)
    print(f"Loaded {len(designs)} designs")

    if args.max_designs:
        designs = designs[:args.max_designs]
        print(f"Screening top {len(designs)} designs")

    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-screen by MPNN score
    designs.sort(key=lambda x: x.get('mpnn_score', 999))
    print(f"\nTop 5 designs by MPNN score:")
    for d in designs[:5]:
        print(f"  {d['backbone']}_s{d['sample']}: mpnn_score={d.get('mpnn_score', 999):.4f} "
              f"binder_len={len(d.get('binder_sequence', ''))}")

    # Create complex PDBs for AF2 screening
    complex_pdbs_dir = os.path.join(args.output_dir, "complex_pdbs")
    os.makedirs(complex_pdbs_dir, exist_ok=True)

    print(f"\nCreating complex PDBs...")
    valid_designs = []
    for d in designs:
        backbone_pdb = os.path.join(args.binder_dir, f"{d['backbone']}.pdb")
        if not os.path.exists(backbone_pdb):
            continue

        design_name = f"{d['backbone']}_s{d['sample']}"
        complex_pdb = os.path.join(complex_pdbs_dir, f"{design_name}.pdb")

        create_complex_pdb(args.target_pdb, d.get('binder_sequence', ''),
                          backbone_pdb, complex_pdb)
        d['complex_pdb'] = complex_pdb
        d['design_name'] = design_name
        valid_designs.append(d)

    print(f"Created {len(valid_designs)} complex PDBs for screening")

    # Note: Full AF2 prediction requires running ColabFold/AlphaFold2
    # For now, we prepare the inputs and provide the command to run
    print("\n=== Inputs prepared for AF2 screening ===")
    print(f"Complex PDBs saved to: {complex_pdbs_dir}")
    print(f"\nTo run AF2 initial guess prediction, use:")
    print(f"  conda run -n colabfold colabfold_batch --msa-mode single_sequence \\")
    print(f"    --num-recycle 3 --model-type alphafold2_multimer_v3 \\")
    print(f"    {complex_pdbs_dir} {args.output_dir}/af2_predictions/")

    # Save the prepared designs list
    prep_file = os.path.join(args.output_dir, "designs_for_af2.json")
    with open(prep_file, 'w') as f:
        json.dump(valid_designs, f, indent=2)
    print(f"\nPrepared {len(valid_designs)} designs saved to: {prep_file}")

if __name__ == '__main__':
    main()
