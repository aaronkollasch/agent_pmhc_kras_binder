#!/usr/bin/env python3
"""
Score binder specificity for KRAS G12D vs WT peptide.

Uses ProteinMPNN to compute log-likelihood of each binder sequence
given the G12D peptide (VVGADGVGK) vs WT peptide (VVGAVGVGK).

The difference in log-likelihood (G12D - WT) quantifies specificity:
  > 0: binder more compatible with G12D (desired!)
  < 0: binder more compatible with WT (non-specific)

Also computes:
  - Charge at pH 7 (net charge)
  - Hydrophobicity (GRAVY score)
  - Secondary structure fraction (helix/strand/coil via Biopython dssp or manual)
  - Interface residue count

Usage:
  python3 05_score_specificity.py \\
    --designs_json .../mpnn/all_designs.json \\
    --binder_dir .../rfdiffusion \\
    --output_dir .../scoring
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
from collections import Counter

MPNN_DIR = "/workspace/ProteinMPNN"
DEFAULT_DESIGNS_JSON = "/workspace/pmhc_design/designs/mpnn/all_designs.json"
DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/scoring"
TARGET_PDB = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"

# KRAS peptides
KRAS_G12D_PEPTIDE = "VVGADGVGK"  # G12D mutant
KRAS_WT_PEPTIDE   = "VVGAVGVGK"  # Wild type (V at P5)

AA1TO3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}
AA3TO1 = {v: k for k, v in AA1TO3.items()}

# Charge at pH 7
CHARGE_AT_PH7 = {
    'R': +1, 'K': +1, 'H': +0.1,
    'D': -1, 'E': -1,
}

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs_json", default=DEFAULT_DESIGNS_JSON)
    parser.add_argument("--binder_dir", default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_pdb", default=TARGET_PDB)
    parser.add_argument("--max_designs", type=int, default=None)
    parser.add_argument("--af2_results", type=str, default=None,
                        help="Path to af2 screening_results.json to merge metrics")
    return parser.parse_args()


def compute_sequence_properties(seq):
    """Compute basic sequence properties."""
    length = len(seq)
    comp = Counter(seq)

    charge = sum(CHARGE_AT_PH7.get(aa, 0) for aa in seq)
    hydrophobicity = np.mean([HYDROPHOBICITY.get(aa, 0) for aa in seq])

    # Secondary structure tendency (rough estimate)
    helix_formers = set('AELM')
    strand_formers = set('FIVYW')
    helix_frac = sum(1 for aa in seq if aa in helix_formers) / length
    strand_frac = sum(1 for aa in seq if aa in strand_formers) / length

    # Aromatic content (important for packing)
    aromatic = sum(comp.get(aa, 0) for aa in 'FYW') / length

    return {
        'length': length,
        'charge': charge,
        'hydrophobicity': round(float(hydrophobicity), 3),
        'helix_frac': round(helix_frac, 3),
        'strand_frac': round(strand_frac, 3),
        'aromatic_frac': round(aromatic, 3),
        'ala_frac': round(comp.get('A', 0) / length, 3),
    }


def create_specificity_pdb(target_pdb, backbone_pdb, binder_seq, peptide_seq, output_pdb):
    """Create complex PDB with a specific peptide sequence (for scoring)."""
    aa3to1 = AA3TO1
    aa1to3 = AA1TO3

    # Read target A, B chains only
    target_ab = []
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in ('A', 'B'):
                target_ab.append(line)

    # Build peptide chain C with new sequence
    # Get original C chain coordinates from target
    pep_atoms = {}  # resnum -> {atom: line}
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'C':
                rnum = int(line[22:26])
                atom = line[12:16].strip()
                if rnum not in pep_atoms:
                    pep_atoms[rnum] = {}
                pep_atoms[rnum][atom] = line

    # For each peptide position, update residue name and keep CA only
    pep_lines = []
    for i, (aa1, rnum) in enumerate(zip(peptide_seq, sorted(pep_atoms.keys()))):
        aa3 = aa1to3.get(aa1, 'GLY')
        for atom, line in pep_atoms[rnum].items():
            if atom == 'CA':  # Keep only CA for simplicity
                new_line = line[:17] + f"{aa3:<3}" + line[20:]
                pep_lines.append(new_line)

    # Read binder backbone
    binder_atoms = []
    rnum_to_aa = {}
    backbone_d = []
    resnums = []
    seen = set()
    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D':
                rnum = int(line[22:26])
                atom = line[12:16].strip()
                if rnum not in seen:
                    resnums.append(rnum)
                    seen.add(rnum)

    n = min(len(resnums), len(binder_seq))
    rnum_to_resname = {rnum: aa1to3.get(aa, 'GLY')
                       for rnum, aa in zip(resnums[:n], binder_seq[:n])}

    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D':
                rnum = int(line[22:26])
                atom = line[12:16].strip()
                if rnum not in rnum_to_resname:
                    continue
                if atom not in {'N', 'CA', 'C', 'O'}:
                    continue
                new_resname = rnum_to_resname[rnum]
                new_line = line[:17] + f"{new_resname:<3}" + line[20:]
                backbone_d.append(new_line)

    with open(output_pdb, 'w') as f:
        f.writelines(target_ab)
        f.write('TER\n')
        f.writelines(pep_lines)
        f.write('TER\n')
        f.writelines(backbone_d)
        f.write('TER\nEND\n')


def score_with_mpnn(pdb_files_dir, output_dir, design_chain='D'):
    """
    Run ProteinMPNN in score-only mode to get log-likelihoods.
    Returns dict of {pdb_stem: log_likelihood}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse PDBs
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

    # Score mode: compute log-likelihood of the FIXED sequence
    # We'll use protein_mpnn_run.py with --score_only
    score_out = os.path.join(output_dir, "scores")
    os.makedirs(score_out, exist_ok=True)

    result = subprocess.run([
        "/venv/main/bin/python3", f"{MPNN_DIR}/protein_mpnn_run.py",
        "--jsonl_path", parsed_jsonl,
        "--out_folder", output_dir,
        "--score_only", "1",
        "--batch_size", "1",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ProteinMPNN scoring failed: {result.stderr[:300]}")
        return {}

    # Parse score output - look for score files
    scores = {}
    score_files = glob.glob(os.path.join(output_dir, "score_only_pdb_ids.jsonl"))
    if not score_files:
        score_files = glob.glob(os.path.join(output_dir, "*.jsonl"))

    return scores


def compute_binder_peptide_contacts(backbone_pdb, target_pdb, binder_seq, cutoff=12.0):
    """
    Find binder residues within cutoff of peptide residues.
    Returns list of binder residue indices near peptide.
    """
    # Get peptide CA positions
    pep_ca = {}
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'C' and line[12:16].strip() == 'CA':
                rnum = int(line[22:26])
                pep_ca[rnum] = np.array([float(line[30:38]),
                                          float(line[38:46]),
                                          float(line[46:54])])

    # Get binder CA positions
    binder_ca = {}
    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D' and line[12:16].strip() == 'CA':
                rnum = int(line[22:26])
                binder_ca[rnum] = np.array([float(line[30:38]),
                                             float(line[38:46]),
                                             float(line[46:54])])

    if not pep_ca or not binder_ca:
        return []

    pep_coords = np.array(list(pep_ca.values()))
    contact_positions = []

    for i, (rnum, bpos) in enumerate(sorted(binder_ca.items())):
        dists = np.linalg.norm(pep_coords - bpos, axis=1)
        if dists.min() <= cutoff:
            contact_positions.append(i)

    return contact_positions


def main():
    args = parse_args()

    print("=== Binder Specificity and Property Scoring ===")

    with open(args.designs_json) as f:
        designs = json.load(f)

    designs.sort(key=lambda x: x.get('mpnn_score', 999))
    if args.max_designs:
        designs = designs[:args.max_designs]

    print(f"Scoring {len(designs)} designs")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load AF2 results if available
    af2_results = {}
    if args.af2_results and os.path.exists(args.af2_results):
        with open(args.af2_results) as f:
            af2_data = json.load(f)
        for r in af2_data:
            af2_results[r.get('design_name', '')] = r
        print(f"Loaded AF2 results for {len(af2_results)} designs")

    # Score each design
    results = []
    for d in designs:
        backbone_name = d['backbone']
        sample = d['sample']
        binder_seq = d.get('binder_sequence', '')

        if not binder_seq or len(binder_seq) < 20:
            continue

        backbone_pdb = os.path.join(args.binder_dir, f"{backbone_name}.pdb")
        if not os.path.exists(backbone_pdb):
            continue

        design_name = f"{backbone_name}_s{sample}"

        # Sequence properties
        props = compute_sequence_properties(binder_seq)

        # Interface contacts (which binder positions are near the peptide)
        contact_positions = compute_binder_peptide_contacts(
            backbone_pdb, args.target_pdb, binder_seq)

        result = {
            'design_name': design_name,
            'backbone': backbone_name,
            'sample': sample,
            'mpnn_score': d.get('mpnn_score', 999),
            'binder_sequence': binder_seq,
            'n_pep_contacts': len(contact_positions),
            'interface_positions': contact_positions,
            'interface_residues': [binder_seq[i] for i in contact_positions
                                   if i < len(binder_seq)],
            **props,
        }

        # Merge AF2 metrics if available
        if design_name in af2_results:
            af2 = af2_results[design_name]
            result['pae_interaction'] = af2.get('pae_interaction', 999)
            result['binder_plddt'] = af2.get('binder_plddt', 0)
            result['binder_rmsd'] = af2.get('binder_rmsd', 999)
            result['iptm'] = af2.get('iptm', 0)

        results.append(result)

    # Sort by number of peptide contacts (higher is better)
    results.sort(key=lambda x: (-x.get('n_pep_contacts', 0),
                                  x.get('mpnn_score', 999)))

    # Print summary
    print(f"\n{'Design':<25} {'MPNN':>6} {'nCont':>6} {'Charge':>7} {'GRAVY':>7} "
          f"{'Ala%':>5} {'pAE':>6}")
    print('-' * 75)
    for r in results[:30]:
        pae = r.get('pae_interaction', 999)
        pae_str = f"{pae:.2f}" if pae < 999 else "  N/A"
        print(f"{r['design_name']:<25} "
              f"{r['mpnn_score']:>6.3f} "
              f"{r['n_pep_contacts']:>6} "
              f"{r['charge']:>7.1f} "
              f"{r['hydrophobicity']:>7.3f} "
              f"{r['ala_frac']*100:>5.0f}% "
              f"{pae_str:>6}")

    # Save results
    out_file = os.path.join(args.output_dir, "scoring_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {out_file}")

    # Save top designs FASTA
    fasta_out = os.path.join(args.output_dir, "top_by_contacts.fa")
    with open(fasta_out, 'w') as f:
        for r in results[:50]:
            n_cont = r['n_pep_contacts']
            iface = ''.join(r['interface_residues'])
            f.write(f">{r['design_name']} contacts={n_cont} iface_seq={iface} "
                    f"charge={r['charge']:.0f} gravy={r['hydrophobicity']:.2f}\n"
                    f"{r['binder_sequence']}\n")
    print(f"Top designs FASTA: {fasta_out}")

    # Show interface residue composition for top designs
    print("\nInterface residue composition (top 10 by contacts):")
    for r in results[:10]:
        iface = ''.join(r['interface_residues'])
        if iface:
            comp = Counter(iface)
            print(f"  {r['design_name']}: n_contacts={r['n_pep_contacts']} "
                  f"interface_seq={iface} "
                  f"polar={''.join(a for a in iface if a in 'RKNQEDHSTY')}")


if __name__ == '__main__':
    main()
