#!/usr/bin/env python3
"""
AF2 initial guess prediction for pMHC binder designs.

Uses ColabFold Python API to:
1. Load AF2 model ONCE
2. For each design: run predict_structure with the designed complex PDB as initial_guess
   - Chain D backbone coordinates from RFdiffusion seed the AF2 recycling
   - AF2 refines from the designed geometry

Key metrics:
  pAE_interaction: mean PAE between binder (D) and peptide (C) — < 10 threshold
  binder_pLDDT:   mean pLDDT of chain D — > 0.7 threshold
  binder_RMSD:    Cα RMSD between AF2 prediction and RFdiffusion backbone

Usage:
  conda run -n colabfold python 04b_af2_predict.py \\
    --designs_json .../mpnn/all_designs.json \\
    --binder_dir .../rfdiffusion \\
    --output_dir .../af2_screening \\
    --max_designs 40 --num_recycles 3
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import subprocess
from pathlib import Path

DEFAULT_DESIGNS_JSON = "/workspace/pmhc_design/designs/mpnn/all_designs.json"
DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/af2_screening"
TARGET_PDB = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"

TARGET_SEQUENCES = {
    'A': "GSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQTDRVDLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHAAEQQRAYLEGRCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWE",
    'B': "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM",
    'C': "VVGADGVGK",
}

AA1TO3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}
AA3TO1 = {v: k for k, v in AA1TO3.items()}
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs_json", default=DEFAULT_DESIGNS_JSON)
    parser.add_argument("--binder_dir", default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_pdb", default=TARGET_PDB)
    parser.add_argument("--max_designs", type=int, default=50)
    parser.add_argument("--num_recycles", type=int, default=3)
    parser.add_argument("--pae_threshold", type=float, default=10.0)
    parser.add_argument("--plddt_threshold", type=float, default=0.7)
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


def create_design_pdb(target_pdb, backbone_pdb, binder_seq, output_pdb):
    """
    Create a complex PDB for AF2 initial guess.
    Chain D residue names updated from GLY to the designed sequence.
    Only backbone atoms (N, CA, C, O) retained for chain D.
    """
    target_lines = read_pdb_atoms(target_pdb, chain_filter={'A', 'B', 'C'})
    backbone_d = read_pdb_atoms(backbone_pdb, chain_filter={'D'})

    residue_nums = []
    seen = set()
    for line in backbone_d:
        rnum = int(line[22:26])
        if rnum not in seen:
            residue_nums.append(rnum)
            seen.add(rnum)

    n = min(len(residue_nums), len(binder_seq))
    rnum_to_resname = {rnum: AA1TO3.get(aa, 'GLY')
                       for rnum, aa in zip(residue_nums[:n], binder_seq[:n])}

    updated_d = []
    for line in backbone_d:
        rnum = int(line[22:26])
        if rnum not in rnum_to_resname:
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        new_resname = rnum_to_resname[rnum]
        new_line = line[:17] + f"{new_resname:<3}" + line[20:]
        updated_d.append(new_line)

    with open(output_pdb, 'w') as f:
        f.writelines(target_lines)
        f.write('TER\n')
        f.writelines(updated_d)
        f.write('TER\nEND\n')


def compute_pae_interaction(pae_matrix, chain_lengths):
    chains = ['A', 'B', 'C', 'D']
    starts = {}
    pos = 0
    for c in chains:
        starts[c] = pos
        pos += chain_lengths.get(c, 0)

    bs, be = starts['D'], starts['D'] + chain_lengths['D']
    ts, te = starts['C'], starts['C'] + chain_lengths['C']

    pae = np.array(pae_matrix)
    return float((pae[bs:be, ts:te].mean() + pae[ts:te, bs:be].mean()) / 2)


def compute_binder_plddt(plddt, chain_lengths):
    chains = ['A', 'B', 'C', 'D']
    pos = 0
    for c in chains:
        if c == 'D':
            return float(np.array(plddt)[pos:pos + chain_lengths['D']].mean())
        pos += chain_lengths.get(c, 0)
    return 0.0


def compute_ca_rmsd(pred_pdb, design_pdb, chain='D'):
    def get_ca(pdb, c):
        coords = {}
        with open(pdb) as f:
            for line in f:
                if line.startswith('ATOM') and line[21] == c and line[12:16].strip() == 'CA':
                    rnum = int(line[22:26])
                    coords[rnum] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        return np.array([coords[r] for r in sorted(coords)])

    pred = get_ca(pred_pdb, chain)
    design = get_ca(design_pdb, chain)
    if len(pred) == 0 or len(pred) != len(design):
        return 999.0

    pc = pred - pred.mean(0)
    dc = design - design.mean(0)
    H = dc.T @ pc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((( R @ pc.T).T - dc)**2).sum(1).mean()))


def parse_colabfold_result(pred_dir, design_name, backbone_pdb, chain_lengths):
    score_files = sorted(glob.glob(
        os.path.join(pred_dir, f"{design_name}_scores_rank_001_*.json")))
    if not score_files:
        return None

    with open(score_files[0]) as f:
        scores = json.load(f)

    pae_inter = compute_pae_interaction(scores['pae'], chain_lengths)
    binder_plddt = compute_binder_plddt(scores['plddt'], chain_lengths)

    pred_pdbs = sorted(glob.glob(
        os.path.join(pred_dir, f"{design_name}*rank_001*.pdb")))
    binder_rmsd = 999.0
    if pred_pdbs and backbone_pdb and os.path.exists(backbone_pdb):
        try:
            binder_rmsd = compute_ca_rmsd(pred_pdbs[0], backbone_pdb, 'D')
        except Exception as e:
            print(f"  RMSD error: {e}")

    return {
        'pae_interaction': pae_inter,
        'binder_plddt': binder_plddt,
        'binder_rmsd': binder_rmsd,
        'ptm': scores.get('ptm', 0),
        'iptm': scores.get('iptm', 0),
    }


def run_colabfold_with_initial_guess(complex_pdb, pred_dir, design_name, num_recycles=3):
    """
    Run colabfold_batch for a single design using the complex PDB as initial guess.
    The PDB provides both the sequence (from ATOM records) and initial coordinates.
    """
    design_output = os.path.join(pred_dir, design_name)
    os.makedirs(design_output, exist_ok=True)

    # Check if already done
    done_marker = os.path.join(design_output, f"{design_name}.done.txt")
    if os.path.exists(done_marker):
        return True

    cmd = [
        "/venv/colabfold/bin/colabfold_batch",
        "--initial-guess",          # Use input PDB as initial guess
        "--msa-mode", "single_sequence",
        "--num-recycle", str(num_recycles),
        "--model-type", "alphafold2_multimer_v3",
        "--num-models", "1",
        complex_pdb,                # Input: complex PDB (sequence + initial coords)
        design_output,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Create done marker
        with open(done_marker, 'w') as f:
            f.write("done\n")
        return True
    else:
        print(f"  ColabFold failed for {design_name}:")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        return False


def main():
    args = parse_args()

    print("=== AF2 Initial Guess Screening for pMHC Binders ===")
    print("Protocol: per-design initial-guess (RFdiffusion backbone seeds AF2)")

    with open(args.designs_json) as f:
        designs = json.load(f)

    designs.sort(key=lambda x: x.get('mpnn_score', 999))
    if args.max_designs:
        designs = designs[:args.max_designs]
    print(f"Screening {len(designs)} designs (top by MPNN score)")

    os.makedirs(args.output_dir, exist_ok=True)
    complex_pdb_dir = os.path.join(args.output_dir, "complex_pdbs")
    pred_dir = os.path.join(args.output_dir, "colabfold_predictions")
    os.makedirs(complex_pdb_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # Prepare per-design complex PDBs
    print("\nPreparing design complex PDBs...")
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
        complex_pdb = os.path.join(complex_pdb_dir, f"{design_name}.pdb")

        try:
            create_design_pdb(args.target_pdb, backbone_pdb, binder_seq, complex_pdb)
        except Exception as e:
            print(f"  Failed {design_name}: {e}")
            continue

        chain_lengths = {
            'A': len(TARGET_SEQUENCES['A']),
            'B': len(TARGET_SEQUENCES['B']),
            'C': len(TARGET_SEQUENCES['C']),
            'D': len(binder_seq),
        }
        valid_designs.append({
            **d,
            'design_name': design_name,
            'chain_lengths': chain_lengths,
            'backbone_pdb': backbone_pdb,
            'complex_pdb': complex_pdb,
        })

    print(f"Prepared {len(valid_designs)} complex PDBs")

    # Run ColabFold per design with initial guess
    print(f"\nRunning ColabFold predictions (individual --initial-guess per design)...")
    n_done = 0
    n_failed = 0

    for i, d in enumerate(valid_designs):
        design_name = d['design_name']
        design_output = os.path.join(pred_dir, design_name)

        # Check if already done
        done_marker = os.path.join(design_output, f"{design_name}.done.txt")
        if os.path.exists(done_marker):
            print(f"  [{i+1}/{len(valid_designs)}] {design_name}: already done, skipping")
            n_done += 1
            continue

        print(f"  [{i+1}/{len(valid_designs)}] {design_name} "
              f"(len={d['chain_lengths']['D']}, mpnn={d.get('mpnn_score',999):.3f})...",
              end='', flush=True)

        success = run_colabfold_with_initial_guess(
            d['complex_pdb'], pred_dir, design_name, args.num_recycles)

        if success:
            n_done += 1
            print(" done")
        else:
            n_failed += 1
            print(" FAILED")

    print(f"\nCompleted: {n_done} done, {n_failed} failed out of {len(valid_designs)}")

    # Parse results
    print("\nParsing results...")
    results = []
    for d in valid_designs:
        design_name = d['design_name']
        design_output = os.path.join(pred_dir, design_name)

        metrics = parse_colabfold_result(
            design_output, design_name, d['backbone_pdb'], d['chain_lengths'])

        if not metrics:
            continue

        result = {**d, **metrics}
        results.append(result)

    print(f"Parsed {len(results)} results")

    # Filter
    passing = [r for r in results
               if r.get('pae_interaction', 999) < args.pae_threshold
               and r.get('binder_plddt', 0) > args.plddt_threshold]

    print(f"\n{len(passing)}/{len(results)} pass: "
          f"pAE < {args.pae_threshold} AND pLDDT > {args.plddt_threshold}")

    passing.sort(key=lambda x: x.get('pae_interaction', 999))

    if results:
        results.sort(key=lambda x: x.get('pae_interaction', 999))
        print(f"\nTop 20 by pAE_interaction:")
        print(f"{'Design':<25} {'pAE_int':>8} {'pLDDT':>6} {'RMSD':>6} {'ipTM':>6} {'MPNN':>6}")
        print('-' * 60)
        for r in results[:20]:
            flag = " *" if r in passing else ""
            print(f"{r['design_name']:<25} "
                  f"{r.get('pae_interaction',999):>8.2f} "
                  f"{r.get('binder_plddt',0):>6.2f} "
                  f"{r.get('binder_rmsd',999):>6.2f} "
                  f"{r.get('iptm',0):>6.3f} "
                  f"{r.get('mpnn_score',999):>6.3f}{flag}")

    # Save results
    results_file = os.path.join(args.output_dir, "screening_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    passing_file = os.path.join(args.output_dir, "passing_designs.json")
    with open(passing_file, 'w') as f:
        json.dump(passing, f, indent=2, default=float)

    print(f"\nResults: {results_file}")
    print(f"Passing: {passing_file}")

    if passing:
        fasta_out = os.path.join(args.output_dir, "top_binders.fa")
        with open(fasta_out, 'w') as f:
            for r in passing[:50]:
                f.write(f">{r['design_name']} pae={r['pae_interaction']:.2f} "
                        f"plddt={r['binder_plddt']:.2f} rmsd={r['binder_rmsd']:.2f}\n"
                        f"{r.get('binder_sequence','')}\n")
        print(f"Top binders FASTA: {fasta_out}")


if __name__ == '__main__':
    main()
