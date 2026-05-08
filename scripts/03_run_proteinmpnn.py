#!/usr/bin/env python3
"""
Run ProteinMPNN on RFdiffusion output backbones to design sequences.

For each binder backbone:
- Chain A (HLA heavy chain): fixed
- Chain B (beta-2 microglobulin): fixed
- Chain C (KRAS G12D peptide VVGADGVGK): fixed
- Chain D (binder): designed

Output: FASTA files with designed binder sequences.
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import tempfile

MPNN_DIR = "/workspace/ProteinMPNN"
DEFAULT_BINDER_DIR = "/workspace/pmhc_design/designs/rfdiffusion"
DEFAULT_OUTPUT_DIR = "/workspace/pmhc_design/designs/mpnn"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seqs", type=int, default=8, help="Sequences per backbone")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--binder_dir", type=str, default=DEFAULT_BINDER_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", type=str, default="binder_*.pdb",
                        help="Glob pattern for binder PDB files")
    parser.add_argument("--design_chain", type=str, default="D",
                        help="Chain to design (default: D)")
    return parser.parse_args()

def run_mpnn_on_pdbs(pdb_files, output_dir, num_seqs=8, temperature=0.1, design_chain="D"):
    """Run ProteinMPNN on a list of PDB files."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/seqs", exist_ok=True)

    # Create temp dir with PDB links
    tmpdir = f"{output_dir}/tmp_pdbs"
    os.makedirs(tmpdir, exist_ok=True)
    for pdb in pdb_files:
        dst = os.path.join(tmpdir, os.path.basename(pdb))
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(pdb), dst)

    # Parse PDBs to JSONL
    parsed_jsonl = f"{output_dir}/parsed.jsonl"
    subprocess.run([
        "/venv/main/bin/python3",
        f"{MPNN_DIR}/helper_scripts/parse_multiple_chains.py",
        "--input_path", tmpdir,
        "--output_path", parsed_jsonl,
    ], check=True, capture_output=True)

    # Assign which chain to design (D), fix the rest (A, B, C)
    assigned_jsonl = f"{output_dir}/assigned.jsonl"
    subprocess.run([
        "/venv/main/bin/python3",
        f"{MPNN_DIR}/helper_scripts/assign_fixed_chains.py",
        "--input_path", parsed_jsonl,
        "--output_path", assigned_jsonl,
        "--chain_list", design_chain,  # chains to DESIGN
    ], check=True, capture_output=True)

    # Run ProteinMPNN
    cmd = [
        "/venv/main/bin/python3", f"{MPNN_DIR}/protein_mpnn_run.py",
        "--jsonl_path", parsed_jsonl,
        "--chain_id_jsonl", assigned_jsonl,
        "--out_folder", output_dir,
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temperature),
        "--batch_size", "1",
        "--save_score", "1",
    ]

    print(f"Running ProteinMPNN on {len(pdb_files)} backbones, {num_seqs} seqs each...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

def parse_mpnn_output(output_dir, binder_dir):
    """Parse ProteinMPNN output FASTA files and extract binder sequences."""
    designs = []

    for fa_file in sorted(glob.glob(f"{output_dir}/seqs/*.fa")):
        backbone_name = os.path.splitext(os.path.basename(fa_file))[0]
        pdb_file = os.path.join(binder_dir, f"{backbone_name}.pdb")

        # Parse FASTA
        entries = []
        with open(fa_file) as f:
            current_header = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    current_header = line[1:]
                    entries.append({'header': current_header, 'sequence': ''})
                elif entries:
                    entries[-1]['sequence'] += line

        if not entries:
            continue

        # First entry is the original poly-G backbone (reference)
        # Subsequent entries are designed sequences
        for i, entry in enumerate(entries):
            full_seq = entry['sequence']

            # For multi-chain PDBs, ProteinMPNN outputs all chains separated by /
            # We want only chain D (the binder)
            parts = full_seq.split('/')

            # Parse header info
            header_parts = entry['header'].split(', ')
            header_dict = {}
            for p in header_parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    header_dict[k.strip()] = v.strip()

            design = {
                'backbone': backbone_name,
                'sample': i,
                'header': entry['header'],
                'full_sequence': full_seq,
                'mpnn_score': float(header_dict.get('score', 999)),
                'global_score': float(header_dict.get('global_score', 999)),
                'fixed_chains': header_dict.get('fixed_chains', ''),
                'designed_chains': header_dict.get('designed_chains', ''),
            }

            # Extract binder-only sequence from the full sequence
            # Determine which part is the binder by looking at chain structure
            if os.path.exists(pdb_file):
                chain_info = get_chain_lengths(pdb_file)
                binder_seq = extract_binder_seq(full_seq, chain_info)
                design['binder_sequence'] = binder_seq
            else:
                # If PDB not found, take the last part (chain D is last)
                design['binder_sequence'] = parts[-1] if parts else full_seq

            if i > 0:  # Skip the reference (poly-G) entry
                designs.append(design)

    return designs

def get_chain_lengths(pdb_file):
    """Get chain lengths from PDB file."""
    chains = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            if chain not in chains:
                chains[chain] = set()
            chains[chain].add(resnum)

    # Return sorted chain info
    result = {}
    for chain in sorted(chains.keys()):
        result[chain] = len(chains[chain])
    return result

def extract_binder_seq(full_seq, chain_info):
    """Extract the binder (chain D) sequence from the concatenated full sequence."""
    # ProteinMPNN separates chains by /
    parts = full_seq.split('/')
    chains = sorted(chain_info.keys())

    if 'D' in chains and len(parts) >= len(chains):
        d_idx = chains.index('D')
        if d_idx < len(parts):
            return parts[d_idx]

    # Fallback: last part
    return parts[-1] if parts else full_seq

def save_designs_fasta(designs, output_file):
    """Save all designs as a FASTA file."""
    with open(output_file, 'w') as f:
        for d in designs:
            name = f"{d['backbone']}_s{d['sample']}"
            score = d.get('mpnn_score', 999)
            seq = d.get('binder_sequence', '')
            if seq and seq != 'G' * len(seq):  # Skip poly-G references
                f.write(f">{name} mpnn_score={score:.4f}\n{seq}\n")

def main():
    args = parse_args()

    print("=== Running ProteinMPNN for pMHC binder sequence design ===")
    print(f"Input dir: {args.binder_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Output dir: {args.output_dir}")
    print(f"Sequences per backbone: {args.num_seqs}")
    print(f"Temperature: {args.temperature}")
    print(f"Design chain: {args.design_chain}")
    print()

    pdb_files = sorted(glob.glob(os.path.join(args.binder_dir, args.pattern)))
    print(f"Found {len(pdb_files)} binder PDB files")

    if not pdb_files:
        print("No PDB files found!")
        sys.exit(1)

    # Run ProteinMPNN
    os.makedirs(args.output_dir, exist_ok=True)
    success = run_mpnn_on_pdbs(
        pdb_files,
        args.output_dir,
        num_seqs=args.num_seqs,
        temperature=args.temperature,
        design_chain=args.design_chain,
    )

    if not success:
        print("ProteinMPNN failed!")
        sys.exit(1)

    # Parse output
    designs = parse_mpnn_output(args.output_dir, args.binder_dir)
    print(f"\nParsed {len(designs)} designs")

    # Save
    designs_json = f"{args.output_dir}/all_designs.json"
    with open(designs_json, 'w') as f:
        json.dump(designs, f, indent=2)
    print(f"Saved designs to {designs_json}")

    designs_fasta = f"{args.output_dir}/all_binder_sequences.fa"
    save_designs_fasta(designs, designs_fasta)
    print(f"Saved binder FASTA to {designs_fasta}")

    # Print top designs by MPNN score (lower is better)
    designs.sort(key=lambda x: x.get('mpnn_score', 999))
    print(f"\nTop 10 designs by MPNN score:")
    for d in designs[:10]:
        print(f"  {d['backbone']}_s{d['sample']}: score={d.get('mpnn_score', 999):.4f} "
              f"binder_len={len(d.get('binder_sequence', ''))}")

if __name__ == '__main__':
    main()
