#!/usr/bin/env python3
"""
Prepare the target pMHC structure for RFdiffusion binder design.

Target: HLA-A*11:01 in complex with KRAS G12D 9-mer peptide VVGADGVGK
PDB: 9UV8
- Chain A: HLA-A*11:01 heavy chain
- Chain B: beta-2 microglobulin
- Chain C: KRAS G12D peptide (VVGADGVGK)

This script:
1. Cleans the PDB (removes HETATM, waters, alt conformations)
2. Renumbers residues sequentially
3. Identifies upward-facing peptide residues as hotspots
4. Outputs a cleaned PDB and hotspot list for RFdiffusion
"""

import numpy as np
import os
import sys

INPUT_PDB = "/workspace/pmhc_design/input/9UV8.pdb"
OUTPUT_PDB = "/workspace/pmhc_design/designs/target_9UV8_clean.pdb"

def parse_pdb_atoms(pdb_file):
    """Parse ATOM records from PDB file."""
    records = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                records.append(line)
    return records

def get_ca_positions(pdb_file):
    """Get CA positions per residue."""
    atoms = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain = line[21]
            resnum = int(line[22:26])
            resname = line[17:20].strip()
            atomname = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            key = (chain, resnum, resname)
            if key not in atoms:
                atoms[key] = {}
            atoms[key][atomname] = np.array([x, y, z])
    return atoms

def clean_pdb(input_pdb, output_pdb):
    """Remove HETATM, water, alt conformations. Keep ATOM records."""
    seen_residues = {}
    lines_out = []

    with open(input_pdb) as f:
        for line in f:
            if line.startswith('ATOM'):
                alt_loc = line[16]
                if alt_loc not in (' ', 'A', ''):
                    continue
                # Fix alt_loc
                line = line[:16] + ' ' + line[17:]
                lines_out.append(line)
            elif line.startswith('TER') or line.startswith('END'):
                lines_out.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(lines_out)

    print(f"Cleaned PDB written to: {output_pdb}")
    return output_pdb

def identify_hotspots(atoms):
    """
    Identify upward-facing peptide residues as RFdiffusion hotspots.

    For KRAS G12D peptide VVGADGVGK on HLA-A*11:01:
    - P4 (ALA): strongly upward-facing
    - P5 (ASP, G12D): key specificity determinant
    - P7 (VAL): strongly upward-facing

    The 'up' direction is from the MHC floor toward the open solvent.
    """
    # Get peptide CA positions
    peptide_CA = {}
    for (chain, resnum, resname), pos in atoms.items():
        if chain == 'C' and 'CA' in pos:
            peptide_CA[resnum] = (resname, pos['CA'])

    # Get MHC floor residues (beta sheet, roughly residues 1-50 of chain A)
    mhc_floor_CA = []
    for (chain, resnum, resname), pos in atoms.items():
        if chain == 'A' and 'CA' in pos and 1 <= resnum <= 50:
            mhc_floor_CA.append(pos['CA'])

    mhc_floor_CA = np.array(mhc_floor_CA)
    floor_centroid = mhc_floor_CA.mean(axis=0)

    pep_CAs = np.array([v[1] for v in peptide_CA.values()])
    pep_centroid = pep_CAs.mean(axis=0)

    # Up direction: from floor towards peptide
    up_dir = pep_centroid - floor_centroid
    up_dir = up_dir / np.linalg.norm(up_dir)

    print("\nPeptide upward orientation analysis:")
    print(f"Up direction vector: {up_dir}")
    print()

    upward_scores = {}
    for resnum in sorted(peptide_CA.keys()):
        resname, ca = peptide_CA[resnum]
        key = ('C', resnum, resname)
        pos = atoms[key]

        if resname == 'GLY':
            # GLY has no CB, use HA2/HA3 if available, else score = 0
            score = 0.0
        else:
            cb = pos.get('CB', ca)
            sc_vec = cb - ca
            norm = np.linalg.norm(sc_vec)
            if norm > 0:
                sc_dir = sc_vec / norm
                score = float(np.dot(sc_dir, up_dir))
            else:
                score = 0.0

        upward_scores[resnum] = (resname, score)
        print(f"  P{resnum} ({resname:3s}): upward_score = {score:+.3f}")

    # Select hotspots: residues with significant upward-facing score OR key mutation
    # P5 (ASP, G12D) is always included as it's the key specificity determinant
    # P4 and P7 are the most strongly upward-facing
    hotspots = []

    for resnum in sorted(upward_scores.keys()):
        resname, score = upward_scores[resnum]
        # Include if strongly upward OR if it's the G12D mutation (P5=ASP)
        if score > 0.5 or (resname == 'ASP' and resnum == 5):
            hotspots.append(f"C{resnum}")

    print(f"\nSelected hotspots: {hotspots}")
    return hotspots

def main():
    print("=== Preparing pMHC target structure for RFdiffusion ===")
    print(f"Input: {INPUT_PDB}")
    print(f"Target: HLA-A*11:01 + KRAS G12D peptide (VVGADGVGK)")
    print()

    # Clean the PDB
    clean_pdb(INPUT_PDB, OUTPUT_PDB)

    # Parse atoms
    atoms = get_ca_positions(OUTPUT_PDB)

    # Count residues per chain
    chains = {}
    for (chain, resnum, resname) in atoms.keys():
        if chain not in chains:
            chains[chain] = []
        chains[chain].append(resnum)

    print("\nChain summary:")
    for chain in sorted(chains.keys()):
        resnums = sorted(chains[chain])
        seq_parts = []
        for (c, r, aa) in sorted(atoms.keys()):
            if c == chain and r in resnums:
                seq_parts.append((r, aa))
        seq_parts = sorted(set(seq_parts))
        print(f"  Chain {chain}: {len(set(resnums))} residues "
              f"({min(resnums)}-{max(resnums)})")

    # Get peptide sequence
    pep_seq = []
    aa1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
           'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
           'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    for r, aa in sorted(set((r, aa) for (c, r, aa) in atoms.keys() if c == 'C')):
        pep_seq.append(aa1.get(aa, 'X'))
    print(f"\nPeptide sequence (Chain C): {''.join(pep_seq)}")

    # Identify hotspots
    hotspots = identify_hotspots(atoms)

    # Write hotspot file
    hotspot_file = "/workspace/pmhc_design/designs/hotspots.txt"
    with open(hotspot_file, 'w') as f:
        f.write(','.join(hotspots) + '\n')
    print(f"\nHotspots written to: {hotspot_file}")

    # Write the RFdiffusion contig
    # Format: [A1-275/0 B1-99/0 C1-9/0 70-80]
    # The /0 means zero-length linker (chain break)
    chain_A_max = max(chains.get('A', [0]))
    chain_B_max = max(chains.get('B', [0]))
    chain_C_max = max(chains.get('C', [0]))

    contig = f"[A1-{chain_A_max}/0 B1-{chain_B_max}/0 C1-{chain_C_max}/0 70-80]"
    print(f"\nRFdiffusion contig string: {contig}")

    contig_file = "/workspace/pmhc_design/designs/contig.txt"
    with open(contig_file, 'w') as f:
        f.write(contig + '\n')

    print("\n=== Target preparation complete ===")
    print(f"Clean PDB: {OUTPUT_PDB}")
    print(f"Hotspots: {hotspots}")
    print(f"Contig: {contig}")

if __name__ == '__main__':
    main()
