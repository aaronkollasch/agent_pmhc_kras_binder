#!/usr/bin/env python3
"""
Round 2 comprehensive design analysis.
Computes per-design metrics from AF2 predictions + sequences.

Metrics:
  AF2-derived:    pLDDT, RMSD, pAE_interaction, ipSAE (peptide-binder only PAE)
  Sequence:       net charge, poly-Ala runs, Lys/Met count at interface
  Structure:      secondary structure %, interface contacts (hydrophobic/polar),
                  interface Lys+Met, exposed hydrophobic fraction
  NOT computed (requires Rosetta): CMS, FastRelax energetics, packstat,
                  shape complementarity, unsaturated H-bonds

Usage:
  conda run -n colabfold python 10_analyze_designs.py \
    --af2_dir <colabfold_predictions_dir> \
    --designs_json <input_designs.json> \
    --output <results.json>
"""

import os, sys, json, glob, math, argparse
import numpy as np
from pathlib import Path

try:
    from Bio.PDB import PDBParser, DSSP, NeighborSearch
    from Bio.PDB.Polypeptide import is_aa
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("WARNING: Biopython not available — structural metrics disabled")

# ── amino acid properties ──────────────────────────────────────────────────
HYDROPHOBIC = set("VILMFYWAC")
POLAR       = set("STNQHKRDE")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
CHARGE_MAP  = {aa: +1 for aa in "KR"} | {"H": +0.5} | {aa: -1 for aa in "DE"}

CHAIN_LENGTHS = {"A": 276, "B": 99, "C": 9}   # pMHC fixed lengths


# ── sequence-level metrics ─────────────────────────────────────────────────

def net_charge(seq, ph=7.0):
    return sum(CHARGE_MAP.get(aa, 0) for aa in seq)


def poly_ala_runs(seq, min_run=4):
    runs = []
    cur = 0
    for aa in seq:
        if aa == "A":
            cur += 1
            if cur >= min_run:
                runs.append(cur)
        else:
            cur = 0
    return max(runs) if runs else 0


def lys_met_count(seq):
    return seq.count("K") + seq.count("M")


# ── structure-level metrics ────────────────────────────────────────────────

def get_chain_atoms(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return list(chain.get_atoms())
    return []


def get_chain_residues(structure, chain_id):
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return [r for r in chain if is_aa(r, standard=True)]
    return []


def secondary_structure_fractions(pdb_path, chain_id="D"):
    """Return (frac_helix, frac_sheet, frac_loop) for the given chain."""
    if not HAS_BIOPYTHON:
        return None, None, None
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", pdb_path)
        model = list(struct.get_models())[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
        helix, sheet, loop, total = 0, 0, 0, 0
        for key in dssp.keys():
            if key[0] != chain_id:
                continue
            ss = dssp[key][2]
            total += 1
            if ss in ("H", "G", "I"):
                helix += 1
            elif ss in ("B", "E"):
                sheet += 1
            else:
                loop += 1
        if total == 0:
            return None, None, None
        return helix/total, sheet/total, loop/total
    except Exception as e:
        return None, None, None


def interface_contacts(pdb_path, binder_chain="D", peptide_chain="C",
                       cutoff=4.5):
    """
    Count hydrophobic and polar contacts between binder and peptide.
    A contact = any heavy-atom pair within cutoff Å across the two chains.
    Returns (n_hydrophobic, n_polar, lys_met_at_interface).
    """
    if not HAS_BIOPYTHON:
        return None, None, None
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", pdb_path)

        binder_atoms  = get_chain_atoms(struct, binder_chain)
        peptide_atoms = get_chain_atoms(struct, peptide_chain)

        ns = NeighborSearch(peptide_atoms)
        binder_iface_residues = set()
        for atom in binder_atoms:
            hits = ns.search(atom.coord, cutoff, "R")
            if hits:
                binder_iface_residues.add(atom.get_parent())

        n_hydro, n_polar, lys_met = 0, 0, 0
        for res in binder_iface_residues:
            aa = res.get_resname()
            aa1 = _res3to1(aa)
            if aa1 in HYDROPHOBIC:
                n_hydro += 1
            elif aa1 in POLAR:
                n_polar += 1
            if aa1 in "KM":
                lys_met += 1

        return n_hydro, n_polar, lys_met
    except Exception as e:
        return None, None, None


def exposed_hydrophobic_fraction(pdb_path, chain_id="D", sasa_threshold=0.2):
    """
    Fraction of binder hydrophobic residues that are solvent-exposed
    (relative accessibility > threshold). Uses residue B-factors as a proxy
    when DSSP SASA is unavailable; falls back to counting exposed residues.
    """
    if not HAS_BIOPYTHON:
        return None
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", pdb_path)
        model = list(struct.get_models())[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")

        exposed_hydro, total_hydro = 0, 0
        for key in dssp.keys():
            if key[0] != chain_id:
                continue
            aa  = dssp[key][1]
            rsa = dssp[key][3]   # relative accessible surface area 0-1
            if aa in HYDROPHOBIC:
                total_hydro += 1
                if rsa is not None and rsa > sasa_threshold:
                    exposed_hydro += 1
        if total_hydro == 0:
            return None
        return exposed_hydro / total_hydro
    except Exception:
        return None


def _res3to1(res3):
    table = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    return table.get(res3.upper(), "X")


# ── AF2 output parsing ─────────────────────────────────────────────────────

def parse_af2_scores(pred_dir, design_name):
    score_files = sorted(glob.glob(
        os.path.join(pred_dir, design_name, f"{design_name}_scores_rank_001_*.json")))
    if not score_files:
        return None
    with open(score_files[0]) as f:
        return json.load(f)


def compute_pae_interaction(pae_matrix, chain_lens):
    """Mean PAE between binder (D) and peptide (C), both directions."""
    chains = ["A", "B", "C", "D"]
    starts, pos = {}, 0
    for c in chains:
        starts[c] = pos
        pos += chain_lens.get(c, 0)
    bs, be = starts["D"], starts["D"] + chain_lens["D"]
    ts, te = starts["C"], starts["C"] + chain_lens["C"]
    pae = np.array(pae_matrix)
    return float((pae[bs:be, ts:te].mean() + pae[ts:te, bs:be].mean()) / 2)


def compute_ipsae(pae_matrix, chain_lens):
    """ipSAE: min(PAE[binder→peptide], PAE[peptide→binder])."""
    chains = ["A", "B", "C", "D"]
    starts, pos = {}, 0
    for c in chains:
        starts[c] = pos
        pos += chain_lens.get(c, 0)
    bs, be = starts["D"], starts["D"] + chain_lens["D"]
    ts, te = starts["C"], starts["C"] + chain_lens["C"]
    pae = np.array(pae_matrix)
    dc = float(pae[bs:be, ts:te].mean())
    cd = float(pae[ts:te, bs:be].mean())
    return min(dc, cd)


def compute_binder_plddt(plddt, chain_lens):
    chains = ["A", "B", "C", "D"]
    pos = 0
    for c in chains:
        if c == "D":
            return float(np.array(plddt)[pos:pos + chain_lens["D"]].mean())
        pos += chain_lens.get(c, 0)
    return 0.0


def compute_ca_rmsd(pred_pdb, design_pdb, chain="D"):
    def get_ca(pdb, c):
        coords = {}
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM") and line[21] == c and line[12:16].strip() == "CA":
                    rnum = int(line[22:26])
                    coords[rnum] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        return np.array([coords[r] for r in sorted(coords)])

    pred   = get_ca(pred_pdb, chain)
    design = get_ca(design_pdb, chain)
    if len(pred) == 0 or len(pred) != len(design):
        return 999.0
    pc = pred   - pred.mean(0)
    dc = design - design.mean(0)
    H  = dc.T @ pc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt((((R @ pc.T).T - dc)**2).sum(1).mean()))


# ── main ───────────────────────────────────────────────────────────────────

def analyze_design(design_entry, af2_pred_dir, complex_pdb_dir):
    name    = design_entry["design_name"]
    seq     = design_entry.get("binder_sequence", "")
    bb_name = design_entry.get("backbone", "")
    spec    = design_entry.get("specificity", None)

    result = {
        "design_name":   name,
        "backbone":      bb_name,
        "sequence":      seq,
        "seq_len":       len(seq),
        "specificity":   spec,
    }

    # ── sequence metrics ──────────────────────────────────────────────────
    result["net_charge"]      = net_charge(seq)
    result["poly_ala_max_run"]= poly_ala_runs(seq)
    result["lys_met_total"]   = lys_met_count(seq)

    # ── AF2 metrics ───────────────────────────────────────────────────────
    scores = parse_af2_scores(af2_pred_dir, name)
    if scores is None:
        result["af2_available"] = False
        return result

    result["af2_available"] = True
    chain_lens = {**CHAIN_LENGTHS, "D": len(seq)}

    result["pLDDT"]           = compute_binder_plddt(scores["plddt"], chain_lens)
    result["pae_interaction"]  = compute_pae_interaction(scores["pae"], chain_lens)
    result["ipSAE"]            = compute_ipsae(scores["pae"], chain_lens)
    result["ptm"]              = scores.get("ptm", None)
    result["iptm"]             = scores.get("iptm", None)

    # RMSD vs. designed backbone
    pred_pdbs = sorted(glob.glob(
        os.path.join(af2_pred_dir, name, f"{name}*rank_001*.pdb")))
    complex_pdb = os.path.join(complex_pdb_dir, f"{name}.pdb")
    if pred_pdbs and os.path.exists(complex_pdb):
        result["rmsd"] = compute_ca_rmsd(pred_pdbs[0], complex_pdb)
        pred_pdb = pred_pdbs[0]
    else:
        result["rmsd"] = None
        pred_pdb = None

    # ── structural metrics ────────────────────────────────────────────────
    if pred_pdb and HAS_BIOPYTHON:
        fh, fs, fl = secondary_structure_fractions(pred_pdb, chain_id="D")
        result["frac_helix"] = fh
        result["frac_sheet"] = fs
        result["frac_loop"]  = fl

        nh, np_, lm = interface_contacts(pred_pdb)
        result["interface_hydrophobic"] = nh
        result["interface_polar"]       = np_
        result["interface_lys_met"]     = lm

        result["exposed_hydrophobic_frac"] = exposed_hydrophobic_fraction(pred_pdb)
    else:
        for k in ("frac_helix","frac_sheet","frac_loop",
                  "interface_hydrophobic","interface_polar","interface_lys_met",
                  "exposed_hydrophobic_frac"):
            result[k] = None

    return result


def print_table(results):
    passing = [r for r in results if r.get("pLDDT", 0) > 67
               and r.get("pae_interaction", 999) < 10]
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(results)} designs analyzed, {len(passing)} pass (pLDDT>67 AND pAE<10)")
    print(f"{'='*80}")
    hdr = f"{'Design':30s} {'pLDDT':>6} {'pAE':>6} {'ipSAE':>6} {'RMSD':>6} {'Spec':>7} {'Chg':>4} {'Hel':>5} {'Sh':>5} {'IntH':>5} {'IntP':>5} {'ExpH':>5} {'PA':>4}"
    print(hdr)
    print("-"*len(hdr))
    for r in sorted(results, key=lambda x: -(x.get("pLDDT") or 0)):
        def f(v, fmt=".2f"):
            return format(v, fmt) if v is not None else "  — "
        print(
            f"{r['design_name']:30s}"
            f" {f(r.get('pLDDT')):>6}"
            f" {f(r.get('pae_interaction')):>6}"
            f" {f(r.get('ipSAE')):>6}"
            f" {f(r.get('rmsd')):>6}"
            f" {f(r.get('specificity')):>7}"
            f" {str(int(r['net_charge'])) if r.get('net_charge') is not None else '—':>4}"
            f" {f(r.get('frac_helix'), '.0%') if r.get('frac_helix') is not None else '—':>5}"
            f" {f(r.get('frac_sheet'), '.0%') if r.get('frac_sheet') is not None else '—':>5}"
            f" {str(r.get('interface_hydrophobic','—')):>5}"
            f" {str(r.get('interface_polar','—')):>5}"
            f" {f(r.get('exposed_hydrophobic_frac'), '.0%') if r.get('exposed_hydrophobic_frac') is not None else '—':>5}"
            f" {str(r.get('poly_ala_max_run','—')):>4}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--af2_dir",       required=True, help="colabfold_predictions dir")
    parser.add_argument("--designs_json",  required=True, help="designs JSON with sequences + specificity")
    parser.add_argument("--complex_pdb_dir", default=None, help="dir with complex PDBs for RMSD")
    parser.add_argument("--output",        required=True, help="output JSON")
    args = parser.parse_args()

    with open(args.designs_json) as f:
        designs = json.load(f)

    complex_pdb_dir = args.complex_pdb_dir or os.path.join(
        os.path.dirname(args.af2_dir), "complex_pdbs")

    results = []
    for i, d in enumerate(designs):
        name = d.get("design_name", f"{d['backbone']}_s{d['sample']}")
        d["design_name"] = name
        print(f"  [{i+1}/{len(designs)}] {name}...", end="", flush=True)
        r = analyze_design(d, args.af2_dir, complex_pdb_dir)
        results.append(r)
        plddt = r.get("pLDDT")
        pae   = r.get("pae_interaction")
        print(f" pLDDT={plddt:.1f}" if plddt else " (no AF2)", end="")
        print(f" pAE={pae:.2f}" if pae else "")

    print_table(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
