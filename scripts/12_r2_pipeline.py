#!/usr/bin/env python3
"""
Round-2 pipeline: geometry screen → MPNN → specificity → AF2 → analysis.

Processes partial-diffusion output for one seed at a time.

Usage:
  python3 12_r2_pipeline.py --seed b7_t03_s7 [--max_af2 40] [--skip_af2]
"""

import os, sys, json, glob, argparse, subprocess, tempfile, shutil
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DESIGN_DIR    = "/workspace/pmhc_design/designs"
SCRIPTS_DIR   = "/workspace/pmhc_design/scripts"
RFDIFF_R2     = f"{DESIGN_DIR}/rfdiffusion_r2"
MPNN_DIR      = "/workspace/ProteinMPNN"
TARGET_G12D   = f"{DESIGN_DIR}/target_9UV8_clean.pdb"
TARGET_WT     = f"{DESIGN_DIR}/target_9UV8_wt.pdb"

HOTSPOT_RESNUMS = {1, 4, 5, 7}   # chain-C residues C1,C4,C5(G12D),C7
P5_RESNUM       = 5               # Asp in G12D peptide

AA1TO3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
           'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
           'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
AA3TO1 = {v: k for k, v in AA1TO3.items()}
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}

TARGET_SEQS = {
    'A': "GSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQTDRVDLG"
         "TLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHAAEQQR"
         "AYLEGRCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRP"
         "AGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWE",
    'B': "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYA"
         "CRVNHVTLSQPKIVKWDRDM",
    'C': "VVGADGVGK",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Geometry screening
# ═══════════════════════════════════════════════════════════════════════════════

def get_cb_positions(pdb_file, chain):
    atoms = {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            if line[21] != chain:
                continue
            aname  = line[12:16].strip()
            rnum   = int(line[22:26])
            xyz    = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            if rnum not in atoms:
                atoms[rnum] = {'ca': None, 'cb': None, 'resname': line[17:20].strip()}
            if aname == 'CA':
                atoms[rnum]['ca'] = xyz
            elif aname == 'CB':
                atoms[rnum]['cb'] = xyz
    result = {}
    for rnum, v in atoms.items():
        if v['ca'] is not None:
            result[rnum] = {'cb': v['cb'] if v['cb'] is not None else v['ca'],
                            'ca': v['ca'], 'resname': v['resname']}
    return result


def score_backbone(pdb_file, contact_cutoff=8.0, p5_reach=9.5):
    """Return geometry metrics dict; None if binder chain D absent."""
    pep_cb   = get_cb_positions(pdb_file, 'C')
    binder_cb = get_cb_positions(pdb_file, 'D')

    if not binder_cb or not pep_cb:
        return None

    n_binder = len(binder_cb)
    pep_rnums = sorted(pep_cb.keys())

    pep_contacts = set()   # binder residue indices in contact with peptide
    hotspot_hit  = set()

    p5_min = 999.0
    p4_min = 999.0
    p7_min = 999.0
    near_p5 = []

    for bi, bdata in binder_cb.items():
        bcb = bdata['cb']
        for pi, pdata in pep_cb.items():
            d = float(np.linalg.norm(bcb - pdata['cb']))
            if d < contact_cutoff:
                pep_contacts.add(bi)
                if pi in HOTSPOT_RESNUMS:
                    hotspot_hit.add(pi)
            if pi == P5_RESNUM:
                if d < p5_min:
                    p5_min = d
                if d < p5_reach:
                    near_p5.append({'binder_rnum': bi, 'dist_to_p5': d})
            elif pi == 4:
                p4_min = min(p4_min, d)
            elif pi == 7:
                p7_min = min(p7_min, d)

    hotspot_coverage = len(hotspot_hit) / len(HOTSPOT_RESNUMS)
    pep_frac = len(pep_contacts) / n_binder if n_binder else 0

    return {
        'backbone': os.path.splitext(os.path.basename(pdb_file))[0],
        'pdb_file': pdb_file,
        'binder_len': n_binder,
        'n_pep_contacts': len(pep_contacts),
        'hotspot_coverage': round(hotspot_coverage, 3),
        'hotspots_contacted': sorted(hotspot_hit),
        'p5_min_dist': round(float(p5_min), 3),
        'p4_min_dist': round(float(p4_min), 3),
        'p7_min_dist': round(float(p7_min), 3),
        'pep_frac': round(pep_frac, 3),
        'near_p5': near_p5,
        'geo_pass': p5_min < p5_reach,
    }


def run_geometry_screen(binder_dir, output_dir):
    pdbs = sorted(glob.glob(os.path.join(binder_dir, "binder_*.pdb")))
    print(f"  Scoring {len(pdbs)} backbones...")
    results = []
    for pdb in pdbs:
        r = score_backbone(pdb)
        if r:
            results.append(r)

    results.sort(key=lambda x: x['p5_min_dist'])

    with open(os.path.join(output_dir, "geo_scores.json"), 'w') as f:
        json.dump(results, f, indent=2)

    passing = [r for r in results if r['geo_pass']]
    print(f"  {len(passing)}/{len(results)} pass geometry (p5_min_dist < 9.5 Å)")
    for r in passing[:10]:
        print(f"    {r['backbone']:20s}  p5={r['p5_min_dist']:.2f}  "
              f"hotspot_cov={r['hotspot_coverage']:.2f}  pep_contacts={r['n_pep_contacts']}")
    return passing


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ProteinMPNN sequence design
# ═══════════════════════════════════════════════════════════════════════════════

def run_mpnn(pdb_files, mpnn_dir, num_seqs=8, temperature=0.1, design_chain='D'):
    """Run MPNN on given PDB files, return list of designs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Symlink PDBs
        for pdb in pdb_files:
            dst = os.path.join(tmpdir, os.path.basename(pdb))
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(pdb), dst)

        parsed_jsonl  = os.path.join(tmpdir, "parsed.jsonl")
        assigned_jsonl = os.path.join(tmpdir, "assigned.jsonl")
        out_dir       = os.path.join(tmpdir, "mpnn_out")
        os.makedirs(out_dir, exist_ok=True)

        subprocess.run([
            "/venv/main/bin/python3",
            f"{MPNN_DIR}/helper_scripts/parse_multiple_chains.py",
            "--input_path", tmpdir, "--output_path", parsed_jsonl,
        ], check=True, capture_output=True)

        subprocess.run([
            "/venv/main/bin/python3",
            f"{MPNN_DIR}/helper_scripts/assign_fixed_chains.py",
            "--input_path", parsed_jsonl, "--output_path", assigned_jsonl,
            "--chain_list", design_chain,
        ], check=True, capture_output=True)

        subprocess.run([
            "/venv/main/bin/python3", f"{MPNN_DIR}/protein_mpnn_run.py",
            "--jsonl_path", parsed_jsonl,
            "--chain_id_jsonl", assigned_jsonl,
            "--out_folder", out_dir,
            "--num_seq_per_target", str(num_seqs),
            "--sampling_temp", str(temperature),
            "--batch_size", "1",
            "--save_score", "1",
        ], check=True, capture_output=False)

        designs = _parse_mpnn_output(os.path.join(out_dir, "seqs"), pdb_files)

    return designs


def _parse_mpnn_output(seqs_dir, pdb_files):
    pdb_lens = {}
    for pdb in pdb_files:
        bname = os.path.splitext(os.path.basename(pdb))[0]
        lens = {}
        with open(pdb) as f:
            for line in f:
                if line.startswith('ATOM'):
                    c = line[21]
                    lens[c] = lens.get(c, set())
                    lens[c].add(int(line[22:26]))
        pdb_lens[bname] = {c: len(v) for c, v in lens.items()}

    designs = []
    for fa_file in sorted(glob.glob(os.path.join(seqs_dir, "*.fa"))):
        backbone = os.path.splitext(os.path.basename(fa_file))[0]
        entries = []
        with open(fa_file) as f:
            hdr = None
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    hdr = line[1:]
                    entries.append({'header': hdr, 'seq': ''})
                elif entries:
                    entries[-1]['seq'] += line

        for i, e in enumerate(entries):
            if i == 0:
                continue  # skip poly-G reference
            hdict = {}
            for part in e['header'].split(', '):
                if '=' in part:
                    k, v = part.split('=', 1)
                    hdict[k.strip()] = v.strip()

            full_seq = e['seq']
            parts    = full_seq.split('/')
            lens     = pdb_lens.get(backbone, {})
            # Chain order in MPNN output matches PDB chain order: A B C D
            # Last part is chain D (binder)
            binder_seq = parts[-1] if len(parts) == 4 else full_seq

            designs.append({
                'backbone': backbone,
                'sample': i,
                'header': e['header'],
                'full_sequence': full_seq,
                'binder_sequence': binder_seq,
                'mpnn_score': float(hdict.get('score', 999)),
                'global_score': float(hdict.get('global_score', 999)),
            })
    return designs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Specificity scoring
# ═══════════════════════════════════════════════════════════════════════════════

def create_scored_pdb(target_pdb, backbone_pdb, binder_seq, output_pdb):
    target_lines = []
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in ('A', 'B', 'C'):
                target_lines.append(line)

    backbone_d = []
    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D':
                backbone_d.append(line)

    rnums, seen = [], set()
    for line in backbone_d:
        rn = int(line[22:26])
        if rn not in seen:
            rnums.append(rn); seen.add(rn)

    n = min(len(rnums), len(binder_seq))
    rn2res = {rn: AA1TO3.get(aa, 'GLY') for rn, aa in zip(rnums[:n], binder_seq[:n])}

    updated = []
    for line in backbone_d:
        rn = int(line[22:26])
        if rn not in rn2res:
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_ATOMS:
            continue
        new_line = line[:17] + f"{rn2res[rn]:<3}" + line[20:]
        updated.append(new_line)

    with open(output_pdb, 'w') as f:
        f.writelines(target_lines)
        f.write('TER\n')
        f.writelines(updated)
        f.write('TER\nEND\n')


def score_specificity_batch(designs, binder_dir, work_dir):
    """Run ProteinMPNN score_only for G12D and WT targets. Returns updated designs."""
    os.makedirs(work_dir, exist_ok=True)

    for target_label, target_pdb in [('g12d', TARGET_G12D), ('wt', TARGET_WT)]:
        pdb_dir = os.path.join(work_dir, f"pdbs_{target_label}")
        os.makedirs(pdb_dir, exist_ok=True)

        name_to_design = {}
        for d in designs:
            backbone_pdb = os.path.join(binder_dir, f"{d['backbone']}.pdb")
            if not os.path.exists(backbone_pdb):
                continue
            dname = f"{d['backbone']}_s{d['sample']}"
            out_pdb = os.path.join(pdb_dir, f"{dname}.pdb")
            try:
                create_scored_pdb(target_pdb, backbone_pdb, d['binder_sequence'], out_pdb)
                name_to_design[dname] = d
            except Exception as e:
                print(f"  PDB prep failed for {dname}: {e}")

        if not name_to_design:
            continue

        parsed_jsonl  = os.path.join(work_dir, f"parsed_{target_label}.jsonl")
        assigned_jsonl = os.path.join(work_dir, f"assigned_{target_label}.jsonl")
        scores_file   = os.path.join(work_dir, f"scores_{target_label}.npz")

        subprocess.run([
            "/venv/main/bin/python3",
            f"{MPNN_DIR}/helper_scripts/parse_multiple_chains.py",
            "--input_path", pdb_dir, "--output_path", parsed_jsonl,
        ], check=True, capture_output=True)

        subprocess.run([
            "/venv/main/bin/python3",
            f"{MPNN_DIR}/helper_scripts/assign_fixed_chains.py",
            "--input_path", parsed_jsonl, "--output_path", assigned_jsonl,
            "--chain_list", "D",
        ], check=True, capture_output=True)

        target_out = os.path.join(work_dir, f"mpnn_{target_label}")
        os.makedirs(target_out, exist_ok=True)

        subprocess.run([
            "/venv/main/bin/python3", f"{MPNN_DIR}/protein_mpnn_run.py",
            "--jsonl_path", parsed_jsonl,
            "--chain_id_jsonl", assigned_jsonl,
            "--out_folder", target_out,
            "--score_only", "1",
            "--batch_size", "1",
        ], check=True, capture_output=False)

        # MPNN writes score_only/{basename}_pdb.npz (note _pdb suffix)
        score_dir = os.path.join(target_out, "score_only")
        for d in designs:
            dname = f"{d['backbone']}_s{d['sample']}"
            sc_file = os.path.join(score_dir, f"{dname}_pdb.npz")
            if not os.path.exists(sc_file):
                continue
            data = np.load(sc_file, allow_pickle=True)
            score_key = f"score_{target_label}"
            d[score_key] = float(data['score'].mean())

    # Compute specificity = score_wt - score_g12d
    for d in designs:
        sg  = d.get('score_g12d')
        swt = d.get('score_wt')
        if sg is not None and swt is not None:
            d['specificity'] = float(swt - sg)

    return designs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — AF2 screening
# ═══════════════════════════════════════════════════════════════════════════════

def make_complex_pdb(target_pdb, backbone_pdb, binder_seq, output_pdb):
    """Same as create_scored_pdb but backbone-atoms only for chain D."""
    target_lines = []
    with open(target_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] in ('A', 'B', 'C'):
                target_lines.append(line)

    backbone_d = []
    with open(backbone_pdb) as f:
        for line in f:
            if line.startswith('ATOM') and line[21] == 'D':
                backbone_d.append(line)

    rnums, seen = [], set()
    for line in backbone_d:
        rn = int(line[22:26])
        if rn not in seen:
            rnums.append(rn); seen.add(rn)

    n = min(len(rnums), len(binder_seq))
    rn2res = {rn: AA1TO3.get(aa, 'GLY') for rn, aa in zip(rnums[:n], binder_seq[:n])}

    updated = []
    for line in backbone_d:
        rn = int(line[22:26])
        if rn not in rn2res or line[12:16].strip() not in BACKBONE_ATOMS:
            continue
        updated.append(line[:17] + f"{rn2res[rn]:<3}" + line[20:])

    with open(output_pdb, 'w') as f:
        f.writelines(target_lines)
        f.write('TER\n')
        f.writelines(updated)
        f.write('TER\nEND\n')


def run_colabfold(complex_pdb, pred_dir, design_name, num_recycles=3):
    done = os.path.join(pred_dir, design_name, f"{design_name}.done.txt")
    if os.path.exists(done):
        return True
    os.makedirs(os.path.join(pred_dir, design_name), exist_ok=True)
    result = subprocess.run([
        "/venv/colabfold/bin/colabfold_batch",
        "--initial-guess",
        "--msa-mode", "single_sequence",
        "--num-recycle", str(num_recycles),
        "--model-type", "alphafold2_multimer_v3",
        "--num-models", "1",
        complex_pdb,
        os.path.join(pred_dir, design_name),
    ], capture_output=True, text=True)
    if result.returncode == 0:
        with open(done, 'w') as f: f.write("done\n")
        return True
    print(f"  ColabFold failed: {result.stderr[-300:]}")
    return False


def parse_af2_result(pred_dir, design_name, backbone_pdb, binder_len):
    chains = {'A': 275, 'B': 99, 'C': 9, 'D': binder_len}
    score_files = sorted(glob.glob(
        os.path.join(pred_dir, design_name, f"{design_name}_scores_rank_001_*.json")))
    if not score_files:
        return {}
    with open(score_files[0]) as f:
        sc = json.load(f)

    starts, pos = {}, 0
    for c in ['A','B','C','D']:
        starts[c] = pos; pos += chains[c]
    bs, be = starts['D'], starts['D']+chains['D']
    ts, te = starts['C'], starts['C']+chains['C']
    pae = np.array(sc['pae'])
    pae_i = float((pae[bs:be,ts:te].mean() + pae[ts:te,bs:be].mean()) / 2)
    plddt = float(np.array(sc['plddt'])[bs:be].mean())

    # RMSD
    pred_pdbs = sorted(glob.glob(
        os.path.join(pred_dir, design_name, f"{design_name}*rank_001*.pdb")))
    rmsd = 999.0
    if pred_pdbs and backbone_pdb and os.path.exists(backbone_pdb):
        def get_ca(p, c):
            coords = {}
            with open(p) as f:
                for line in f:
                    if line.startswith('ATOM') and line[21]==c and line[12:16].strip()=='CA':
                        coords[int(line[22:26])] = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
            return np.array([coords[k] for k in sorted(coords)])
        try:
            pred_ca = get_ca(pred_pdbs[0], 'D')
            ref_ca  = get_ca(backbone_pdb, 'D')
            if len(pred_ca) and len(pred_ca) == len(ref_ca):
                pc, dc = pred_ca-pred_ca.mean(0), ref_ca-ref_ca.mean(0)
                U, S, Vt = np.linalg.svd(dc.T @ pc)
                R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]) @ U.T
                rmsd = float(np.sqrt((((R@pc.T).T - dc)**2).sum(1).mean()))
        except Exception:
            pass

    return {'pae_interaction': pae_i, 'binder_plddt': plddt,
            'binder_rmsd': rmsd, 'ptm': sc.get('ptm',0), 'iptm': sc.get('iptm',0)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True,
                   help="Seed name (e.g. b7_t03_s7). Must match subdir in rfdiffusion_r2/")
    p.add_argument("--max_af2", type=int, default=40,
                   help="Max designs to send to AF2 (top by specificity)")
    p.add_argument("--skip_af2", action='store_true',
                   help="Skip AF2 step (just geo+MPNN+spec)")
    p.add_argument("--num_seqs", type=int, default=8,  help="MPNN sequences per backbone")
    p.add_argument("--mpnn_temp", type=float, default=0.1, help="MPNN sampling temperature")
    p.add_argument("--geo_cutoff", type=float, default=9.5, help="p5 reach cutoff Å")
    p.add_argument("--plddt_thresh", type=float, default=67.0, help="pLDDT pass threshold")
    p.add_argument("--spec_thresh", type=float, default=0.025, help="Specificity pass threshold")
    return p.parse_args()


def main():
    args = parse_args()
    seed = args.seed

    binder_dir = os.path.join(RFDIFF_R2, seed)
    out_base   = os.path.join(DESIGN_DIR, f"r2_{seed}")
    os.makedirs(out_base, exist_ok=True)

    if not os.path.isdir(binder_dir):
        print(f"ERROR: binder directory not found: {binder_dir}", file=sys.stderr)
        sys.exit(1)

    n_pdbs = len(glob.glob(os.path.join(binder_dir, "binder_*.pdb")))
    print(f"\n{'='*70}")
    print(f"ROUND-2 PIPELINE: {seed}")
    print(f"{'='*70}")
    print(f"  Binder dir : {binder_dir}  ({n_pdbs} PDBs)")
    print(f"  Output dir : {out_base}")

    # ── Step 1: Geometry screen ────────────────────────────────────────────────
    print(f"\n[1/4] Geometry screening (p5_reach < {args.geo_cutoff} Å)...")
    geo_dir = os.path.join(out_base, "geo")
    os.makedirs(geo_dir, exist_ok=True)
    passing_geo = run_geometry_screen(binder_dir, geo_dir)

    if not passing_geo:
        print("  No backbones pass geometry screen. Stopping.")
        sys.exit(0)

    passing_pdbs = [r['pdb_file'] for r in passing_geo]

    # ── Step 2: MPNN ──────────────────────────────────────────────────────────
    print(f"\n[2/4] ProteinMPNN (T={args.mpnn_temp}, {args.num_seqs} seqs/backbone)...")
    designs = run_mpnn(passing_pdbs, MPNN_DIR, num_seqs=args.num_seqs,
                       temperature=args.mpnn_temp)
    print(f"  Generated {len(designs)} sequences from {len(passing_pdbs)} backbones")

    designs_file = os.path.join(out_base, "mpnn_designs.json")
    with open(designs_file, 'w') as f:
        json.dump(designs, f, indent=2)

    # ── Step 3: Specificity scoring ───────────────────────────────────────────
    print(f"\n[3/4] Specificity scoring (G12D vs WT)...")
    spec_dir = os.path.join(out_base, "spec_work")
    designs = score_specificity_batch(designs, binder_dir, spec_dir)

    has_spec = [d for d in designs if 'specificity' in d]
    print(f"  Scored {len(has_spec)}/{len(designs)} designs")
    if has_spec:
        specs = [d['specificity'] for d in has_spec]
        print(f"  Spec range: {min(specs):.3f} – {max(specs):.3f}  "
              f"(mean={np.mean(specs):.3f})")

    has_spec.sort(key=lambda x: -x['specificity'])

    # Save scored designs
    scored_file = os.path.join(out_base, "scored_designs.json")
    with open(scored_file, 'w') as f:
        json.dump(has_spec, f, indent=2, default=float)

    # Print top 20 by specificity
    print(f"\n  Top 20 by specificity:")
    print(f"  {'Design':30s} {'MPNN':>7} {'Spec':>8} {'Score_G12D':>10} {'Score_WT':>10}")
    print(f"  {'-'*67}")
    for d in has_spec[:20]:
        dname = f"{d['backbone']}_s{d['sample']}"
        print(f"  {dname:30s} {d['mpnn_score']:>7.3f} {d['specificity']:>+8.3f} "
              f"{d.get('score_g12d',999):>10.4f} {d.get('score_wt',999):>10.4f}")

    if args.skip_af2:
        print("\n  [skip_af2 set — stopping before AF2]")
        return

    # ── Step 4: AF2 screening ─────────────────────────────────────────────────
    af2_candidates = [d for d in has_spec if d.get('specificity', -999) >= args.spec_thresh]
    if not af2_candidates:
        print(f"\n  No designs pass spec ≥ {args.spec_thresh} — using top {args.max_af2}")
        af2_candidates = has_spec[:args.max_af2]
    else:
        af2_candidates = af2_candidates[:args.max_af2]

    print(f"\n[4/4] AF2 screening ({len(af2_candidates)} designs, initial-guess)...")

    af2_dir     = os.path.join(out_base, "af2")
    complex_dir = os.path.join(af2_dir, "complex_pdbs")
    pred_dir    = os.path.join(af2_dir, "colabfold_predictions")
    os.makedirs(complex_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    n_done = n_fail = 0
    af2_results = []

    for i, d in enumerate(af2_candidates):
        backbone_pdb = os.path.join(binder_dir, f"{d['backbone']}.pdb")
        dname        = f"{d['backbone']}_s{d['sample']}"
        complex_pdb  = os.path.join(complex_dir, f"{dname}.pdb")
        binder_len   = len(d['binder_sequence'])

        if not os.path.exists(backbone_pdb):
            print(f"  [{i+1}/{len(af2_candidates)}] {dname}: backbone PDB missing, skip")
            n_fail += 1
            continue

        try:
            make_complex_pdb(TARGET_G12D, backbone_pdb, d['binder_sequence'], complex_pdb)
        except Exception as e:
            print(f"  [{i+1}/{len(af2_candidates)}] {dname}: PDB prep failed: {e}")
            n_fail += 1
            continue

        print(f"  [{i+1}/{len(af2_candidates)}] {dname} "
              f"(len={binder_len}, spec={d.get('specificity',0):+.3f})...",
              end='', flush=True)

        ok = run_colabfold(complex_pdb, pred_dir, dname)
        if ok:
            n_done += 1
            print(" done")
        else:
            n_fail += 1
            print(" FAILED")
            continue

        metrics = parse_af2_result(pred_dir, dname, backbone_pdb, binder_len)
        af2_results.append({
            **d, 'design_name': dname, 'backbone_pdb': backbone_pdb,
            'complex_pdb': complex_pdb, **metrics,
        })

    print(f"\n  Completed: {n_done} done, {n_fail} failed")

    # Sort and display
    af2_results.sort(key=lambda x: x.get('binder_plddt', 0), reverse=True)

    passing = [r for r in af2_results
               if r.get('binder_plddt', 0) >= args.plddt_thresh
               and r.get('specificity', -999) >= args.spec_thresh]

    print(f"\n  {len(passing)}/{len(af2_results)} pass "
          f"(pLDDT ≥ {args.plddt_thresh} AND spec ≥ {args.spec_thresh})")

    print(f"\n  Top results by pLDDT:")
    print(f"  {'Design':30s} {'pLDDT':>6} {'pAE':>6} {'RMSD':>5} {'Spec':>7} {'MPNN':>6}")
    print(f"  {'-'*65}")
    for r in af2_results[:20]:
        flag = " ✓" if r in passing else ""
        print(f"  {r['design_name']:30s} "
              f"{r.get('binder_plddt',0):>6.2f} "
              f"{r.get('pae_interaction',999):>6.2f} "
              f"{r.get('binder_rmsd',999):>5.1f} "
              f"{r.get('specificity',0):>+7.3f} "
              f"{r.get('mpnn_score',999):>6.3f}{flag}")

    # Save AF2 results
    af2_file = os.path.join(out_base, "af2_results.json")
    with open(af2_file, 'w') as f:
        json.dump(af2_results, f, indent=2, default=float)

    pass_file = os.path.join(out_base, "af2_passing.json")
    with open(pass_file, 'w') as f:
        json.dump(passing, f, indent=2, default=float)

    print(f"\n  Results: {af2_file}")
    print(f"  Passing: {pass_file}")
    print(f"\n{'='*70}")
    print(f"DONE: {seed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
