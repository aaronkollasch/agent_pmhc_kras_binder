#!/usr/bin/env python3
"""Analyze round-1 qualifying candidates with comprehensive metrics."""
import json, glob, os, numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

HYDROPHOBIC = set("VILMFYWAC")
POLAR       = set("STNQHKRDE")
CHARGE_MAP  = {"K":+1,"R":+1,"H":+0.5,"D":-1,"E":-1}
CHAIN_LENS  = {"A":276,"B":99,"C":9}
AA3TO1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
           "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
           "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

# Priority / display metadata
META = {
    "binder_7_s7":  {"priority":1,  "label":"binder_7_T03_s7", "mechanism":"Asp@51→P5"},
    "binder_7_s2":  {"priority":2,  "label":"binder_7_T03_s2", "mechanism":"Asp@51→P5"},
    "binder_17_s1": {"priority":3,  "label":"binder_17_s1",    "mechanism":"Arg@44 salt-bridge"},
    "binder_7_s6":  {"priority":4,  "label":"binder_7_s6",     "mechanism":"Gln@51→P5"},
    "binder_93_s6": {"priority":5,  "label":"binder_93_s6",    "mechanism":"Arg@55 salt-bridge"},
    "binder_7_s14": {"priority":6,  "label":"binder_7_T03_s14","mechanism":"Gln@51→P5"},
    "binder_7_s4":  {"priority":7,  "label":"binder_7_s4",     "mechanism":"Gln@51→P5"},
    "binder_7_s3":  {"priority":8,  "label":"binder_7_s3",     "mechanism":"Gln@51→P5"},
    "binder_86_s7": {"priority":9,  "label":"binder_86_s7",    "mechanism":"Gln@59→P5"},
}

def net_charge(seq):
    return sum(CHARGE_MAP.get(aa, 0) for aa in seq)

def poly_ala(seq):
    mx, cur = 0, 0
    for aa in seq:
        cur = cur+1 if aa=="A" else 0
        mx = max(mx, cur)
    return mx

def pae_metrics(score_json, binder_len):
    with open(score_json) as f:
        s = json.load(f)
    lens = {**CHAIN_LENS, "D": binder_len}
    starts, pos = {}, 0
    for c in ["A","B","C","D"]:
        starts[c] = pos; pos += lens[c]
    bs, be = starts["D"], starts["D"]+lens["D"]
    ts, te = starts["C"], starts["C"]+lens["C"]
    pae = np.array(s["pae"])
    pae_i = float((pae[bs:be,ts:te].mean() + pae[ts:te,bs:be].mean()) / 2)
    ipsae = min(float(pae[bs:be,ts:te].mean()), float(pae[ts:te,bs:be].mean()))
    plddt = float(np.array(s["plddt"])[bs:be].mean())
    return plddt, pae_i, ipsae, s.get("ptm"), s.get("iptm")

def ca_rmsd(pred_pdb, ref_pdb, chain="D"):
    def get_ca(path):
        coords = {}
        with open(path) as f:
            for line in f:
                if line.startswith("ATOM") and line[21]==chain and line[12:16].strip()=="CA":
                    coords[int(line[22:26])] = [float(line[30:38]),float(line[38:46]),float(line[46:54])]
        return np.array([coords[k] for k in sorted(coords)])
    p, d = get_ca(pred_pdb), get_ca(ref_pdb)
    if not len(p) or len(p) != len(d):
        return 999.
    pc, dc = p-p.mean(0), d-d.mean(0)
    U, S, Vt = np.linalg.svd(dc.T @ pc)
    R = Vt.T @ np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]) @ U.T
    return float(np.sqrt((((R@pc.T).T - dc)**2).sum(1).mean()))

def struct_metrics(pred_pdb):
    try:
        fa = pdb_io.PDBFile.read(pred_pdb)
        atoms = pdb_io.get_structure(fa, model=1)
        cd = atoms[atoms.chain_id == "D"]
        cc = atoms[atoms.chain_id == "C"]
        sse = struc.annotate_sse(cd)
        fh = float(np.sum(sse=="a")) / len(sse)
        fs = float(np.sum(sse=="b")) / len(sse)
        fl = float(np.sum(sse=="c")) / len(sse)
        cell = struc.CellList(cc, cell_size=4.5)
        hits = cell.get_atoms(cd.coord, radius=4.5)
        iface = set()
        for i, h in enumerate(hits):
            if len(h) > 0:
                iface.add((cd.res_id[i], cd.res_name[i]))
        n_h = sum(1 for (_, rn) in iface if AA3TO1.get(rn,"X") in HYDROPHOBIC)
        n_p = sum(1 for (_, rn) in iface if AA3TO1.get(rn,"X") in POLAR)
        km  = sum(1 for (_, rn) in iface if AA3TO1.get(rn,"X") in "KM")
        return fh, fs, fl, n_h, n_p, km
    except Exception as e:
        return None, None, None, None, None, None

with open("/workspace/pmhc_design/designs/r1_candidates_for_analysis.json") as f:
    candidates = json.load(f)

results = []
for d in sorted(candidates, key=lambda x: META.get(x["design_name"],{}).get("priority",99)):
    name     = d["design_name"]
    seq      = d["binder_sequence"]
    af2_dir  = d["_af2_dir"]
    cpdb_dir = d["_complex_pdb_dir"]
    spec     = d["specificity"]
    meta     = META.get(name, {})

    pred_pdbs   = sorted(glob.glob(os.path.join(af2_dir, name, f"{name}*rank_001*.pdb")))
    score_jsons = sorted(glob.glob(os.path.join(af2_dir, name, f"{name}_scores_rank_001_*.json")))
    complex_pdb = os.path.join(cpdb_dir, f"{name}.pdb")

    row = {
        "priority":   meta.get("priority"),
        "label":      meta.get("label", name),
        "mechanism":  meta.get("mechanism"),
        "seq":        seq,
        "spec":       spec,
        "net_charge": net_charge(seq),
        "poly_ala":   poly_ala(seq),
        "lys_met":    seq.count("K") + seq.count("M"),
    }

    if score_jsons:
        plddt, pae_i, ipsae, ptm, iptm = pae_metrics(score_jsons[0], len(seq))
        row.update(pLDDT=plddt, pae_interaction=pae_i, ipSAE=ipsae, ptm=ptm, iptm=iptm)

    if pred_pdbs and os.path.exists(complex_pdb):
        row["rmsd"] = ca_rmsd(pred_pdbs[0], complex_pdb)

    if pred_pdbs:
        fh, fs, fl, n_h, n_p, km = struct_metrics(pred_pdbs[0])
        row.update(frac_helix=fh, frac_sheet=fs, frac_loop=fl,
                   interface_hydrophobic=n_h, interface_polar=n_p, interface_lys_met=km)

    results.append(row)

# ── Print table ──────────────────────────────────────────────────────────────
print("\n" + "="*120)
print("ROUND 1 CANDIDATE ANALYSIS")
print("="*120)
hdr = (f"{'#':>2} {'Label':22s} {'Spec':>7} {'pLDDT':>6} {'pAE':>6} {'ipSAE':>6} "
       f"{'RMSD':>5} {'Chg':>4} {'Helix':>6} {'Sheet':>6} {'IntH':>5} {'IntP':>5} "
       f"{'KM_i':>5} {'polyA':>5} {'Mechanism'}")
print(hdr)
print("-"*120)
for r in results:
    def f(v, fmt=".2f"): return format(v, fmt) if v is not None else "—"
    print(
        f"{r['priority']:>2} {r['label']:22s}"
        f" {r['spec']:+7.3f}"
        f" {f(r.get('pLDDT')):>6}"
        f" {f(r.get('pae_interaction')):>6}"
        f" {f(r.get('ipSAE')):>6}"
        f" {f(r.get('rmsd')):>5}"
        f" {r['net_charge']:+4.0f}"
        f" {f(r.get('frac_helix'),'.0%') if r.get('frac_helix') is not None else '—':>6}"
        f" {f(r.get('frac_sheet'),'.0%') if r.get('frac_sheet') is not None else '—':>6}"
        f" {str(r.get('interface_hydrophobic','—')):>5}"
        f" {str(r.get('interface_polar','—')):>5}"
        f" {str(r.get('interface_lys_met','—')):>5}"
        f" {r['poly_ala']:>5}"
        f"  {r['mechanism']}"
    )

with open("/workspace/pmhc_design/designs/r1_analysis_results.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to r1_analysis_results.json")
print(f"\nNOTE — metrics requiring Rosetta (not computed): CMS scores, FastRelax energetics,")
print(f"        shape complementarity, packstat, unsaturated H-bonds.")
