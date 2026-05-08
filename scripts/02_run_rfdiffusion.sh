#!/bin/bash
# Run RFdiffusion to generate miniprotein binder backbones against pMHC (9UV8)
# Target: HLA-A*11:01 + KRAS G12D 9-mer peptide VVGADGVGK
# Hotspots: C1 (P1=VAL), C4 (P4=ALA), C5 (P5=ASP/G12D), C7 (P7=VAL)
# Binder length: 70-80 residues

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_DIR="$(dirname "$SCRIPT_DIR")/designs"
RFDIFF_DIR="/workspace/RFdiffusion"
MODEL_PATH="${RFDIFF_DIR}/models/Complex_base_ckpt.pt"
TARGET_PDB="${DESIGN_DIR}/target_9UV8_clean.pdb"
OUTPUT_DIR="${DESIGN_DIR}/rfdiffusion"

NUM_DESIGNS=${1:-500}

echo "=== Running RFdiffusion for pMHC binder design ==="
echo "Target: HLA-A*11:01 + KRAS G12D (VVGADGVGK)"
echo "Hotspots: C1, C4, C5 (G12D), C7"
echo "Binder length: 70-80 residues"
echo "Number of designs: ${NUM_DESIGNS}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# DGLBACKEND=pytorch is required; e3nn and graphbolt patched for PyTorch 2.7 compat
export DGLBACKEND=pytorch

conda run -n rfdiffusion \
    env DGLBACKEND=pytorch \
    python "${RFDIFF_DIR}/scripts/run_inference.py" \
    inference.input_pdb="${TARGET_PDB}" \
    "contigmap.contigs=[A1-275/0 B1-99/0 C1-9/0 70-80]" \
    "ppi.hotspot_res=[C1,C4,C5,C7]" \
    inference.output_prefix="${OUTPUT_DIR}/binder" \
    inference.num_designs="${NUM_DESIGNS}" \
    "inference.ckpt_override_path=${MODEL_PATH}" \
    denoiser.noise_scale_ca=1 \
    denoiser.noise_scale_frame=1 \
    2>&1 | tee "${OUTPUT_DIR}/rfdiffusion_run.log"

echo ""
echo "=== RFdiffusion complete ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of designs generated: $(ls ${OUTPUT_DIR}/*.pdb 2>/dev/null | wc -l)"
