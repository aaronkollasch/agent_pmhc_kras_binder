#!/bin/bash
# Round 2: Partial diffusion (scaffold recycling) on top round-1 candidates
# partial_T=20 out of 50 denoising steps (per Liu et al. 2025: range 12-25)
# Seeds: binder_7_T03_s7, binder_7_T03_s2, binder_17_s1, binder_93_s6
#
# CRITICAL: For partial diffusion of a binder in a PPI context, the RFdiffusion
# PPI sampler places the BINDER FIRST in its internal ordering. The assertion
# hal_idx0 == ref_idx0 (required for partial_T) means the input PDB must also
# have the BINDER FIRST (chain D before A,B,C). The contig uses the BINDER
# as a DESIGNABLE region (length only, no chain letter) so it gets noise added.
# Wrong: D1-79 (treats binder as fixed motif → zero diversity)
# Right: 79-79/0 A1-275/0 B1-99/0 C1-9/0 (binder designable, receptor fixed)

set -e

RFDIFF_DIR="/workspace/RFdiffusion"
DESIGN_DIR="/workspace/pmhc_design/designs"
OUTPUT_BASE="${DESIGN_DIR}/rfdiffusion_r2"
SEEDS_DIR="${DESIGN_DIR}/rfdiffusion_r2_seeds"
NUM_DESIGNS=${1:-200}  # per seed
PARTIAL_T=${2:-20}

mkdir -p "${OUTPUT_BASE}" "${SEEDS_DIR}"
export DGLBACKEND=pytorch

# Binder-first reordered PDBs (created by prepare step below)
declare -A SEEDS=(
  ["b7_t03_s7"]="${SEEDS_DIR}/b7_t03_s7_binder_first.pdb:79"
  ["b7_t03_s2"]="${SEEDS_DIR}/b7_t03_s2_binder_first.pdb:79"
  ["b17_s1"]="${SEEDS_DIR}/b17_s1_binder_first.pdb:75"
  ["b93_s6"]="${SEEDS_DIR}/b93_s6_binder_first.pdb:79"
)

for SEED_NAME in b7_t03_s7 b17_s1 b93_s6 b7_t03_s2; do
  IFS=':' read -r SEED_PDB BINDER_LEN <<< "${SEEDS[$SEED_NAME]}"
  OUT_DIR="${OUTPUT_BASE}/${SEED_NAME}"
  mkdir -p "${OUT_DIR}"
  # Contig: binder (designable, gets noise) first, then receptor (fixed)
  CONTIG="${BINDER_LEN}-${BINDER_LEN}/0 A1-275/0 B1-99/0 C1-9/0"

  echo "=== Partial diffusion: ${SEED_NAME} (partial_T=${PARTIAL_T}, n=${NUM_DESIGNS}) ==="
  echo "  Seed PDB: ${SEED_PDB}"
  echo "  Contig: ${CONTIG}"

  conda run -n rfdiffusion \
    env DGLBACKEND=pytorch \
    python "${RFDIFF_DIR}/scripts/run_inference.py" \
    inference.input_pdb="${SEED_PDB}" \
    "contigmap.contigs=[${CONTIG}]" \
    "ppi.hotspot_res=[C1,C4,C5,C7]" \
    inference.output_prefix="${OUT_DIR}/binder" \
    inference.num_designs="${NUM_DESIGNS}" \
    "inference.ckpt_override_path=${RFDIFF_DIR}/models/Complex_base_ckpt.pt" \
    diffuser.partial_T="${PARTIAL_T}" \
    denoiser.noise_scale_ca=1 \
    denoiser.noise_scale_frame=1 \
    2>&1 | tee "${OUT_DIR}/rfdiffusion.log"

  echo "  Done: ${SEED_NAME}"
done

echo ""
echo "=== All partial diffusion runs complete ==="
echo "Output: ${OUTPUT_BASE}"
