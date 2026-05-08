#!/bin/bash
# Run ProteinMPNN on RFdiffusion output backbones to design sequences
# For each binder backbone, generate 8 sequences
# The binder chain (D) is redesigned; MHC chains (A, B, C) are fixed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_DIR="$(dirname "$SCRIPT_DIR")/designs"
MPNN_DIR="/workspace/ProteinMPNN"
BINDER_DIR="${DESIGN_DIR}/rfdiffusion"
OUTPUT_DIR="${DESIGN_DIR}/mpnn"
NUM_SEQS=${1:-8}

echo "=== Running ProteinMPNN for sequence design ==="
echo "Input: ${BINDER_DIR}/*.pdb"
echo "Output: ${OUTPUT_DIR}"
echo "Sequences per backbone: ${NUM_SEQS}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# Get list of binder PDBs (exclude test files)
BINDER_PDBS=$(ls "${BINDER_DIR}"/binder_*.pdb 2>/dev/null || ls "${BINDER_DIR}"/test_binder_*.pdb 2>/dev/null)
N_PDBS=$(echo "$BINDER_PDBS" | wc -l)
echo "Found ${N_PDBS} binder PDBs"

# Create a directory with softlinks to just the binder PDBs
TMPDIR="${OUTPUT_DIR}/tmp_pdbs"
mkdir -p "$TMPDIR"
for pdb in $BINDER_PDBS; do
    ln -sf "$(realpath $pdb)" "$TMPDIR/$(basename $pdb)"
done

# Run ProteinMPNN
# Fixed chains: A (HLA heavy chain), B (beta2m), C (peptide)
# Designed chain: D (binder)
/venv/main/bin/python3 "${MPNN_DIR}/protein_mpnn_run.py" \
    --pdb_path_multi "$TMPDIR" \
    --out_folder "${OUTPUT_DIR}" \
    --num_seq_per_target "${NUM_SEQS}" \
    --sampling_temp "0.1" \
    --fixed_chains "A B C" \
    --batch_size 1 \
    --save_score 1 \
    --save_probs 1 \
    2>&1 | tee "${OUTPUT_DIR}/mpnn_run.log"

echo ""
echo "=== ProteinMPNN complete ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of FASTA files: $(ls ${OUTPUT_DIR}/seqs/*.fa 2>/dev/null | wc -l)"
