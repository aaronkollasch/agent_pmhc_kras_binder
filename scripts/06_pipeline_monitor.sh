#!/bin/bash
# Continuous pipeline monitor: runs MPNN on new RFdiffusion backbones
# Run this in the background while RFdiffusion is generating designs.
#
# Usage: nohup bash 06_pipeline_monitor.sh > pipeline.log 2>&1 &

set -e

DESIGN_DIR="/workspace/pmhc_design/designs"
SCRIPTS_DIR="/workspace/pmhc_design/scripts"
RFDIFF_DIR="${DESIGN_DIR}/rfdiffusion"
MPNN_DIR="${DESIGN_DIR}/mpnn"
MPNN_T03_DIR="${DESIGN_DIR}/mpnn_t03"

echo "=== Pipeline Monitor Started ==="
echo "Checking for new RFdiffusion backbones every 5 minutes..."
echo "Started at: $(date)"

while true; do
    # Find all RFdiffusion backbones
    ALL_PDBS=$(ls "${RFDIFF_DIR}"/binder_*.pdb 2>/dev/null | sort -V)
    N_TOTAL=$(echo "$ALL_PDBS" | wc -l)

    # Find which have MPNN seqs at T=0.1
    N_MPNN=$(ls "${MPNN_DIR}/seqs"/binder_*.fa 2>/dev/null | wc -l)

    # Find which have MPNN seqs at T=0.3
    N_MPNN_T03=$(ls "${MPNN_T03_DIR}/seqs"/binder_*.fa 2>/dev/null | wc -l)

    echo "[$(date +%H:%M)] Backbones: ${N_TOTAL}, MPNN T=0.1: ${N_MPNN}, MPNN T=0.3: ${N_MPNN_T03}"

    # Find new backbones without MPNN sequences
    NEW_PDBS=""
    for pdb in $ALL_PDBS; do
        name=$(basename "${pdb%.pdb}")
        if [ ! -f "${MPNN_DIR}/seqs/${name}.fa" ]; then
            NEW_PDBS="${NEW_PDBS} ${pdb}"
        fi
    done

    if [ -n "$NEW_PDBS" ]; then
        N_NEW=$(echo "$NEW_PDBS" | wc -w)
        echo "[$(date +%H:%M)] Running MPNN on ${N_NEW} new backbones..."

        # Build glob pattern for new backbones
        # Just run MPNN on all backbones (parse_multiple_chains handles duplicates
        # but we need to be careful not to overwrite existing seqs)
        # Use --pattern for specific new ones

        /venv/main/bin/python3 "${SCRIPTS_DIR}/03_run_proteinmpnn.py" \
            --binder_dir "${RFDIFF_DIR}" \
            --output_dir "${MPNN_DIR}" \
            --pattern "binder_*.pdb" \
            --num_seqs 8 \
            --temperature 0.1

        echo "[$(date +%H:%M)] MPNN T=0.1 done. Updating all_designs.json..."
    fi

    # Same for T=0.3
    NEW_PDBS_T03=""
    for pdb in $ALL_PDBS; do
        name=$(basename "${pdb%.pdb}")
        if [ ! -f "${MPNN_T03_DIR}/seqs/${name}.fa" ]; then
            NEW_PDBS_T03="${NEW_PDBS_T03} ${pdb}"
        fi
    done

    if [ -n "$NEW_PDBS_T03" ]; then
        N_NEW=$(echo "$NEW_PDBS_T03" | wc -w)
        echo "[$(date +%H:%M)] Running MPNN T=0.3 on ${N_NEW} new backbones..."

        /venv/main/bin/python3 "${SCRIPTS_DIR}/03_run_proteinmpnn.py" \
            --binder_dir "${RFDIFF_DIR}" \
            --output_dir "${MPNN_T03_DIR}" \
            --pattern "binder_*.pdb" \
            --num_seqs 16 \
            --temperature 0.3

        echo "[$(date +%H:%M)] MPNN T=0.3 done."
    fi

    # Sleep for 5 minutes before checking again
    sleep 300
done
