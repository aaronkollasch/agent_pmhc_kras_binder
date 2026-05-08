#!/bin/bash
# Run round-2 pipeline on each seed sequentially as partial diffusion completes.
# Each seed waits for its 200 PDBs, then runs geo→MPNN→spec→AF2.
#
# Usage:  bash 14_r2_run_pipeline.sh [--skip_af2] 2>&1 | tee /tmp/r2_pipeline.log

SEEDS=(b7_t03_s7 b17_s1 b93_s6 b7_t03_s2)
RFDIFF_R2="/workspace/pmhc_design/designs/rfdiffusion_r2"
SCRIPTS="/workspace/pmhc_design/scripts"
EXTRA_ARGS="$@"

for SEED in "${SEEDS[@]}"; do
  BINDER_DIR="${RFDIFF_R2}/${SEED}"
  echo ""
  echo "$(date): ===== Waiting for ${SEED} partial diffusion to complete ====="

  while true; do
    N=$(ls "${BINDER_DIR}"/binder_*.pdb 2>/dev/null | wc -l)
    echo "$(date):   ${SEED}: ${N}/200 PDBs done (~$(echo "scale=0; $N * 100 / 200" | bc)%)"
    if [ "$N" -ge 200 ]; then
      echo "$(date):   ${SEED}: COMPLETE"
      break
    fi
    sleep 120
  done

  echo "$(date): ===== Running R2 pipeline for ${SEED} ====="
  python3 "${SCRIPTS}/12_r2_pipeline.py" --seed "${SEED}" ${EXTRA_ARGS}
  echo "$(date): ===== Pipeline done for ${SEED} ====="
done

echo ""
echo "$(date): ===== All seeds done — running comparison ====="
python3 "${SCRIPTS}/13_r2_compare.py"
