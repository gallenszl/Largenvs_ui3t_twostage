#!/bin/bash
# Launch 4 parallel probe jobs, one per baseline checkpoint.
# Output goes to a shared CSV per job (later merged by tools/plot_probe_results.py).
#
# Usage: bash scripts/launch_probe_parallel.sh
#
# Checkpoints probed: 2k, 6k, 10k, 15k (4 representative points in baseline trajectory).

set -euo pipefail

BASELINE_DIR=/home/z50057756/code/RnG/experiments/checkpoints/FA3_tar_company
OUT_DIR=/home/z50057756/code/RnG_feature_allignment/experiments/evaluation/baseline_probe
mkdir -p "${OUT_DIR}"

declare -A CKPTS=(
    [step2k]=ckpt_0000000000002000.pt
    [step6k]=ckpt_0000000000006000.pt
    [step10k]=ckpt_0000000000010000.pt
    [step15k]=ckpt_0000000000015000.pt
)

for LABEL in "${!CKPTS[@]}"; do
    CKPT_FILE="${CKPTS[$LABEL]}"
    CKPT_PATH="${BASELINE_DIR}/${CKPT_FILE}"
    OUT_CSV="${OUT_DIR}/probe_${LABEL}.csv"

    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[ERROR] missing ckpt: ${CKPT_PATH}"
        exit 1
    fi

    # Remove existing CSV so this job writes a fresh one
    rm -f "${OUT_CSV}"

    echo "Submitting probe for ${LABEL}  ckpt=${CKPT_FILE}  out=${OUT_CSV}"
    PROBE_CKPT_PATH="${CKPT_PATH}" \
    PROBE_OUT_CSV="${OUT_CSV}" \
    PROBE_LABEL="baseline_${LABEL}" \
    PROBE_LAYERS="3,7,12,17,22" \
    PROBE_MAX_OBJECTS="0" \
    sbatch \
        --job-name="repa-probe-${LABEL}" \
        scripts/baseline_feature_probe.sh
done

echo "Submitted 4 parallel probe jobs. Outputs:"
ls -1 "${OUT_DIR}"/probe_*.csv 2>/dev/null || echo "(will appear after jobs run)"

echo ""
echo "Once all jobs complete, merge with:"
echo "  python3 tools/plot_probe_results.py ${OUT_DIR}"
