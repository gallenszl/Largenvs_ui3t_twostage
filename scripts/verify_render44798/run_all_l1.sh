#!/bin/bash
# Run all L1 (per-patch) sanity checks in sequence.
# Usage:  bash scripts/verify_render44798/run_all_l1.sh
set -e
cd "$(dirname "$0")/../.."   # back to repo root

PY=${PY:-python}
SCRIPTS=scripts/verify_render44798

echo "=========================================="
echo "L1.P1: per-frame fov"
echo "=========================================="
$PY $SCRIPTS/verify_l1_p1_fov.py

echo
echo "=========================================="
echo "L1.P5a: tar prefix auto-detect"
echo "=========================================="
$PY $SCRIPTS/verify_l1_p5_tar.py

echo
echo "=========================================="
echo "L1.P4: alpha_mask emission + loss path"
echo "=========================================="
$PY $SCRIPTS/verify_l1_p4_alpha.py

echo
echo "=========================================="
echo "L1.P2: roll augment"
echo "=========================================="
$PY $SCRIPTS/verify_l1_p2_roll.py

echo
echo "=========================================="
echo "L3: statistics over 100 batches"
echo "=========================================="
$PY $SCRIPTS/verify_l3_stats.py

echo
echo "=========================================="
echo "ALL L1 + L3 PASSED."
echo "For L4.5 viser viz, run:"
echo "  $PY $SCRIPTS/verify_l4p5_viser.py"
echo "and forward port 8126."
echo "=========================================="
