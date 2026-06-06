#!/usr/bin/env bash
# Minimal end-to-end example for the PSF selection workflow.
# Uses the bundled example metadata; no real image data or EPFL JAR required
# (the psfmodels backend is used automatically when the JAR is absent).
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=${1:-example_outputs}
PY=${PYTHON:-python}

echo "== Backends =="
$PY -m psfselect.cli backends

echo
echo "== 1. Ingest metadata =="
$PY -m psfselect.cli ingest examples/example_metadata.yaml -o "$OUT"

echo
echo "== 2-5. End-to-end (generate -> compare -> rank -> report) =="
# Small grids (nx/nz) keep the example fast. Drop --no-stability for full analysis.
$PY -m psfselect.cli run examples/example_metadata.yaml \
    -o "$OUT" \
    --models gibson_lanni born_wolf richards_wolf vri_gibson_lanni \
    --nx 96 --nz 48

echo
echo "Done. See:"
echo "  $OUT/report.md"
echo "  $OUT/comparison_table.csv"
echo "  $OUT/recommendations.json"
echo "  $OUT/figures/"

echo
echo "== Optional: parameter sweep over NA and depth =="
echo "  $PY -m psfselect.cli generate examples/example_metadata.yaml -o ${OUT}_sweep --sweep configs/sweep.example.yaml --nx 80 --nz 40"
echo "  $PY -m psfselect.cli report ${OUT}_sweep/candidates_manifest.json -o ${OUT}_sweep"
