#!/bin/bash
# Run the full 4-mode ablation study.
# Each mode runs 200 steps with seed 42 for reproducibility.
# Results are saved to separate directories under results/.
#
# Requirements: Apple Silicon Mac with mlx-lm installed.
# Runtime: ~20 minutes per mode, ~80 minutes total.

set -e

echo "=== Dream Layer v0.1 Ablation Study ==="
echo ""

for mode in random template structure feedback; do
    echo "--- Running mode: $mode ---"
    python src/concept_learner.py --mode $mode --steps 200 --seed 42 --reset
    echo ""
done

echo "=== All modes complete ==="
echo ""
echo "Generate figures with:"
echo "  python paper/figures/generate_figures.py"
