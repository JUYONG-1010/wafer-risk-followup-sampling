# Final Public Entry Points

These scripts are the public-facing entry points for quick reproduction and
demo runs.

The original numbered scripts in `scripts/01_...` through `scripts/79_...` are
preserved as experiment history. Use this folder when you want a cleaner command
surface for GitHub review.

## Recommended Quick Demo

```bash
python scripts/final/run_final_demo.py --max-test-wafers 100
```

This trains a lightweight RandomForest point-risk ranker from the processed
patterned WM-811K subset, compares it against the geometry-only `coverage32`
baseline, prints a compact result table, and writes:

```text
reports/final_demo/final_demo_summary.md
outputs/figures/final_demo/final_demo_hit_rate.png
```

The demo is intentionally small. It is not the headline 3-seed robustness run.
The headline result reported in the README comes from the repeated wafer-level
split evaluation.

## Other Wrappers

```bash
python scripts/final/run_topk_curve.py
python scripts/final/run_cnn_smoke_test.py
python scripts/final/run_repeated_split_note.py
```

The full repeated-split CNN robustness run was executed in Colab. Local
smoke-test commands are provided for reproducibility checks, but the full run
can be slow without GPU access.
