# Experiment History

This folder contains the chronological experiment scripts used while developing
the wafer follow-up sampling project.

The numbering is intentionally preserved so the project history remains
traceable:

```text
01-07   data extraction, EDA, and early sampling outputs
08-17   baseline sampling and domain/random/radial analyses
18-34   adaptive sampling and policy exploration
35-40   first point-ranking model experiments
41-58   morphology-aware and risk-map policy experiments
59-67   final non-CNN deliverable and adequacy checks
68-73   sparse CNN training, comparison, and visual examples
74-79   low-evidence, Top-K, ensemble, Scratch, and robustness runs
```

For normal GitHub review or quick reproduction, start from:

```text
scripts/final/
```

The scripts in this folder are useful for auditability and deeper experimental
reproduction, but they are not intended to be the first entry point.
