# Risk-Aware Follow-Up Sampling for Sparse Wafer Defect Observation

**A technical report on limited-budget wafer follow-up sampling**

## Abstract

Wafer defect map projects usually focus on classifying the overall defect
pattern of a fully observed wafer. This project studies a different inspection
problem: after a sparse first-pass observation, where should the next limited
follow-up measurements be placed to find defective dies?

Using WM-811K / LSWMD patterned wafer maps as offline dense reference labels, the
system simulates first-pass observations at 1%, 3%, 5%, and 10% of valid dies.
At recommendation time, the model only sees the sparse first-pass observation,
wafer geometry, and candidate die coordinates. It does not see the hidden dense
wafer map, the true failure type, or the true defect ratio.

The final operating policy is a Random Forest point-risk ranker that assigns a
defect probability to every unmeasured valid die and recommends the Top-32 dies
for follow-up inspection. On a 3-seed repeated wafer-level split, the model
improves Top-32 hit rate from 20.1% for a geometry-only baseline to 69.1%,
finding 22.10 true defective dies out of 32 on average versus 6.43 for the
baseline.

![Sparse first-pass input, CNN risk map, and Top-32 follow-up overlay](assets/example_donut_risk_map.jpg)

## 1. Motivation

In semiconductor inspection, measuring every die or every suspicious location can
be expensive. A practical inspection workflow often needs to decide where to
spend a limited second-pass or follow-up measurement budget.

Most public WM-811K projects answer:

```text
What defect pattern class does this wafer belong to?
```

This project answers:

```text
Given only a sparse first-pass observation, which unmeasured dies should be
probed next under a fixed follow-up budget?
```

The output is therefore not only a class label. The system produces:

- a defect risk score for each unmeasured die,
- a wafer-level 2D risk map,
- a Top-K follow-up recommendation list,
- offline hit/miss evaluation against hidden dense reference labels.

## 2. Dataset

The project uses WM-811K / LSWMD wafer maps. Each wafer map is a 2D grid with
die-level states:

```text
0 = outside wafer / invalid die location
1 = normal die
2 = defective die
```

The public dataset contains 811,457 total rows. Among them, 172,950 rows are
labeled and 25,519 rows are patterned defect wafers excluding `none`.

Patterned failure type counts:

| Failure type | Count |
|---|---:|
| Edge-Ring | 9,680 |
| Edge-Loc | 5,189 |
| Center | 4,294 |
| Loc | 3,593 |
| Scratch | 1,193 |
| Random | 866 |
| Donut | 555 |
| Near-full | 149 |

The raw dataset is not included in the repository. It is used locally as
offline reference data only.

## 3. Problem Formulation

For each wafer, the dense map is treated as hidden ground truth. The experiment
simulates the following process:

1. Sample a sparse first-pass observation, such as 1%, 3%, 5%, or 10% of valid
   dies.
2. Hide all remaining die labels from the model.
3. Score every unmeasured valid die by predicted defect risk.
4. Recommend the Top-K follow-up dies.
5. Reveal the dense wafer map only after selection to evaluate how many
   recommended dies were truly defective.

The main operating setting is:

```text
K = 32 follow-up dies
```

The project also evaluates K = 16, 32, 64, and 128 to check whether the result is
only a Top-32 artifact.

## 4. Information Boundary

The most important design rule is the information boundary. Dense wafer maps are
used only for offline evaluation. They are not used when selecting follow-up
points.

Allowed at recommendation time:

```text
- sparse first-pass observation
- wafer geometry
- candidate die coordinates
- distance to sampled points
- distance to observed first-pass defect hits
- derived coordinate and regional features
```

Forbidden at recommendation time:

```text
- hidden dense defect map
- hidden defect coordinates
- true total defect count
- true defect ratio
- true failureType label
- future wafer or lot information
```

The train/validation/test split is wafer-level, not die-level. This avoids
placing candidate dies from the same wafer in both training and test evaluation.

## 5. Baseline

The primary baseline is a geometry-only follow-up sampler called `coverage32` in
the code.

It selects 32 unmeasured valid dies to improve spatial coverage across the wafer.
It does not use defect prediction, CNN outputs, pattern labels, hidden dense
labels, or first-pass defect-risk ranking.

This baseline is useful because it represents a conservative inspection
strategy:

```text
If we do not predict defects at all and only spread points across the wafer,
how many actual defects do we find?
```

The proposed model answers the paired question:

```text
If we use sparse first-pass evidence to rank unmeasured dies by defect risk,
how many actual defects do we find with the same 32-point budget?
```

## 6. Proposed Method

### 6.1 Point-Risk Ranking Model

The final operating policy is a Random Forest point-risk ranker.

For one wafer, it works as follows:

1. List every unmeasured valid die as a candidate.
2. Build a feature vector for each candidate.
3. Predict the probability that the candidate is defective.
4. Sort candidates by predicted probability.
5. Recommend the Top-32 candidates.

The scoring rule is:

```text
risk_score(candidate) =
    RandomForestClassifier.predict_proba(candidate_features)[defect_class]
```

Feature groups:

| Feature group | Examples | Purpose |
|---|---|---|
| Wafer geometry | map height, width, valid die count | Captures wafer shape and valid region |
| First-pass evidence | sampled count, hit count, no-hit flag | Captures how much defect evidence was observed |
| Hit location summary | center/mid/edge hit counts | Captures where sparse defects appeared |
| Candidate position | normalized y/x, radius, angle | Captures center, edge, ring, quadrant effects |
| Candidate zone | radial zone, quadrant flags | Encodes coarse spatial regions |
| Distance features | nearest sampled die, nearest hit, hit centroid | Captures proximity to first-pass evidence |
| Shared-region flags | same quadrant or radial zone as first-pass hit | Captures simple pattern continuation |

Final repeated-split model settings:

```text
Model: RandomForestClassifier
Estimators: 100
Max depth: 14
Min samples per leaf: 10
Class weight: balanced
Candidate sampling per wafer:
  - max defect candidates: 80
  - max normal candidates: 120
```

Random Forest was selected because the candidate features are tabular and
spatial, not only image-like. Tree ensembles handle nonlinear feature
interactions well, are stable without GPU training, and provide direct candidate
probability ranking.

### 6.2 Sparse CNN Risk Map

A sparse CNN model was also trained to produce a 2D wafer risk map from sparse
first-pass observation channels.

Its role is different from the final Top-32 policy:

```text
RandomForest: primary Top-32 recommendation policy
Sparse CNN: visual 2D risk-map model and ablation branch
```

CNN architecture:

```text
Input channels: 7
Hidden channels: 32

1. Conv2d(7 -> 32, kernel=7) + GroupNorm + ReLU
2. Conv2d(32 -> 32, kernel=3, dilation=2) + GroupNorm + ReLU
3. Conv2d(32 -> 32, kernel=3, dilation=4) + GroupNorm + ReLU
4. Conv2d(32 -> 32, kernel=3) + GroupNorm + ReLU
5. Conv2d(32 -> 1, kernel=1)
```

Training setup:

```text
Loss: binary cross entropy with positive-class weighting
Optimizer: Adam-style PyTorch training loop
Learning rate: 1e-3
Weight decay: 1e-4
Max epochs in robustness run: 8
Early stopping patience: 3
AMP mixed precision: enabled on Colab GPU
```

The CNN is intentionally small. Wafer maps are structured 2D grids, and dilated
convolutions expand the receptive field without requiring a deep architecture.
Coordinate and radius channels help the model understand wafer geometry.

![CNN and non-CNN risk-map quality metrics](assets/risk_map_metrics.png)

### 6.3 Ensemble Ablation

The project also tested a CNN + Random Forest ensemble:

```text
score = w * RandomForest_probability + (1 - w) * CNN_probability
```

Validation tested:

```text
w = 0.0, 0.1, 0.2, ..., 1.0
```

The best validation weight was:

```text
w = 0.3
score = 0.3 * RandomForest_probability + 0.7 * CNN_probability
```

However, the repeated-split Top-32 comparison selected the Random Forest-only
ranker as the final policy because it was slightly more stable for the direct
follow-up recommendation task.

![Ensemble weight validation sweep](assets/ensemble_weight_validation.png)

## 7. Experimental Setup

The final headline result uses repeated wafer-level splits:

| Item | Setting |
|---|---|
| Dataset subset | WM-811K patterned defect wafers |
| Split type | wafer-level train/validation/test |
| Seeds | 42, 101, 202 |
| First-pass densities | 1%, 3%, 5%, 10% |
| Test wafers | 500 wafers per density per split |
| Follow-up budget | K = 32 |
| Primary metric | hit rate among Top-32 recommendations |
| Secondary metrics | true defects found, defect coverage, severe miss rate |

The primary metric is Top-K hit rate:

```text
Top-K hit rate = actual defective dies among K recommendations / K
```

For K = 32, this is the same as precision@32. It answers:

```text
Among the 32 recommended follow-up dies, how many were actually defective?
```

## 8. Results

### 8.1 Main Top-32 Result

On the 3-seed repeated wafer-level split:

| Follow-up strategy | Avg. true defects found out of 32 | Hit rate among 32 recommendations |
|---|---:|---:|
| Geometry-only baseline | 6.43 | 20.1% |
| Final Random Forest risk model | 22.10 | 69.1% |

Plain-language interpretation:

```text
With the same 32 follow-up measurements, the final model finds about 15-16 more
defective dies per wafer than the geometry-only baseline.
```

This is a +244.0% relative improvement in Top-32 hit rate over the geometry-only
baseline.

![Top-32 hit rate by first-pass density](assets/result_top32_hit_rate.png)

### 8.2 Result by First-Pass Density

| First-pass observed dies | Geometry-only hit rate | Final RF model hit rate | Relative gain | Extra defects found out of 32 |
|---:|---:|---:|---:|---:|
| 1% | 22.2% | 68.2% | +207.0% | +14.71 |
| 3% | 20.0% | 68.8% | +243.3% | +15.61 |
| 5% | 21.4% | 69.4% | +224.6% | +15.37 |
| 10% | 16.7% | 69.9% | +318.7% | +17.02 |

The model remains effective even when the first-pass observation is very sparse.

### 8.3 Budget Sensitivity

The project also checks K = 16, 32, 64, and 128. As K increases, every method
naturally finds more total defects. The important comparison is whether the
selected K points contain a higher fraction of true defects than the
geometry-only strategy.

The risk-ranking policies remain above the geometry baseline across the budget
range.

![Top-K hit-rate curve](assets/topk_hit_rate_curve.png)

![Top-K defect coverage curve](assets/topk_defect_coverage_curve.png)

### 8.4 Visual Example

The visualization below shows an offline validation example. The model sees only
the sparse first-pass input, produces a risk map, and recommends follow-up
locations. The dense map is revealed only afterward to mark hit/miss.

![Follow-up hit/miss visualization](assets/example_center_hit_miss.jpg)

## 9. Negative Experiments

The project also tested several refinements that were not selected as the final
policy:

```text
- low-evidence gate refinement
- Scratch-specific guard points
- oracle Scratch routing
- predicted Scratch routing
```

These did not reliably improve the final operating result. In particular,
fixed Scratch guard points often displaced better-ranked risk candidates.

![Scratch guard experiment](assets/scratch_guard_result.png)

This is an important result because it prevents the project from becoming a
collection of hand-tuned rules. The final policy remains a general point-risk
ranker, while Scratch-like patterns are documented as a limitation.

## 10. Discussion

### Why the Random Forest beat the CNN for Top-32

The CNN sees the wafer as a 2D image and is useful for making a risk heatmap.
However, the Top-32 task is an extreme ranking problem: the model must order a
small number of candidate dies at the very top of the list.

The Random Forest directly ranks candidates using first-pass hit evidence,
coordinates, zones, and distance features. For this specific Top-32 objective,
that direct point-risk formulation was slightly more stable than the CNN-only
or CNN-dominant ensemble result.

### Why defect ratio error is not the main metric

A policy that intentionally targets defect-rich regions can overestimate the
wafer's global defect ratio. That is a real bias if the goal is unbiased
estimation of the whole wafer.

This project's main objective is different:

```text
Find more actual defective dies under a limited follow-up inspection budget.
```

Therefore, Top-K hit rate and true defects found are the primary metrics.
Defect-ratio bias is tracked as a warning, not treated as the main optimization
target.

## 11. Limitations

1. WM-811K is an offline dense-reference dataset, not a real fab follow-up
   inspection log.
2. The model does not use process context such as tool ID, chamber ID, recipe,
   lot history, FDC traces, or metrology history.
3. Scratch and Loc-like patterns remain harder than more spatially concentrated
   patterns such as Center, Edge-Ring, and Donut.
4. A Top-32 policy cannot cover all defects when a wafer contains far more than
   32 defective dies.
5. The result should not be interpreted as a production yield-improvement claim.
6. The public repository does not include the raw WM-811K pickle or trained
   model checkpoints.

## 12. Conclusion

This project reframes wafer defect map modeling as a limited-budget follow-up
sampling problem. Instead of only predicting a wafer-level defect class, it
scores every unmeasured die and recommends where to inspect next.

The final Random Forest point-risk ranker improves Top-32 hit rate from 20.1%
to 69.1% over a geometry-only baseline on repeated wafer-level splits. In
practical terms, it finds about 22 true defective dies out of 32 follow-up
measurements, compared with about 6 for the geometry-only baseline.

The key contribution is not claiming production deployment. The contribution is
a clear decision-support formulation:

```text
sparse first-pass observation -> die-level risk ranking -> limited-budget
follow-up recommendation -> offline dense-map validation
```

## References

- WM-811K / LSWMD wafer map dataset.
- Public WM-811K classification repositories and dataset explanations reviewed
  during project positioning.
- Internal project experiments in `experiments/01` through `experiments/79`.
