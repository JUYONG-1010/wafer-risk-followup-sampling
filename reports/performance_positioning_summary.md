# Performance and Positioning Summary

## Frozen Project Goal

```text
Build a limited-budget follow-up sampling recommendation system from first-pass
sparse wafer observations.

Input:
- first-pass sparse wafer observation

Output:
- defect risk score for each unmeasured die
- optional 2D risk map
- Top-K follow-up die recommendation under a fixed sampling budget

Offline evaluation:
- use dense WM-811K wafer maps only as hidden reference labels
- compare against geometry-only coverage baselines
```

## Main Internal Baseline

The primary baseline is `coverage32`.

```text
coverage32:
- geometry-only representative follow-up sampling
- selects 32 unmeasured valid dies by spatial coverage
- does not use defect risk, CNN, pattern prediction, or hidden dense labels
```

This is the fair internal baseline because it answers:

```text
If we only spread the 32 follow-up points evenly, how many real defects do we find?
```

The proposed system answers:

```text
If we use the first-pass sparse observation to rank unmeasured dies by defect risk,
how many real defects do we find within the same 32-point follow-up budget?
```

## Best Current Policy

Current shortlisted final policy:

```text
ensemble_raw_w0.30:
score = 0.3 * nonCNN_probability + 0.7 * CNN_probability
```

Interpretation:

```text
non-CNN point-risk model:
- stable direct die-level ranking from tabular/geometry/risk features

sparse CNN:
- 2D spatial risk map from first-pass sparse wafer observation

ensemble:
- combines direct ranking stability and CNN spatial morphology awareness
```

## Headline Result vs coverage32

Held-out test split, 500 wafers per first-pass density, Top-32 follow-up budget.

| First-pass density | coverage32 precision@32 | final ensemble precision@32 | Relative precision gain | coverage32 hits / 32 | final hits / 32 | Extra defects found |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 22.1% | 68.6% | +211.0% | 7.06 | 21.95 | +14.89 |
| 3% | 19.9% | 70.3% | +252.3% | 6.38 | 22.48 | +16.10 |
| 5% | 21.8% | 71.5% | +227.5% | 6.99 | 22.89 | +15.90 |
| 10% | 17.0% | 73.6% | +332.5% | 5.45 | 23.56 | +18.11 |

Average across 1%, 3%, 5%, 10% first-pass density:

```text
coverage32 precision@32: 20.2%
final ensemble precision@32: 71.0%
relative precision gain: +255.8%

coverage32 hits/32: 6.47
final ensemble hits/32: 22.72
extra true defects found per wafer: +16.25 out of 32 follow-up points
```

Plain-language summary:

```text
With the same 32 follow-up measurements, the current final policy finds about
22-24 true defective dies on average, while geometry-only coverage32 finds about
5-7. That is roughly 3.1x to 4.3x more defect-rich follow-up sampling depending
on first-pass density.
```

## Defect Coverage Result

Defect coverage means:

```text
Among all hidden true defective dies on the wafer, what fraction did the selected
follow-up points hit?
```

| First-pass density | coverage32 defect coverage | final ensemble defect coverage | Relative gain |
|---:|---:|---:|---:|
| 1% | 7.0% | 14.4% | +104.4% |
| 3% | 9.5% | 17.7% | +87.1% |
| 5% | 11.9% | 20.0% | +68.9% |
| 10% | 17.1% | 26.5% | +54.8% |

Average:

```text
coverage32 defect coverage: 11.4%
final ensemble defect coverage: 19.6%
relative gain: +78.8%
```

Interpretation:

```text
precision@32 is the main operating metric because the actual follow-up action is
limited to 32 dies.

Defect coverage is secondary because a wafer may contain far more than 32
defective dies, so no Top-32 policy can cover the entire defect population.
```

## Top-K Budget Curve

The improvement is not only a K=32 artifact.

Across K = 16, 32, 64, 128, ML/CNN risk ranking consistently beats coverageK.

Representative precision@K relative gain over geometry-only coverage:

```text
K=16:  +185% to +324%
K=32:  +206% to +329%
K=64:  +183% to +294%
K=128: +168% to +224%
```

Important interpretation:

```text
As K increases, total defects found naturally increases for every policy.
The important result is precision@K: among the K selected follow-up dies, the
ML/CNN policies keep a much higher fraction of actual defects than coverageK.
```

## 2D Risk Map Quality

Matched held-out split, 500 wafers per density.

| First-pass density | sparse CNN ROC-AUC | sparse CNN AP | sparse CNN top-10% IoU |
|---:|---:|---:|---:|
| 1% | 0.714 | 0.484 | 0.282 |
| 3% | 0.729 | 0.499 | 0.288 |
| 5% | 0.737 | 0.511 | 0.295 |
| 10% | 0.744 | 0.523 | 0.300 |

Interpretation:

```text
The CNN risk map is not perfect, but it is meaningfully better than random
ranking and can produce a wafer-level 2D risk heatmap from sparse first-pass
observations.
```

## External Literature Positioning

### Sampling literature

The review paper in `papers/asi-09-00001-v3.pdf` frames semiconductor sampling
as a progression:

```text
static sampling -> adaptive sampling -> dynamic/risk-driven sampling
```

Our project fits this direction because it does not only classify wafer maps.
It makes die-level follow-up sampling decisions under a fixed budget.

### WM-811K classification/localization literature

The paper in `papers/s41598-026-52885-x_reference.pdf` reports:

```text
Swin Transformer defect-pattern classification:
- accuracy: 94.83%
- macro F1: 92.44%
- macro ROC-AUC: 99.71%

Grad-CAM localization:
- baseline Top-10% IoU: 0.129
- radial suppression Top-10% IoU: 0.155
- oracle upper bound: 0.278
```

This is useful context, but it is not a direct baseline for our project.

Reason:

```text
Their main model sees the wafer image for defect-pattern classification and
then evaluates Grad-CAM explanation localization.

Our model starts from sparse first-pass observations and recommends where to
measure next under a limited follow-up budget.
```

Therefore, the honest external positioning is:

```text
Existing WM-811K work usually reports high classification accuracy.
This project targets a different manufacturing decision problem:
where to spend the next limited follow-up measurements.
```

Do not claim:

```text
Our model is X% better than Swin Transformer classification.
```

That would be invalid because the tasks and metrics are different.

Reasonable claim:

```text
Compared with a geometry-only follow-up sampling baseline, the proposed
first-pass risk-ranking system improves Top-32 follow-up precision from about
20.2% to 71.0% on the held-out split, a +255.8% relative gain, while finding
about 16.25 additional true defective dies per wafer under the same 32-point
budget.
```

## Limitations

```text
1. Current headline numbers are from a held-out split.
   Repeated split robustness is currently running on Colab GPU.

2. WM-811K is used as an offline dense reference.
   This is not yet validated on a real fab's live first-pass/follow-up process.

3. Scratch remains a weak pattern.
   Fixed Scratch guard experiments did not improve the final policy.

4. Defect-ratio error is not the primary objective.
   Defect-rich ranking intentionally oversamples high-risk regions, so it can
   bias defect-ratio estimation while still being valuable for defect discovery.
```

## Current Portfolio Claim

```text
Built a first-pass wafer follow-up recommendation system that converts sparse
wafer observations into die-level defect risk scores, 2D risk maps, and Top-K
follow-up sampling recommendations.

On held-out WM-811K dense-reference evaluation, the final ensemble policy
improved Top-32 follow-up precision from 20.2% to 71.0% versus a geometry-only
coverage baseline, finding about 16 additional defective dies per wafer under
the same 32-point inspection budget.
```
