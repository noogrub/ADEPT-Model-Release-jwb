# ADEPT–SVoE Calibration Methodology

## Purpose

This experiment applies Shannon Violation of Expectation (SVoE) calibration to the frame-level surprise scores produced by ADEPT.

ADEPT remains unchanged. SVoE is implemented as an independent post-processing layer that reads ADEPT results, calibrates them against ordinary behavior, and expresses their empirical rarity in bits.

## Working branch

The work is isolated on the local Git branch:

```text
svoe-calibration
```

This branch was created from `adept-quartz`. Nothing from this repository is pushed to the upstream ADEPT repository.

## ADEPT environment

The ADEPT Conda environment is activated on Quartz with:

```bash
module load conda/26.3.2
conda activate adept-model
```

The environment uses Python 3.7.12. Full ADEPT execution uses the Apptainer procedure documented in `ADEPT_install_notes.md`.

## Source ADEPT measurement

ADEPT stores its frame-level surprise values in the `all` field of each result JSON file:

```text
output/default_physics/results/<case_name>.json
```

In `models/particle_filter.py`, ADEPT calculates the weighted likelihood across particles and records:

\[
c(x,t)=-\ln\left(\sum_i w_i e^{-q_i(t)}\right)
\]

The saved `all` array contains these values for every evaluated frame. The `sum`, `mean`, and `max` fields summarize the same array.

## Available control cases

The current sample contains two control cases:

| Case | Frames | Minimum score | Maximum score |
|---|---:|---:|---:|
| `human_block_0051_0` | 170 | 0.0 | 3.819148173423241 |
| `human_disappear_0007_3` | 170 | 0.0 | 2.7488627169157103 |

The ADEPT dataset generator uses the suffix `_0` for the block control and `_3` for one of the create/disappear controls. These labels should be verified against the complete generated pair metadata before publication.

For the initial implementation, the two cases provide 340 provisional ordinary frame scores.

## SVoE calibration

Let the ordinary reference scores be:

\[
\mathcal{R}=\{r_1,\ldots,r_n\}
\]

For an ADEPT frame score \(c(x,t)\), count the reference scores that are at least as large:

\[
k_t=\left|\{r_i\in\mathcal{R}:r_i\geq c(x,t)\}\right|
\]

The add-one empirical tail probability is:

\[
\hat p_t=\frac{k_t+1}{n+1}
\]

The calibrated SVoE score is:

\[
S_t=-\log_2 \hat p_t
\]

The add-one calculation prevents a finite reference sample from assigning zero probability. With 340 reference scores, the largest empirically resolvable score is:

\[
S_{\max}=\log_2(341)\approx 8.41\text{ bits}
\]

## Implementation

The independent calibration code is stored in:

```text
svoe_calibration/calibration.py
```

It uses binary search over sorted reference scores. It does not import or modify ADEPT core code.

A known-value smoke test used reference scores `[1, 2, 3]` and an observed score of `4`. The result was:

```text
tail count:       0
tail probability: 0.25
SVoE:             2.0 bits
```

## Interpretation

ADEPT’s raw score measures disagreement between its physical prediction and observation.

SVoE calibration asks how often ordinary ADEPT behavior produces a score at least that large. It converts the model-specific score into an empirical tail probability and a Shannon surprisal value in bits.

Calibration does not retrain ADEPT or alter its predictions.

## Initial calibration run

The initial run calibrated both available control cases against the pooled 340-frame provisional reference set.

```bash
python -m svoe_calibration.run \
  --config experiments/svoe_calibration.yaml
```

The run produced:

```text
output/svoe_calibration/svoe_calibration.json
output/svoe_calibration/svoe_frame_scores.csv
```

| Case | Mean ADEPT score | Maximum ADEPT score | Mean SVoE | Maximum SVoE |
|---|---:|---:|---:|---:|
| `human_block_0051_0` | 0.4800465 | 3.8191482 | 1.3428720 bits | 7.4136279 bits |
| `human_disappear_0007_3` | 0.4231377 | 2.7488627 | 1.3811122 bits | 6.0916998 bits |

The theoretical resolution limit for 340 reference scores is 8.4136279 bits. The largest observed control score received 7.4136279 bits because that score was itself included in the reference set and therefore had a tail count of one.

The first single-pass measurement calibrated 340 target frames in approximately 0.00109 seconds, with 57,784 bytes of peak Python allocation reported by `tracemalloc`. These values confirm that the implementation is lightweight, but they are not treated as formal benchmark results.

The calibrated traces were plotted with:

```bash
python -m svoe_calibration.plot \
  --config experiments/svoe_calibration.yaml
```

The figures are written to:

```text
output/svoe_calibration/figures/
```

## Validation

Four unit tests verify:

- add-one behavior above the reference maximum;
- inclusive treatment of tied scores;
- monotonic increase of SVoE with the source score;
- rejection of an empty reference set.

All four tests passed under Python 3.7.12.

## Current limitations

The current 340-frame reference is suitable for implementation and timing tests, but not for final scientific claims:

- it contains only two videos;
- successive video frames are temporally dependent;
- only two physical scenarios are represented;
- the controls and future test cases must remain properly separated.

The final experiment will use the larger collection of designated ADEPT control videos and will report the construction of the calibration and test sets explicitly.

When the same reference distribution is used, SVoE is a monotonic transformation of ADEPT’s frame score. It adds interpretation and comparability but does not change ADEPT’s ranking of events.
