# Direct ADEPT–SVoE Surprisal Benchmark

## Question

How do scalar and vectorized implementations affect the execution time and
temporary allocation of ADEPT and direct empirical SVoE?

This benchmark compares the two calculations at the same point in the
pipeline. Both receive the same particle discrepancy scores and the same
particle weights for each video frame. Physics simulation, perception,
matching, file input, and reference preparation are outside the timed region.

## ADEPT calculation

For particle discrepancy scores `s_i` and normalized particle weights `w_i`,
ADEPT calculates

```text
ADEPT = -ln(sum_i(w_i * exp(-s_i)))
```

The implementation in `adept_surprisal_value()` reproduces the loop in
`models/particle_filter.py`.

## Vectorized ADEPT calculation

The optimized candidate preserves ADEPT's formula while replacing 128 scalar
NumPy exponential calls with one vector operation and one dot product:

```text
scores = contiguous float64 array
weights = contiguous float64 array
ADEPT_vectorized = -ln(dot(weights, exp(-scores)))
```

Array-conversion cost remains inside the timed function. The benchmark also
records the maximum absolute difference between the released and vectorized
ADEPT calculations across all evaluated frames.

## Drop-in SVoE calculation

The ordinary reference data provide a sorted collection of particle-level
discrepancy scores. Each current particle score is converted to an add-one
empirical upper-tail probability:

```text
p_i = (number of reference scores >= s_i + 1) /
      (number of reference scores + 1)
```

The existing particle weights combine those probabilities. The resulting
mixture probability is expressed as Shannon information:

```text
SVoE_bits = -log2(sum_i(w_i * p_i))
```

This produces one frame-level value and occupies the same aggregation point
as ADEPT's core calculation. It does not use ADEPT's completed frame-level
surprisal as its input.

## Vectorized SVoE calculation

The optimized candidate preserves the same empirical calibration. It replaces
128 Python-level binary searches and scalar weighted additions with a single
NumPy `searchsorted` call, vector arithmetic, and a dot product:

```text
indices = searchsorted(sorted_reference, scores, side="left")
tail_counts = reference_count - indices
tail_probabilities = (tail_counts + 1) / (reference_count + 1)
SVoE_vectorized = -log2(dot(weights, tail_probabilities))
```

The sorted reference array is prepared once before timing. Converting the
current particle scores and weights into arrays remains inside the timed
function, matching the vectorized ADEPT timing boundary. The benchmark records
the maximum absolute difference between scalar and vectorized SVoE results in
bits.

## Available-data limitation

ADEPT's saved result JSON contains particle discrepancy scores but not the
historical particle weights used during the original run. The benchmark
therefore supplies normalized equal weights to both methods. This preserves
an identical input comparison, but an instrumented ADEPT run will eventually
be required to repeat the benchmark with the original evolving weights.

The supplied configuration uses the two currently reproduced videos as the
reference corpus so that the computational experiment is immediately
repeatable. Final scientific evaluation requires a designated collection of
ordinary control videos that is separate from the evaluated events.

## Timing protocol

The benchmark uses the existing timing harness:

- 20 untimed warmup passes;
- 1,000 complete passes per timing repeat;
- 7 timing repeats;
- median elapsed time reported per frame;
- coefficient of variation reported across repeats;
- file loading and reference sorting excluded.

`peak_incremental_bytes` is the incremental peak Python allocation observed
by `tracemalloc`; it is not total process or native-library memory.

## Run

From the repository root inside the validated ADEPT environment:

```bash
python -m svoe_calibration.surprisal_benchmark \
  --config experiments/svoe_surprisal_benchmark.yaml
```

The machine-readable result is written to
`output/svoe_calibration/svoe_dual_optimization_benchmark.json`.
