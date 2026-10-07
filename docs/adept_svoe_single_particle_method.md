# ADEPT Single-Particle SVoE Experiment

## Purpose

This experiment tests whether a single ADEPT physical hypothesis can provide a
useful nonconformity score after SVoE calibration.

The released ADEPT configuration maintains 128 particles. Each particle is one
possible scene history and current physical state. ADEPT advances every
particle, compares it with the observed object state, and combines the particle
likelihoods into one frame score.

The experimental variant retains ADEPT's derendered observations, PyBullet
dynamics, object matching, and particle-level mismatch calculation. It changes
only the number of maintained hypotheses:

```text
Released ADEPT: 128 hypotheses -> weighted aggregate frame score
Experimental:     1 hypothesis  -> mismatch residual -> SVoE calibration
```

With one particle, ADEPT's aggregate score reduces algebraically to that
particle's mismatch score. SVoE can therefore calibrate the mismatch directly
against mismatch scores from ordinary control videos.

The potential speed advantage comes from removing 127 particle simulations.
SVoE provides the calibrated probability and bit scale; it does not itself
create the speedup.

## Branch and installation

This overlay depends on the completed `svoe-calibration` branch. Begin with a
clean working tree, create the experimental branch, and extract the overlay at
the repository root:

```bash
git switch svoe-calibration
git status --short
git switch -c svoe-single-particle
unzip /path/to/adept_svoe_single_particle_overlay.zip -d .
```

The archive contains repository-relative paths and adds new files only.

## Architectural isolation

No ADEPT core file is modified. The experiment imports ADEPT's existing
`FilterUpdater` and supplies a separate configuration.

The runner deliberately omits image plotting and video encoding. Its timer
includes only `FilterUpdater.run()`, so particle-count comparisons measure
physical reasoning rather than presentation work.

For a single particle, the runner selects ADEPT's direct execution path instead
of creating a multiprocessing pool. This represents how a native one-hypothesis
implementation would operate and avoids measuring process-pool startup as part
of the residual calculation.

## Configuration

The initial configuration is:

```text
experiments/svoe_single_particle.yaml
```

It runs five independent replicates of the two available control cases with one
particle. ADEPT seeds particle evolution from operating-system randomness, so
the replicates measure the variability introduced by committing to one
stochastic history.

Each run receives a separate directory:

```text
output/svoe_single_particle/
  particles_001/
    replicate_001/
      <case_name>/
        result.json
        timing.json
```

The root `manifest.json` records every run and its output paths.

## Running on Quartz

Activate the required module:

```bash
module load apptainer/1.3.6
```

From `/N/slate/jburgoon/ADEPT-Model-Release`, run:

```bash
apptainer exec --cleanenv \
  --bind /N/slate/jburgoon:/N/slate/jburgoon \
  --bind /N/slate/jburgoon/ADEPT-Model-Release:/root/adept-model \
  --pwd /root/adept-model \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 \
  /N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif \
  /N/u/jburgoon/Quartz/.conda/envs/adept-model/bin/python \
  -m svoe_calibration.particle_experiment \
  --config experiments/svoe_single_particle.yaml
```

The physical-reasoning stage is CPU-based; this command does not request NVIDIA
GPU passthrough.

### Resuming an interrupted experiment

With `overwrite_existing: false`, rerunning the same command validates and
skips each case that already has both `result.json` and `timing.json`. Cases
without final output are run normally. After all cases are available, the
runner writes a complete `manifest.json` containing both reused and newly
completed runs.

If only one of the two files exists, the runner stops and reports an incomplete
output pair rather than silently accepting or overwriting ambiguous data.

## Initial evaluation

The first run answers two limited questions:

1. How variable are single-particle residual traces across repeated control runs?
2. How much physical-reasoning time is removed relative to the 128-particle implementation?

It cannot establish violation sensitivity because the current sample contains
only control cases. Matched violation cases are required before comparing
control-versus-violation separation.

## Later comparison

The single-particle results were not stable enough to replace the released
filter. Subsequent exploratory runs examined particle counts of 16, 24, 28,
30, 32, 48, and 128. Every particle-count variant is calibrated against control
scores produced by that same variant.

The reusable analysis procedure and its statistical definitions are documented
in:

```text
docs/adept_svoe_particle_analysis.md
```

The final comparison should report:

- control-versus-violation SVoE separation;
- relative accuracy on matched pairs;
- variation across stochastic replicates;
- physical-reasoning time per frame;
- memory consumption measured by the scheduler or an external process monitor.

The simplified model is successful only if its computational savings preserve
useful violation discrimination. SVoE calibration cannot recover alternative
physical histories discarded by reducing the particle population.
