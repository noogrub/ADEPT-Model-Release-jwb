# Approximate Derenderer, Extended Physics, and Tracking (ADEPT) installation and validation on Indiana University Quartz

Installation completed and validated on 2026-09-14 and 2026-09-15. These notes record the environment used to reproduce the sample ADEPT inference pipeline on the Indiana University (IU) Quartz high-performance computing (HPC) cluster, using the Slate Lustre parallel filesystem, for comparison with Shannon Violation of Expectations (SVoE). They describe the tested installation, including two required source repairs.

Quartz schedules jobs through the Slurm Workload Manager. Apptainer, a container runtime designed for HPC systems, runs an Ubuntu Singularity Image Format (SIF) container. Conda, a software environment and package manager, supplies the Python environment. NVIDIA's CUDA parallel-computing platform and CUDA Deep Neural Network library (cuDNN) supply the historical graphics processing unit (GPU) runtime. Central processing unit (CPU) physics jobs use the same container without GPU passthrough.

## Validated configuration

| Component | Validated value |
|---|---|
| System | Indiana University Quartz; Red Hat Enterprise Linux (RHEL) 8 host environment |
| Storage | Slate Lustre parallel filesystem |
| Slurm account | `r01469` |
| GPU partition | `v100` |
| Tested GPU | NVIDIA Tesla V100 PCI Express model, 32,768 mebibytes (MiB) |
| NVIDIA driver | 580.173.02 |
| Apptainer container runtime | 1.3.6 |
| Container operating system (OS) | Ubuntu 16.04.7 Long-Term Support (LTS) |
| Container CUDA | 10.0.130 |
| Container cuDNN package | 7.6.5.32 for CUDA 10.0 |
| PyTorch machine-learning framework | 1.1.0, CUDA 10.0 wheel |
| Torchvision computer-vision library | 0.2.1 |
| Python | 3.7.12 |
| PyBullet physics simulator | 3.21; application programming interface (API) 202010061 |
| ADEPT checkout | upstream `0b4aed6e128e3e67526b090a9ca514cc5fdbc333`; local repaired commit `f94459c` |
| Mask Region-Based Convolutional Neural Network (Mask R-CNN) checkout | upstream `192261db14d596da52905e91dc608bd4315552dc`; local compatibility commit `b9041ac` |

The successful arrangement is deliberately hybrid:

- source, model files, results, logs, and the container are stored on Slate;
- the Conda environment is stored under the Quartz home filesystem;
- Python is launched from that Conda environment inside the Ubuntu/CUDA 10 Apptainer image;
- GPU programs use `apptainer exec --nv` on a V100 compute node;
- CPU physics uses the same image without `--nv`;
- the ADEPT checkout is bound to `/root/adept-model` because the released code contains that absolute path.

## Paths

```text
/N/slate/jburgoon/ADEPT-Model-Release
/N/slate/jburgoon/maskrcnn-benchmark
/N/u/jburgoon/Quartz/.conda/envs/adept-model
/N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif
/N/slate/jburgoon/containers/ADEPT/cache
/N/slate/jburgoon/containers/ADEPT/tmp
/N/slate/jburgoon/containers/ADEPT/downloads
```

## Quartz resources observed during installation

This was the Slurm layout reported during installation:

| Partition | Nodes | Slurm Generic Resources (GRES) |
|---|---|---|
| `debug` | `c[1-2]` | none |
| `general` | `c[5-92]` | none |
| `interactive` | `c[3-4]` | none |
| `v100` | `g[1-24]` | four V100 GPUs per node |
| `h100-debug` | `g[37-38]` | four H100 GPUs per node |
| `h100-multi` | `g[25-36]` | four H100 GPUs per node |
| `h100-single` | `g[39-88]` | four H100 GPUs per node |

Quartz provided CUDA toolkit modules 11.7, 11.8, 12.1, 12.2, 12.5, 12.6, and 13.2, but not CUDA 10. The historical CUDA 10 user space therefore came from the SIF. IU High-Performance Computing approved this Apptainer approach and recommended the V100 partition for the old software stack.

Slate is Lustre. `df` reports the capacity of the entire filesystem and is not the user's quota. Check the user quota with:

```bash
lfs quota -h -u "$USER" /N/slate
```

The observed Slate limit was 800 GiB.

## Source checkouts

```bash
cd /N/slate/jburgoon
git clone https://github.com/JerryLingjieMei/ADEPT-Model-Release.git
cd ADEPT-Model-Release
git checkout 0b4aed6e128e3e67526b090a9ca514cc5fdbc333
git switch -c adept-quartz

cd /N/slate/jburgoon
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
git checkout 192261db14d596da52905e91dc608bd4315552dc
git switch -c adept-quartz
```

The two local branch tips in the validated installation are `f94459c` for ADEPT and `b9041ac` for Mask R-CNN. Those commits preserve the repairs described below.

### Quartz content-path configuration

The validated ADEPT checkout has one site-specific configuration change in `utils/constants.py`:

```diff
-CONTENT_FOLDER = "/root/adept-model/"
+#CONTENT_FOLDER = "/root/adept-model/"
+CONTENT_FOLDER = "/N/slate/jburgoon/ADEPT-Model-Release"
```

This records the actual Slate checkout. The runtime still binds that checkout to `/root/adept-model` because other released files retain that absolute path.

## CUDA 10 container

Load Apptainer on a compute node:

```bash
module load apptainer/1.3.6
mkdir -p /N/slate/jburgoon/containers/ADEPT/cache /N/slate/jburgoon/containers/ADEPT/tmp
export APPTAINER_CACHEDIR=/N/slate/jburgoon/containers/ADEPT/cache
export APPTAINER_TMPDIR=/N/slate/jburgoon/containers/ADEPT/tmp
```

The image was pulled from the NVIDIA GPU Cloud (NGC) container registry:

```bash
apptainer pull --docker-login /N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif docker://nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
```

`--docker-login` prompts for registry credentials. The NGC username is the literal string `$oauthtoken`; the password is an NGC API key. Do not put the key in a command, script, repository, or shell history.

Validated image, identified by the 256-bit Secure Hash Algorithm (SHA-256):

```text
size:   approximately 1.6 GiB
SHA-256: c6ca430fbf94b6974329ff20e21827690282451166825a8f1cfcf1befda687ba
```

Verify it with:

```bash
sha256sum /N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif
```

The recurring message `gocryptfs not found` is informational and did not affect execution. Lustre also produced ignorable warnings about unsupported extended attributes while the Open Container Initiative (OCI) layers were unpacked.

## Python environment

The final package state is recorded in:

- `quartz_conda_explicit.txt`: exact Conda artifacts;
- `quartz_pip_freeze.txt`: final Python package versions and the editable Mask R-CNN checkout.

Create the base environment on h1:

```bash
module load conda/26.3.2
conda create -n adept-model -c conda-forge python=3.7.12 pip=24.0 pybullet=3.21
conda activate adept-model
```

PyTorch 1.1.0 was installed from its archived CUDA 10 wheel:

```bash
python -m pip install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl torchvision==0.2.1
```

The remaining final direct dependencies were:

```text
Cython==0.29.36
easydict==1.13
imageio==2.9.0
matplotlib==3.5.3
ninja==1.11.1.1
opencv-python-headless==4.8.1.78
Pillow==6.2.2
pycocotools==2.0.7
PyYAML==6.0.1
scikit-image==0.19.3
tensorboardX==2.6.2.2
tqdm==4.68.2
yacs==0.1.8
```

Use `quartz_pip_freeze.txt` for the complete transitive package record. It is an audit manifest, not a portable requirements file: its `numpy` and `pybullet` entries contain Conda build paths, and its Mask R-CNN entry refers to an editable checkout.

Important compatibility decisions:

- `torchvision==0.3.0` caused ADEPT/Mask R-CNN data transforms to receive a Python list instead of a Mask R-CNN target object. Downgrading to the ADEPT Dockerfile's `torchvision==0.2.1` fixed this.
- New Pillow releases removed `PILLOW_VERSION`, which this torchvision release imports. `Pillow==6.2.2` is required.
- `imageio==2.31.2` requires a newer Pillow. `imageio==2.9.0` produced a consistent dependency set.
- `opencv-python` required desktop GIMP Toolkit (GTK) graphical-interface libraries unavailable in the headless container. `opencv-python-headless==4.8.1.78` was used instead.
- OpenBLAS, an open-source implementation of the Basic Linear Algebra Subprograms (BLAS), attempted excessive thread allocation on Quartz. All Apptainer commands set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

The final check returned no broken requirements:

```bash
python -m pip check
```

## Mask R-CNN compatibility repair and build

At the pinned Mask R-CNN commit, `maskrcnn_benchmark/utils/model_zoo.py` imports three helpers from `torch.utils.model_zoo`. In stable PyTorch 1.1.0 those helpers are in `torch.hub`. Local commit `b9041ac` changes only these imports:

```diff
-from torch.utils.model_zoo import _download_url_to_file
-from torch.utils.model_zoo import urlparse
-from torch.utils.model_zoo import HASH_REGEX
+from torch.hub import _download_url_to_file
+from torch.hub import urlparse
+from torch.hub import HASH_REGEX
```

Build the CUDA extension on a V100 node:

```bash
srun --partition=v100 --account=r01469 --gres=gpu:v100:1 --time=00:45:00 --cpus-per-task=4 --mem=16G --pty bash
module load apptainer/1.3.6
cd /N/slate/jburgoon/maskrcnn-benchmark
apptainer exec --nv --cleanenv --bind /N/slate/jburgoon:/N/slate/jburgoon --pwd /N/slate/jburgoon/maskrcnn-benchmark --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 /N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif /N/u/jburgoon/Quartz/.conda/envs/adept-model/bin/python setup.py build develop
```

The resulting extension was:

```text
/N/slate/jburgoon/maskrcnn-benchmark/maskrcnn_benchmark/_C.cpython-37m-x86_64-linux-gnu.so
```

A CUDA non-maximum suppression (NMS) smoke test returned indices `[0, 2]`.

## Pretrained files

The released shell script downloads over Hypertext Transfer Protocol (HTTP) and pipes directly into `tar`. For an auditable installation, the two archives were downloaded separately over HTTP Secure (HTTPS), hashed, inspected, and then extracted.

| File | Bytes | SHA-256 |
|---|---:|---|
| `ADEPT-detection-0050000.tar.gz` | 470,781,815 | `bb93d524a88fac046e03136b7ce293830c645744367a83dc6a9a402d0496ef34` |
| `ADEPT-derender-0050000.tar.gz` | 159,187,339 | `2bec33a6195806bf0492172e5f9a7126637ab431ce205a7830c6caecad79b5de` |

Installed checkpoints:

| File | SHA-256 |
|---|---|
| `output/default_detection/ADEPT-detection-0050000.pth` | `efdba23a26858422deb20df9e0f14fa6b6ffa37da3a8f6ba250d65bf8737ab1a` |
| `output/default_derender/ADEPT-derender-0050000.pth` | `b51b60050a60a51ff8c0b06fb2d6c813ae6004c193ee9ab7cafd1081a43ee63b` |

Checkpoint pointers contain:

```text
output/default_detection/ADEPT-detection-0050000.pth
output/default_derender/ADEPT-derender-0050000.pth
```

The derender stage also downloaded the standard 34-layer residual network (ResNet-34) weights:

```text
/N/u/jburgoon/Quartz/.cache/torch/checkpoints/resnet34-333f7ec4.pth
SHA-256: 333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef
```

## Runtime command pattern

GPU stages use this common prefix on a V100 node:

```bash
apptainer exec --nv --cleanenv --bind /N/slate/jburgoon:/N/slate/jburgoon --bind /N/slate/jburgoon/ADEPT-Model-Release:/root/adept-model --pwd /root/adept-model --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 /N/slate/jburgoon/containers/ADEPT/cuda10-cudnn7-devel-ubuntu16.04.sif /N/u/jburgoon/Quartz/.conda/envs/adept-model/bin/python
```

CPU stages use the same command without `--nv`. The explicit `/root/adept-model` bind satisfies the unchanged absolute paths in `tools/paths_catalog.py` and `experiments/default_physics.yaml`. The separate `CONTENT_FOLDER` adaptation in `utils/constants.py` points directly to the Slate checkout and is recorded above.

## Sample pipeline

From `/root/adept-model` inside the SIF, the released sample data were processed in this order.

### 1. Common Objects in Context (COCO) annotations

```bash
python dataset/makers/make_coco.py -i data_sample/train -o data/ann.json
python dataset/makers/make_coco.py -i data_sample/human -o data/human_ann.json
```

Validated counts:

| File | Images | Annotations | Categories |
|---|---:|---:|---:|
| `data/ann.json` | 244 | 768 | 2 |
| `data/human_ann.json` | 350 | 684 | 2 |

### 2. Detection inference

```bash
python -m tools.detection_test_net --config_file experiments/default_detection.yaml
```

Full log:

```text
logs/detection_test_20260914.log
```

Validated COCO results on the 244-image `physics` dataset are reported using average precision (AP). AP50 and AP75 use intersection-over-union thresholds of 0.50 and 0.75, respectively:

| Type | AP 0.50:0.95 | AP50 | AP75 |
|---|---:|---:|---:|
| Bounding box | 0.960983 | 0.994175 | 0.994175 |
| Segmentation | 0.951493 | 0.994175 | 0.989088 |

Detection outputs are under `output/default_detection/inference/physics` and `output/default_detection/inference/physics_human`.

### 3. Human object proposals

```bash
python dataset/makers/make_proposal.py -i data_sample/human -o data/annotated_human_ann.json -s output/default_detection/inference/physics_human/segm.json
```

The JavaScript Object Notation (JSON) file `data/annotated_human_ann.json` contains both sample cases, `human_block_0051_0` and `human_disappear_0007_3`; each maps to a four-field dictionary.

### 4. Derender inference

```bash
python -m tools.test_net --config_file experiments/default_derender.yaml
```

Full log:

```text
logs/derender_test_20260914.log
```

The run completed nine test batches in approximately 35.8 seconds on a V100 and produced:

```text
output/default_derender/inference/annotated_physics_human/human_block_0051_0.json
output/default_derender/inference/annotated_physics_human/human_disappear_0007_3.json
```

### 5. Physics inference

```bash
python -m tools.run_physics --config_file experiments/default_physics.yaml
```

The published source contains an unconditional debugging statement immediately after each worker starts:

```python
p.start()
break  #ERASE
```

It therefore launches only the first case and later fails while trying to read the nonexistent second result. This is present in upstream ADEPT commit `0b4aed6e...`. Local commit `f94459c` removes only `break  #ERASE`.

The original first-case run and repaired second-case run are retained in:

```text
logs/physics_test_20260914.log
logs/physics_test_disappear_20260915.log
```

The second case was selected without repeating the completed first case:

```bash
python -m tools.run_physics --config_file experiments/default_physics.yaml --start_index 1 --stride 2
```

Validated results:

| Case | Sum | Mean | Maximum |
|---|---:|---:|---:|
| `human_block_0051_0` | 81.6079102236 | 0.4800465307 | 3.8191481734 |
| `human_disappear_0007_3` | 71.9334155115 | 0.4231377383 | 2.7488627169 |

Result files:

```text
output/default_physics/results/human_block_0051_0.json       1,635,638 bytes
output/default_physics/results/human_disappear_0007_3.json   2,177,365 bytes
```

## Validation summary

The tested installation completed all released sample inference stages:

1. CUDA 10 and cuDNN loaded inside the SIF on a Quartz V100.
2. PyTorch reported CUDA available and completed a CUDA tensor operation.
3. The Mask R-CNN CUDA extension compiled and passed CUDA NMS.
4. COCO annotations were generated for the supplied training and human samples.
5. Detection inference completed and produced bounding-box and segmentation metrics.
6. Human object proposals were generated for both supplied cases.
7. Derender inference completed for both cases.
8. Physics inference produced result JSON and visualization frames for both cases after removal of the upstream debugging `break`.

This establishes a working ADEPT inference installation on Quartz. It does not independently reproduce ADEPT training or the full published experimental dataset.

## Routine Slurm commands

CPU interactive allocation:

```bash
srun --partition=interactive --account=r01469 --time=00:30:00 --cpus-per-task=4 --mem=16G --pty bash
```

V100 interactive allocation:

```bash
srun --partition=v100 --account=r01469 --gres=gpu:v100:1 --time=00:45:00 --cpus-per-task=4 --mem=16G --pty bash
```

Queue status:

```bash
squeue -u "$USER" -o "%.18i %.12P %.20j %.2t %.10M %.10l %R"
```

Estimated start time:

```bash
squeue --start -j JOB_ID -o "%.18i %.12P %.2t %.19S %R"
```

Short interactive requests may backfill much sooner than long requests.

## Reproducibility cautions

- Preserve the SIF and verify its SHA-256 before use. Historical registry tags can disappear.
- Preserve the two pretrained archives and their hashes. The upstream download script neither uses HTTPS nor verifies content.
- Preserve the local `adept-quartz` branches or export their commits as patch files before moving the installation.
- Do not expose the NGC API key in documentation, Git, terminal output, or job logs.
- Load `apptainer/1.3.6` in each new compute shell; module state is not reliably inherited.
- Run GPU work on a V100 unless a separate H100 compatibility test is performed and documented.
- Treat the Conda and pip manifests as records of the validated state. Re-run `python -m pip check` after any package change.
