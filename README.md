# SR3 Noise-Schedule Ablation on AFHQ

<!-- ADDED: project README. -->

CSL7860 Foundation Models and Generative AI, IIT Jodhpur (Dr. Angshuman Paul). Group: Rikin,
Sonalika Sharma, Anshul, Bala.

We fine-tune the pretrained **SR3** 16×16 → 128×128 super-resolution diffusion model on
**AFHQ** animal faces twice, changing only the noise schedule:

| Run | Weights | Noise schedule (training and sampling) |
|---|---|---|
| Run 0 | pretrained `I640000_E37_gen.pth`, no fine-tuning | $\theta_t = \beta_t$ (original) |
| Run A | fine-tuned on AFHQ | $\theta_t = \beta_t$ (original) |
| Run B | fine-tuned on AFHQ | $\theta_t^{x}$ with **$x = 1.60$** |

$\beta_t$ is linear from $10^{-6}$ to $10^{-2}$ over $T = 2000$ steps. The exponent is declared once,
as `NOISE_EXPONENT_X = 1.60` at the top of [`scripts/finetune.py`](scripts/finetune.py).

## Repository layout

```text
Solutions/
├── configs/            base.toml (full experiment), smoke.toml (short test), local.example.toml
├── scripts/finetune.py fine-tuning entry point; declares NOISE_EXPONENT_X
├── src/sr3_ablation/   all new code (every file starts with an "# ADDED" header)
│   ├── config/         TOML -> frozen dataclasses, validation, layered merging
│   ├── schedule/       transform strategies (identity, power), NoiseSchedule, verification
│   ├── data/           AFHQ download, seeded manifest split, SR3 16/128 layout, DataLoader
│   ├── model/          interfaces, Sr3Backend adapter (only importer of third_party), checkpoints
│   ├── training/       FineTuner loop + callbacks (loss CSV, snapshots, checkpoints, evidence)
│   ├── sampling/       batched reverse sampler with observers, trajectory recorder
│   ├── evaluation/     PSNR / SSIM / LPIPS registry, test-set evaluator
│   ├── reporting/      plots, image grids, tables, Markdown -> HTML -> PDF, report builder
│   ├── pipelines/      one use case per CLI command
│   ├── io/, utils/     naming rules, image/JSON helpers, seeding, device, env capture, logging
│   └── cli.py          python -m sr3_ablation <command>
├── third_party/sr3/    upstream SR3 @ 01d27a7 (Apache-2.0) with ONE added line, see UPSTREAM.md
├── report/             report_template.md (brief's section order) and build_report.py
├── notebooks/          kaggle_runner.ipynb (one run per 12-hour Kaggle session)
├── tests/              unit tests + CPU end-to-end test on a tiny UNet
└── outputs/            generated artefacts (git-ignored)
```

### The single change to SR3

`third_party/sr3/model/sr3_modules/diffusion.py`, in `set_new_noise_schedule`:

```python
betas = betas ** schedule_opt.get('exponent', 1.0)  # ADDED: noise-schedule ablation, theta_t -> theta_t ** x
```

Every derived buffer (cumulative products, posterior coefficients and the continuous noise level
fed to the UNet) is computed from `betas` after this line, so training and sampling both use the
modified schedule. Without an `exponent` key the code behaves exactly like upstream.
[`third_party/sr3/UPSTREAM.md`](third_party/sr3/UPSTREAM.md) has the diff and how to reproduce it.

Two safeguards make sure the ablation really runs with the declared schedule:

1. **Weights only.** Only the UNet (`denoise_fn.*`) weights are taken from a checkpoint, and the
   schedule is installed afterwards. Upstream's `load_network` would also restore the
   optimiser, the step counters and the schedule buffers saved in the checkpoint.
2. **Independent check.** `NoiseSchedule` recomputes the schedule in float64 without SR3 and is
   compared with the model's buffers before and after training and before sampling. The result is
   written to `schedule.json` and `schedule_sampling.json`, and any mismatch stops the run.

## Environment

Tested on Windows 11 with Python **3.14.5**, PyTorch **2.14.1+cu130** and an NVIDIA **RTX 5070
Laptop** GPU (8 GB). Blackwell GPUs need CUDA 12.8 or newer wheels. Python 3.11 or newer is
required because the config loader uses `tomllib`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[dev,pdf]"      # add ,lpips for the LPIPS metric
```

On Kaggle or Colab torch is preinstalled: `pip install -e . gdown` is enough (see the notebook).

**Large files.** Data (about 3 GB once processed) and the 391 MB checkpoint default to
`data/` and `checkpoints/` inside the repo. If the repo lives in a cloud-synced folder (OneDrive,
Dropbox), copy `configs/local.example.toml` to `configs/local.toml` and point those paths
elsewhere; `local.toml` is git-ignored and merged automatically. The same applies to the virtual
environment (about 5 GB).

## Running the experiment

All commands run from the repo root. Each accepts `--config FILE` (repeatable, merged in order;
default `configs/base.toml`).

```bash
python -m sr3_ablation download-checkpoint     # I640000_E37_gen.pth from the authors' Google Drive
python -m sr3_ablation prepare-data            # downloads AFHQ (730 MB), seeded split, 16/128 layout
python -m sr3_ablation verify-schedule --run B # quick check: model buffers == theta_t ** 1.60

python -m sr3_ablation run0                    # Run 0: pretrained model on the test split (+ bicubic)
python scripts/finetune.py --run A             # Run A: fine-tune with theta_t
python scripts/finetune.py --run B             # Run B: fine-tune with theta_t ** x
python -m sr3_ablation evaluate --run A
python -m sr3_ablation evaluate --run B
python -m sr3_ablation trajectories --run A
python -m sr3_ablation trajectories --run B
python -m sr3_ablation report                  # report/build/report.{md,html,pdf}
```

`python -m sr3_ablation smoke-test` runs everything briefly (`configs/smoke.toml`: 200 training
iterations, 8 test images) and writes timing estimates to `outputs/smoke/timings.json`.

### Reproducibility

- **Data split** (seed 42, `outputs/dataset/manifest.csv`): 14,130 train images and 500
  validation images (stratified by class) from AFHQ `train/`, and 300 test images (100 per class)
  from AFHQ's official `val/` split.
- **Seeds:** training 42, sampling 1234. Test batch *k* uses seed 1234 + *k*, so Runs 0, A and B
  start from identical noise for each test image. Validation, snapshot and trajectory images are
  fixed and spread evenly over the three classes.
- **Recorded per run:** `resolved_config.json`, `environment.json` (versions, GPU, git commit),
  `run.json` (run identity, exponent and per-epoch results) and `log.txt`.
- Runs A and B share every hyperparameter: Adam with learning rate 1e-5, batch 8, 10 epochs,
  L1 noise loss and horizontal-flip augmentation.

### What each run produces (`outputs/<run>/`)

| File | Content | Brief item |
|---|---|---|
| `loss.csv` | columns `epoch,train_loss` | per-epoch training loss |
| `samples/runB_epoch04_sample1.png` … | 3 fixed validation images at epochs 2, 4, 6, 8, 10 | samples at 5 equally spaced epochs |
| `trajectories/runA_t1600_sample1.png` … | $x_t$ at $t$ = 1600, 1200, 800, 400, 0 for 3 test images | reverse-process trajectory |
| `trajectories/trajectory.json` | image ids, $\bar\alpha_t$ and SNR for each saved $t$ | trajectory labels |
| `validation.csv` | validation PSNR / SSIM at checkpoint epochs | model selection evidence |
| `test_outputs/`, `test_metrics.csv`, `test_metrics.json` | super-resolved test images and scores | results table |
| `schedule.json`, `schedule_sampling.json` | verified schedule buffers | proof that Run B used $\theta_t^{x}$ |
| `checkpoints/epochNN_gen.pth`, `final_gen.pth` | UNet weights | |

Trajectory timesteps follow DDPM's 1-indexed convention: $x_{2000}$ is the starting noise, the
reverse step with upstream index $i$ produces $x_i$, and $x_0$ is the final image.

## Measured on the RTX 5070 (smoke test)

Measured with `python -m sr3_ablation smoke-test` on an RTX 5070 Laptop GPU (8 GB), Python 3.14.5,
PyTorch 2.14.1+cu130, on 6 October 2026. The raw numbers are in `outputs/smoke/timings.json`.

| Quantity | Value |
| --- | --- |
| Training batch size 8 fits in memory | yes, no gradient accumulation needed |
| Training speed | 0.374 s per iteration (200 iterations in 75 s) |
| Peak GPU memory while training | 5.1 GB |
| Full 2000-step reverse sampling, one batch of 8 images | 215 to 241 s |
| One trajectory sample (2000 steps, 5 saved states) | 51 s |

These give the following estimates for the full configuration in `configs/base.toml`
(14,130 training images, 1,767 iterations per epoch, 10 epochs, 300 test images):

| Stage | Estimated GPU hours |
| --- | --- |
| Fine-tuning, per run | 1.8 |
| Validation sampling at the 5 checkpoint epochs, per run | 1.0 |
| Test-set evaluation, per run | 2.6 |
| Total for Run 0, Run A and Run B | about 13.3 |

Sampling dominates the cost. Runs A and B are independent, so two teammates can run them at the
same time on separate GPUs (local or Kaggle), which roughly halves the wall-clock time.

## Tests and code quality

```bash
python -m pytest            # unit tests + tiny-UNet end-to-end pipeline on the CPU (~25 s)
ruff format --check . && ruff check .
```

The integration test prepares a synthetic AFHQ tree, fine-tunes Runs A and B, evaluates Runs
0, A and B, records trajectories and builds the report with a 10-step tiny SR3 UNet. Unit tests
cover the schedule numbers declared in Phase 1 ($\bar\alpha'_T = 0.6153$, $\mathrm{SNR}'_T = 1.60$,
original SNR first below 1 at $t = 527$). They also check that the vendored line applies the power,
and that loading an upstream-style checkpoint never restores its stored schedule.

## Design notes

- **Single responsibility:** each package does one job; the `pipelines/` modules only compose
  them for one command.
- **Open/closed:** new schedule transforms implement `ScheduleTransform`; new metrics are
  registered in `METRIC_REGISTRY`; new artefacts are added as `TrainingCallback`,
  `ValidationSink` or `StepObserver` implementations, without touching the training loop or the
  sampler.
- **Interface segregation and dependency inversion:** the trainer depends on `TrainableModel`,
  the sampler on `Denoiser`, verification on `ScheduleHolder` and checkpointing on `WeightStore`
  (`model/protocols.py`). Only `model/sr3_backend.py` knows about the upstream classes.
- **Immutable configuration:** frozen dataclasses with validation; the fully resolved config
  is saved with every run.

## Licences

- Upstream SR3 code: Apache-2.0 (`third_party/sr3/LICENSE`).
- AFHQ: CC BY-NC 4.0 (Choi et al., *StarGAN v2*, CVPR 2020), used for non-commercial coursework.
- Pretrained weights: released by the SR3 implementation's author.
