# SR3 Noise-Schedule Ablation on Kvasir-SEG

CSL7860 Foundation Models and Generative AI, IIT Jodhpur (Dr. Angshuman Paul).  
Group: Rikin Patel (M26AI1096), Balasubramanian M (M26AI1030), Sonalika Sharma (M26AI1120), Anshul (M26AI1020).

We fine-tune the pretrained **SR3** 16×16 → 128×128 model on **Kvasir-SEG** colonoscopy frames. Only the noise schedule changes between Run A and Run B:

| Run | Weights | Schedule (train + sample) |
|---|---|---|
| Run 0 | `I640000_E37_gen.pth`, no fine-tuning | $\theta_t = \beta_t$ |
| Run A | fine-tuned | $\theta_t = \beta_t$ |
| Run B | fine-tuned | $\theta_t^{x}$, **$x = 2.50$** (Phase 1 declaration) |

The exponent is declared once as `NOISE_EXPONENT_X = 2.50` in [`scripts/finetune.py`](scripts/finetune.py).

## Layout

```text
configs/base.toml       experiment settings
scripts/finetune.py     Run A / Run B entry point
src/sr3_ablation/       our pipeline (data, schedule, training, evaluation)
third_party/sr3/        upstream SR3 @ 01d27a7 (modified diffusion schedule only)
report/build/report.pdf written report (included in the submission zip)
outputs/                run logs, samples, trajectories, metrics (generated)
```

Upstream change: marked `# ADDED` lines in `third_party/sr3/model/sr3_modules/diffusion.py` apply the declared power transform to the noise schedule. See [`third_party/sr3/UPSTREAM.md`](third_party/sr3/UPSTREAM.md).

## Environment

Python 3.11+, PyTorch with CUDA (tested on RTX 5070, PyTorch 2.14+cu130).

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[lpips,pdf]"
```

Point large folders outside OneDrive if needed: copy `configs/local.example.toml` → `configs/local.toml` (git-ignored).

## Reproduce the runs

From the repo root (default config `configs/base.toml`):

```bash
python -m sr3_ablation download-checkpoint
python -m sr3_ablation prepare-data
python -m sr3_ablation verify-schedule --run B

python -m sr3_ablation run0
python scripts/finetune.py --run A
python scripts/finetune.py --run B
python -m sr3_ablation evaluate --run 0
python -m sr3_ablation evaluate --run A
python -m sr3_ablation evaluate --run B
python -m sr3_ablation trajectories --run A
python -m sr3_ablation trajectories --run B
python -m sr3_ablation diagnose-start --run B
```

**Split:** seed 42 → train 800, val 100, test 100 (same order as teammates).  
**Preprocessing:** centre-crop → 128 HR → 16 LR → bicubic 128 conditioning.  
**Training:** Adam, lr `3e-5`, batch 8, 60 epochs, checkpoints at 12/24/36/48/60.  
**Metrics:** PSNR, SSIM, LPIPS on the test split.

## Submission

Upload **`report/build/report.pdf`** plus the experiment **code** and **`outputs/`** artefacts for Runs 0, A and B as required by the Phase 2 brief (loss logs, checkpoint samples, trajectories, test metrics, schedule evidence). Assemble the folder or zip the way your instructor specifies.

The PDF and any zip are prepared **on your machine only**; report-builder and packager scripts live under paths listed in `.gitignore` and are **not** pushed to GitHub.

## Licences

Upstream SR3: Apache-2.0. Kvasir-SEG: research/education use; downloaded by `prepare-data`, not redistributed. Pretrained weights: SR3 authors' release.
