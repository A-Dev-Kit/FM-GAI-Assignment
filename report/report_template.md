<!-- ADDED: report template. `python -m sr3_ablation report` fills every double-brace placeholder from
     outputs/ and writes report/build/report.{md,html,pdf}. Text marked [TEAM: ...] must be
     written by the team after the full runs; delete the markers before submitting. -->

# Noise Schedule Ablation in SR3: $\theta_t$ vs $\theta_t^{x}$ on AFHQ

**CSL7860 Foundation Models and Generative AI · IIT Jodhpur · Instructor: Dr. Angshuman Paul**

**Group:** Rikin [full name, roll no.] · Sonalika Sharma (M26AI1120) · Anshul (M26AI1020) · Bala [full name, roll no.]

## 1. Task and Model

**Task: 8× super-resolution of animal faces.** A 16×16 RGB image is mapped to a 128×128 RGB image. The 16×16 input is a bicubic downsample of the 128×128 ground truth; as SR3 requires, it is bicubically upsampled back to 128×128 and used as the conditioning image.

**Model: SR3** (Saharia et al., *Image Super-Resolution via Iterative Refinement*, IEEE TPAMI 2022, [arXiv:2104.07636](https://arxiv.org/abs/2104.07636)), using the unofficial PyTorch implementation by Janspiry pinned to commit [`01d27a7`](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d) and its released checkpoint `I640000_E37_gen.pth` (16→128, trained on FFHQ). SR3 is a conditional DDPM in pixel space: at every step the UNet receives the conditioning image concatenated with $x_t$ and the continuous noise level $\sqrt{\bar\alpha}$, and predicts the noise $\epsilon$.

## 2. Dataset and Verification

**AFHQ v1** (Choi et al., *StarGAN v2*, CVPR 2020; CC BY-NC 4.0): 15,000 animal faces at 512×512 in three classes (cat, dog, wild). Our split uses seed 42; the validation images come from AFHQ `train/`, stratified by class, and the test images are 100 per class from AFHQ's official `val/` split:

{{DATASET_TABLE}}

Each image is resized (bicubic, centre crop) to 128×128 (target) and 16×16 (input), and the input is upsampled back to 128×128, following upstream `data/prepare_data.py`. The full manifest is in `outputs/dataset/manifest.csv`.

**Evidence that AFHQ was not used for pretraining (route (a), stated pretraining data):**

- Training config [`config/sr_sr3_16_128.json` lines 17–19](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d/config/sr_sr3_16_128.json#L17-L19): training set `"name": "FFHQ"`, `"dataroot": "dataset/ffhq_16_128"`; line 29: validation set `"CelebaHQ"`.
- [README line 24](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/01d27a7cbfa8502be1d8dbd4ee02fcbd5e44389d/README.md?plain=1#L24): the checkpoint is "16×16 → 128×128 on FFHQ-CelebaHQ".
- Paper, [arXiv:2104.07636v2](https://arxiv.org/pdf/2104.07636v2), page 5, right column, lines 11–13: "training face super-resolution models on Flickr-Faces-HQ (FFHQ) and evaluating on CelebA-HQ".

FFHQ and CelebA-HQ contain only human faces, so AFHQ's animal faces (fur, eye and ear geometry, colour statistics) are out of the pretraining distribution.

## 3. Schedule Analysis

SR3's native noise parameter is $\theta_t = \beta_t$, the per-step variance of the forward process

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\big), \qquad \bar\alpha_t = \prod_{s=1}^{t}(1-\beta_s), \qquad \mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t},$$

with a linear schedule from $\beta_1 = 10^{-6}$ to $\beta_T = 10^{-2}$ and $T = 2000$. Every $\beta_t$ lies in $(0, 1)$, so no rescaling is needed. Run B uses $\theta'_t = \beta_t^{x}$ with $x = {{EXPONENT}}$.

**Direction of the change.** Because $0 < \beta_t < 1$ and $x > 1$, $\beta_t^{x} < \beta_t$ for every $t$. Each factor $1 - \beta_t^{x}$ is larger, so $\bar\alpha'_t > \bar\alpha_t$ and $\mathrm{SNR}'(t) > \mathrm{SNR}(t)$ at **every** timestep: the modified forward process destroys much less signal.

{{SCHEDULE_TABLE}}

**Consequence at $t = T$.** Under $\theta_t$, $\bar\alpha_T \approx 4\times10^{-5}$ and $x_T$ is essentially pure noise, matching the $\mathcal{N}(0, I)$ start of sampling. Under $\theta_t^{x}$, $\bar\alpha'_T \approx 0.62$: $x_T = 0.78\,x_0 + 0.62\,\epsilon$ still carries most of the image, and the SNR never falls below 1. Sampling, however, still starts from pure noise, so Run B begins from a state the fine-tuned model never sees in training, and its noise-level input $\sqrt{\bar\alpha'_t}$ only covers $[0.78, 1]$ instead of $[0.007, 1]$.

**SNR at the saved trajectory timesteps:**

{{TRAJECTORY_SNR_TABLE}}

{{SCHEDULE_FIGURE}}

**Predictions to test.** (i) Run B's training loss is lower than Run A's, because noise is easier to predict at high SNR. (ii) Run B's reverse process cannot remove the initial noise: each step's variance $\beta_t^{x}$ is tiny, so we expect visible residual noise or colour shifts in its outputs and lower PSNR/SSIM than Run A. (iii) Run B's early trajectory states change little, while Run A moves from noise to structure around $t \approx 500$, where its SNR crosses 1.

## 4. Experimental Setup

{{SETUP_TABLE}}

Runs A and B start from the same checkpoint, use the same data order and seeds, and differ **only** in the noise schedule (training and sampling). Run 0 applies the pretrained checkpoint unchanged with the original schedule. Test outputs use the same sampling seed per batch for every run, so all runs start from identical initial noise for each image.

**Environment:** {{ENVIRONMENT}}

## 5. Implementation

The upstream code is vendored in `third_party/sr3/`; the **only** change is one line in `model/sr3_modules/diffusion.py` (`set_new_noise_schedule`), marked `# ADDED`:

```python
betas = betas ** schedule_opt.get(
    "exponent", 1.0
)  # ADDED: noise-schedule ablation, theta_t -> theta_t ** x
```

All buffers derived from `betas` (including `sqrt_alphas_cumprod_prev`, which drives the continuous noise level in training and sampling) therefore use the modified schedule. Everything else is new code in `src/sr3_ablation/` (each file starts with an `# ADDED` header). The exponent is declared once as `NOISE_EXPONENT_X = {{EXPONENT}}` at the top of `scripts/finetune.py`.

- **Weights only.** We load only the UNet weights (`denoise_fn.*`) from the checkpoint and install the schedule *afterwards*. Upstream `load_network` would also restore the optimiser, step counters and the schedule buffers stored in the checkpoint, which would silently undo the ablation.
- **Independent verification.** `NoiseSchedule` recomputes $\theta_t^{x}$, $\bar\alpha_t$ and SNR in float64, independently of SR3, and compares them with the buffers read back from the model before and after fine-tuning and again before sampling (`schedule.json`, `schedule_sampling.json`):

{{SCHEDULE_EVIDENCE}}

- **Sampling.** A batched reverse sampler calls upstream `p_sample` for $t = T, \dots, 1$ and notifies observers of each state $x_t$; the trajectory recorder keeps the requested timesteps.

## 6. Results

### 6.1 Training loss

{{LOSS_FIGURE}}

{{LOSS_TABLE}}

{{VALIDATION_FIGURE}}

### 6.2 Samples during fine-tuning

{{SNAPSHOT_FIGURES}}

### 6.3 Test set

{{RESULTS_TABLE}}

{{COMPARISON_FIGURE}}

### 6.4 Reverse trajectories

{{TRAJECTORY_FIGURE}}

## 7. Analysis

[TEAM: compare the results with the three predictions in Section 3. Address at least: (a) whether Run B's lower training loss translates into better or worse samples, and why (train/test mismatch at $x_T$, the narrower noise-level range seen in training); (b) what the trajectories show about when structure appears in Run A vs Run B, using the SNR table; (c) how much fine-tuning alone helps (Run 0 vs Run A); (d) any failure cases in the comparison grid, such as residual noise, colour shifts or over-smoothing.]

## 8. Takeaway

[TEAM: two or three sentences on what this ablation shows about how sensitive SR3 is to the noise schedule, and when raising $\theta_t$ to a power would or would not be advisable.]

## 9. Contribution Statement

| Member | Contribution |
|---|---|
| Rikin | [TEAM: contribution] |
| Sonalika Sharma (M26AI1120) | [TEAM: contribution] |
| Anshul (M26AI1020) | [TEAM: contribution] |
| Bala | [TEAM: contribution] |
